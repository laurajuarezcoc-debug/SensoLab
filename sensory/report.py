"""Reporte PDF descargable (fpdf2 + fuentes DejaVu incluidas en matplotlib)."""
from __future__ import annotations

import io
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.fonts import FontFace

from .constants import TEST_TYPES
from .interpret import fmt
from .methods.catalog import AnalysisParams, MethodResult
from .recommender import STATUS_LABEL, Recommendation
from .validation import ERROR, INFO, WARNING, Issue

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
BLUE = (42, 120, 214)
INK = (11, 11, 11)
INK2 = (82, 81, 78)
SOFT = (244, 243, 240)
LEVEL_COLOR = {ERROR: (196, 49, 47), WARNING: (201, 133, 0), INFO: (82, 81, 78)}
LEVEL_NAME = {ERROR: "Error", WARNING: "Advertencia", INFO: "Info"}


def _plain(text: str) -> str:
    """Quita marcas de markdown no soportadas y normaliza saltos."""
    text = re.sub(r"(?<!\*)\*(?!\*)", "", text or "")
    return text.replace("⚠️", "(!)").replace(" ", " ")


def format_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return "—"
        if v.is_integer() and abs(v) < 1e6:
            return str(int(v))
        if abs(v) < 0.001 and v != 0:
            return "< 0,001"
        return fmt(v, 3 if abs(v) < 1 else 2)
    return str(v)


def _mc(pdf: FPDF, w, h, text, **kw) -> None:
    """multi_cell que siempre regresa al margen izquierdo en la línea siguiente."""
    kw.setdefault("new_x", "LMARGIN")
    kw.setdefault("new_y", "NEXT")
    pdf.multi_cell(w, h, text, **kw)


class _PDF(FPDF):
    title_text = ""

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*INK2)
        self.cell(0, 6, self.title_text, align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*INK2)
        self.cell(0, 6, f"Página {self.page_no()} de {{nb}}", align="C")


def _setup() -> _PDF:
    pdf = _PDF(format="Letter")
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", str(FONT_DIR / "DejaVuSans-Oblique.ttf"))
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(16, 14, 16)
    pdf.alias_nb_pages()
    return pdf


def _h1(pdf: FPDF, text: str) -> None:
    pdf.ln(3)
    pdf.set_font("DejaVu", "B", 14)
    pdf.set_text_color(*BLUE)
    _mc(pdf, 0, 7, text)
    pdf.set_draw_color(*BLUE)
    pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.r_margin, pdf.get_y() + 1)
    pdf.ln(4)


def _h2(pdf: FPDF, text: str) -> None:
    if pdf.get_y() > pdf.h - 50:
        pdf.add_page()
    pdf.ln(2)
    pdf.set_font("DejaVu", "B", 12)
    pdf.set_text_color(*INK)
    _mc(pdf, 0, 6, text)
    pdf.ln(1)


def _p(pdf: FPDF, text: str, size: int = 10, color=INK, style: str = "") -> None:
    pdf.set_font("DejaVu", style, size)
    pdf.set_text_color(*color)
    _mc(pdf, 0, 5.2, _plain(text), markdown=True)
    pdf.ln(1.5)


def _box(pdf: FPDF, label: str, text: str, fill=SOFT) -> None:
    pdf.set_fill_color(*fill)
    pdf.set_font("DejaVu", "B", 10)
    pdf.set_text_color(*BLUE)
    _mc(pdf, 0, 6, label, fill=True)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(*INK)
    _mc(pdf, 0, 5.2, _plain(text), fill=True, markdown=True)
    pdf.ln(2)


def _table(pdf: FPDF, caption: str, df: pd.DataFrame, max_rows: int = 40) -> None:
    if df is None or df.empty:
        return
    df = df.head(max_rows)
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()
    pdf.set_font("DejaVu", "B", 9)
    pdf.set_text_color(*INK2)
    _mc(pdf, 0, 5, caption)
    n_cols = len(df.columns)
    size = 8 if n_cols <= 6 else 7 if n_cols <= 9 else 6
    pdf.set_font("DejaVu", "", size)
    pdf.set_text_color(*INK)
    head = FontFace(emphasis="BOLD", fill_color=(228, 236, 248))
    with pdf.table(headings_style=head, line_height=size * 0.55, text_align="CENTER",
                   borders_layout="HORIZONTAL_LINES", cell_fill_color=(250, 250, 248),
                   cell_fill_mode="ROWS") as table:
        row = table.row()
        for c in df.columns:
            row.cell(str(c))
        for _, r in df.iterrows():
            row = table.row()
            for v in r.values:
                row.cell(format_cell(v))
    pdf.ln(3)


def _figure(pdf: FPDF, fig) -> None:
    if fig is None:
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    w_px, h_px = fig.get_size_inches() * 150
    width = min(pdf.w - pdf.l_margin - pdf.r_margin, 150)
    height = width * h_px / w_px
    if pdf.get_y() + height > pdf.h - 16:
        pdf.add_page()
    x = (pdf.w - width) / 2
    pdf.image(buf, x=x, w=width)
    pdf.ln(3)


def build_report(*, product: str, test_type: str, author: str, design_text: str,
                 issues: list[Issue], recs: list[Recommendation], selected: list[str],
                 results: list[MethodResult], params: AnalysisParams, conclusion: str,
                 data_preview: pd.DataFrame | None = None) -> bytes:
    pdf = _setup()
    pdf.title_text = f"Análisis sensorial · {product}"
    pdf.add_page()

    # Portada / encabezado
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(*INK)
    _mc(pdf, 0, 10, "Reporte de análisis sensorial")
    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(*BLUE)
    _mc(pdf, 0, 7, product)
    pdf.ln(2)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(*INK2)
    meta = [f"Tipo de prueba: {TEST_TYPES[test_type]}",
            f"Nivel de significancia: α = {fmt(params.alpha)}",
            f"Fecha: {datetime.now():%d/%m/%Y %H:%M}"]
    if author:
        meta.insert(0, f"Elaborado por: {author}")
    for m in meta:
        _mc(pdf, 0, 5, m)
    pdf.ln(2)

    _box(pdf, "Conclusión y decisión", conclusion, fill=(232, 241, 252))

    _h1(pdf, "1. Diseño experimental y datos")
    _p(pdf, design_text)
    if data_preview is not None:
        _table(pdf, "Resumen por muestra", data_preview)

    _h1(pdf, "2. Revisión de calidad de los datos")
    for it in issues:
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_text_color(*LEVEL_COLOR[it.level])
        _mc(pdf, 0, 5, f"[{LEVEL_NAME[it.level]}] {_plain(it.title)}")
        if it.detail:
            _p(pdf, it.detail, size=9, color=INK2)

    _h1(pdf, "3. Métodos recomendados y seleccionados")
    for r in recs:
        if r.status == "no_aplicable" and r.method_id not in selected:
            continue
        mark = "✔ " if r.method_id in selected else "   "
        pdf.set_font("DejaVu", "B", 9.5)
        pdf.set_text_color(*INK)
        _mc(pdf, 0, 5, f"{mark}{r.name} — {STATUS_LABEL[r.status]}")
        _p(pdf, f"{r.reason} Qué hace: {r.what} Cómo se decide: {r.decides}.", size=9, color=INK2)

    _h1(pdf, "4. Resultados e interpretación")
    for res in results:
        _h2(pdf, res.title)
        if res.error:
            _p(pdf, f"No se pudo calcular: {res.error}", color=LEVEL_COLOR[ERROR])
            continue
        if res.question:
            _p(pdf, f"Pregunta: {res.question}", size=9, color=INK2, style="I")
        for cap, df in res.tables:
            _table(pdf, cap, df)
        for fig in res.figures:
            _figure(pdf, fig)
        _box(pdf, "Interpretación", res.interpretation)
        _box(pdf, "Decisión para el producto", res.decision, fill=(232, 245, 236))
        for n in res.notes:
            _p(pdf, f"Nota: {n}", size=8, color=INK2)

    _h1(pdf, "5. Validación")
    _p(pdf, "Los valores críticos (mínimo de la tabla binomial, F, t, χ² y rango studentizado de Tukey) se "
            "calcularon con las distribuciones exactas. Para validar, compare con las tablas del curso o con el "
            "cálculo manual en Excel usando los datos depurados.", size=9, color=INK2)
    return bytes(pdf.output())
