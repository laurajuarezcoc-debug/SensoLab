"""SensoLab · App de análisis sensorial automático (Streamlit).

Flujo: 1) cargar datos → 2) tipo de prueba → 3) revisión de datos →
4) métodos recomendados → 5) resultados interpretados → 6) reporte PDF.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from sensory import plots
from sensory.constants import (AFECTIVA, DESCRIPTIVA, DISCRIMINATIVA, ESCALA, PROTOCOLS, RANGO,
                               TEST_TYPES)
from sensory.io_utils import DataLoadError, LoadResult, combine_ballots, read_table
from sensory.methods.catalog import AnalysisParams, detect_protocol
from sensory.pipeline import overall_conclusion, run_methods, sample_summary
from sensory.recommender import (NO_APLICA, RECOMENDADO, STATUS_LABEL, describe_design, recommend,
                                 suggest_test_type)
from sensory.report import build_report, format_cell
from sensory.schema import (ColumnSpec, build_profile, guess_spec, robust_range, suggest_scale,
                            to_long)
from sensory.validation import ERROR, INFO, WARNING, prepare

logging.basicConfig(level=logging.INFO)
logging.getLogger("fontTools").setLevel(logging.WARNING)
EXAMPLES = Path(__file__).parent / "data" / "ejemplos"
EXAMPLE_LABELS = {
    "afectiva_hedonica_3_muestras.csv": "Afectiva · hedónica 9 pts, 3 niveles de azúcar (ANOVA + Tukey)",
    "afectiva_2_muestras_ancho.csv": "Afectiva · DNP vs. comercial, formato ancho (t de Student)",
    "afectiva_preferencia_pareada.csv": "Afectiva · preferencia pareada 28/40 (binomial)",
    "afectiva_compra_si_no_cochran.csv": "Afectiva · ¿lo compraría? sí/no, 3 prototipos (Q de Cochran)",
    "afectiva_con_errores.csv": " Afectiva con errores de captura (demostración de alertas)",
    "discriminativa_triangular.csv": "Discriminativa · triangular, 16/30 aciertos (binomial)",
    "discriminativa_a_no_a.csv": "Discriminativa · A – no A (Ji² 2×2)",
    "ordenamiento_friedman.csv": "Ordenamiento · 4 galletas por dulzor (Friedman)",
    "descriptiva_qda.csv": "Descriptiva · QDA 4 galletas × 5 atributos (ANOVA, PCA)",
}
LEVEL_UI = {ERROR: st.error, WARNING: st.warning, INFO: st.info}

st.set_page_config(page_title="SensoLab · Análisis sensorial", page_icon=":material/lab_panel:", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.step {font-size: .8rem; font-weight: 700; letter-spacing: .08em; color: #d14996; text-transform: uppercase;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;}
.b-rec {background:#e1effd; color:#1d5fae;} .b-alt {background:#eef6ee; color:#1a7f37;}
.b-comp {background:#f4f3f0; color:#52514e;} .b-no {background:#f9eaea; color:#9b2c2b;}
.decision {border-left: 4px solid #1a7f37; background:#eef6ee; padding:.8rem 1rem; border-radius:6px;}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Utilidades de interfaz
# ==========================================================================
@st.cache_data(show_spinner=False)
def _read_cached(raw: bytes, name: str) -> LoadResult:
    return read_table(raw, name)


def show_df(df: pd.DataFrame, **kw) -> None:
    """Muestra tablas con coma decimal y p < 0,001, igual que en el PDF."""
    st.dataframe(df.map(format_cell) if not df.empty else df, hide_index=True, use_container_width=True, **kw)


def step(n: int, title: str, help_text: str = "") -> None:
    st.markdown(f"<div class='step'>Paso {n}</div>", unsafe_allow_html=True)
    st.subheader(title)
    if help_text:
        st.caption(help_text)


def reset_results() -> None:
    old = st.session_state.pop("results", None)
    if old:
        plots.close_all([f for r in old["results"] for f in r.figures])


def examples_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted(EXAMPLES.glob("*.csv")):
            z.write(f, f.name)
    return buf.getvalue()


MANUAL_TEMPLATES = {
    "Notas por juez y muestra (hedónica / intensidad)": pd.DataFrame(
        {"juez": ["J1", "J1", "J1", "J2", "J2", "J2"], "muestra": ["A", "B", "C"] * 2,
         "calificacion": [7, 5, 8, 6, 5, 7]}),
    "Aciertos de una prueba discriminativa (1 = acierto, 0 = fallo)": pd.DataFrame(
        {"juez": [f"J{i}" for i in range(1, 7)], "acierto": [1, 0, 1, 1, 0, 1]}),
    "Preferencia pareada (muestra elegida)": pd.DataFrame(
        {"juez": [f"J{i}" for i in range(1, 7)], "preferencia": ["A", "B", "A", "A", "B", "A"]}),
    "Ordenamiento (lugar de cada muestra por juez)": pd.DataFrame(
        {"juez": ["J1", "J2", "J3"], "A": [1, 2, 1], "B": [2, 1, 3], "C": [3, 3, 2]}),
}


# ==========================================================================
# Barra lateral
# ==========================================================================
with st.sidebar:
    st.markdown("##  SensoLab")
    st.caption("Análisis y evaluación sensorial · de las boletas al reporte")
    product = st.text_input("Producto evaluado", "Galleta DNP", help="Aparece en la interpretación y en el PDF.")
    author = st.text_input("Elaborado por", "", placeholder="Nombre(s) del equipo")
    alpha = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05,
                             help="0,05 = 95 % de confianza (lo habitual en análisis sensorial).")
    ia_threshold = st.number_input("Umbral de aceptabilidad (IA %)", 50.0, 95.0, 70.0, 5.0,
                                   help="Un producto se considera aceptado si IA ≥ este valor.")
    st.divider()
    with st.expander("¿Cómo preparar mi archivo?"):
        st.markdown(
            "- **Formato largo**: una fila por juez × muestra, p. ej. `juez, muestra, agrado`.\n"
            "- **Formato ancho**: una fila por juez y una columna por muestra, p. ej. `juez, A, B, C`.\n"
            "- **Descriptiva**: `juez, muestra, dulzor, dureza, …` (un atributo por columna).\n"
            "- **Discriminativa**: `juez, acierto` con 1/0, sí/no o correcto/incorrecto.\n"
            "- **A – no A**: `juez, muestra_presentada, respuesta` con «A» / «no A».\n"
            "- **Preferencia**: `juez, preferencia` con el código de la muestra elegida.\n\n"
            "Se aceptan `,` o `;` como separador y coma decimal (7,5).")
        st.download_button("Descargar ejemplos / plantillas (.zip)", examples_zip(),
                           "plantillas_sensolab.zip", "application/zip", use_container_width=True)

st.title("Análisis sensorial automático")
st.caption("Carga tus boletas, elige el tipo de prueba y obtén el método adecuado, la interpretación "
           "para tu producto y un reporte PDF.")

# ==========================================================================
# PASO 1 · Cargar datos
# ==========================================================================
step(1, "Carga tus datos", "CSV o Excel. Puedes subir el consolidado del panel o varias boletas individuales a la vez.")
tab_file, tab_example, tab_manual = st.tabs([" Subir archivo(s)", " Usar un ejemplo", " Ingresar a mano"])
loaded: LoadResult | None = None
source_name = ""

with tab_file:
    files = st.file_uploader("Archivos CSV / Excel", type=["csv", "txt", "xlsx", "xls"],
                             accept_multiple_files=True, label_visibility="collapsed")
with tab_example:
    ex = st.selectbox("Conjunto de ejemplo", ["—"] + list(EXAMPLE_LABELS),
                      format_func=lambda f: "Selecciona…" if f == "—" else EXAMPLE_LABELS[f])
with tab_manual:
    tpl = st.selectbox("Plantilla", list(MANUAL_TEMPLATES))
    manual_df = st.data_editor(MANUAL_TEMPLATES[tpl], num_rows="dynamic", use_container_width=True,
                               key=f"editor_{tpl}")
    use_manual = st.checkbox("Analizar estos datos", value=False)

try:
    if files:
        parts = [(f.name, _read_cached(f.getvalue(), f.name)) for f in files]
        loaded = combine_ballots(parts)
        source_name = ", ".join(f.name for f in files)
    elif use_manual:
        df_m = manual_df.dropna(how="all")
        if df_m.empty:
            raise DataLoadError("La tabla manual está vacía.")
        csv_bytes = df_m.to_csv(index=False).encode()
        loaded = read_table(csv_bytes, "ingreso_manual.csv")
        source_name = "ingreso manual"
    elif ex != "—":
        loaded = _read_cached((EXAMPLES / ex).read_bytes(), ex)
        source_name = ex
except DataLoadError as exc:
    st.error(f"❌ {exc}")
    st.stop()

if loaded is None:
    st.info(" Sube un archivo, elige un ejemplo o ingresa los datos a mano para comenzar.")
    st.stop()

df = loaded.df
data_key = hashlib.md5(pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes()).hexdigest()
if st.session_state.get("data_key") != data_key:
    st.session_state["data_key"] = data_key
    reset_results()

with st.expander(f"Vista previa · {source_name} ({len(df)} filas × {df.shape[1]} columnas)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    for n in loaded.notes:
        st.caption(f"ℹ️ {n}")

# ---------------- Mapeo de columnas ----------------
guess = guess_spec(df)
cols = list(df.columns)
with st.expander("Columnas detectadas (revisa o corrige)", expanded=not guess.value_cols):
    c1, c2, c3 = st.columns(3)
    layout = c1.radio("Formato", ["largo", "ancho"], index=0 if guess.layout == "largo" else 1, horizontal=True,
                      key=f"layout_{data_key}",
                      help="Largo: una fila por juez × muestra. Ancho: una fila por juez y una columna por muestra.")
    judge_opts = ["(no hay)"] + cols
    judge_col = c2.selectbox("Columna de juez", judge_opts,
                             index=judge_opts.index(guess.judge_col) if guess.judge_col else 0, key=f"j_{data_key}")
    judge_col = None if judge_col == "(no hay)" else judge_col
    if layout == "largo":
        s_opts = ["(no hay)"] + [c for c in cols if c != judge_col]
        sample_col = c3.selectbox("Columna de muestra", s_opts,
                                  index=s_opts.index(guess.sample_col) if guess.sample_col in s_opts else 0,
                                  key=f"s_{data_key}")
        sample_col = None if sample_col == "(no hay)" else sample_col
        value_label = "Columnas de respuesta (calificación, atributos, aciertos…)"
    else:
        sample_col = None
        value_label = "Columnas que son muestras"
    candidates = [c for c in cols if c not in {judge_col, sample_col}]
    default_vals = [c for c in guess.value_cols if c in candidates] or candidates
    value_cols = st.multiselect(value_label, candidates, default=default_vals, key=f"v_{layout}_{data_key}")
    wide_name = "Respuesta"
    if layout == "ancho":
        wide_name = st.text_input("¿Qué midieron? (nombre de la respuesta)", "agrado", key=f"w_{data_key}")

spec = ColumnSpec(layout=layout, judge_col=judge_col, sample_col=sample_col, value_cols=value_cols,
                  wide_attribute_name=wide_name or "Respuesta")
try:
    long_raw, map_notes = to_long(df, spec)
except ValueError as exc:
    st.error(f" {exc}")
    st.stop()
if long_raw.empty:
    st.error(" No quedaron filas con juez identificado. Revisa la columna de juez.")
    st.stop()
profile0 = build_profile(long_raw)

# ==========================================================================
# PASO 2 · Tipo de prueba
# ==========================================================================
st.divider()
step(2, "Selecciona el tipo de prueba")
prep_guess = prepare(long_raw, AFECTIVA, None)
suggested, why = suggest_test_type(prep_guess)
keys = list(TEST_TYPES)
test_type = st.radio("Tipo de prueba", keys, index=keys.index(suggested), format_func=lambda k: TEST_TYPES[k],
                     horizontal=True, label_visibility="collapsed", key=f"tt_{data_key}")
if why:
    st.caption(f" Sugerencia según los datos: **{TEST_TYPES[suggested].split(' ·')[0]}** ({why}).")

scale_attrs = [a for a in profile0.attributes if a.kind == ESCALA]
rank_attrs = [a for a in profile0.attributes if a.kind == RANGO]
binary_attrs = [a for a in profile0.attributes if a.kind == "binario"]
scale = None
p1, p2, p3 = st.columns(3)
if scale_attrs:
    vals = long_raw[long_raw["atributo"].isin([a.name for a in scale_attrs])]["valor"]
    lo_s, hi_s = suggest_scale(*robust_range(vals))
    p1.markdown("**Escala usada en la boleta**")
    sc1, sc2 = p1.columns(2)
    lo = sc1.number_input("Mínimo", value=float(lo_s), step=1.0, key=f"lo_{data_key}")
    hi = sc2.number_input("Máximo", value=float(hi_s), step=1.0, key=f"hi_{data_key}")
    if hi <= lo:
        p1.error("El máximo debe ser mayor que el mínimo.")
        st.stop()
    scale = (lo, hi)
    drop_out = p1.checkbox("Excluir valores fuera de escala", True, key=f"drop_{data_key}")
else:
    drop_out = True
rank_one_best = True
if rank_attrs or (test_type == DISCRIMINATIVA and scale_attrs):
    rank_one_best = p2.radio("En la boleta, el lugar 1 significa…", ["mayor agrado / intensidad",
                                                                     "menor agrado / intensidad"],
                             key=f"r1_{data_key}").startswith("mayor")
protocol = "triangular"
if test_type == DISCRIMINATIVA and binary_attrs:
    auto_p = detect_protocol(prep_guess, "triangular")
    disc_keys = [k for k in PROTOCOLS if k != "preferencia"]
    protocol = p3.selectbox("Protocolo discriminativo", disc_keys, index=disc_keys.index(auto_p),
                            format_func=lambda k: f"{PROTOCOLS[k].name} (azar = {PROTOCOLS[k].p0:.3g})",
                            key=f"proto_{data_key}")
    p3.caption(PROTOCOLS[protocol].description)

# ==========================================================================
# PASO 3 · Revisión de datos
# ==========================================================================
prep = prepare(long_raw, test_type, scale, drop_out)
st.divider()
step(3, "Revisión automática de los datos", "Se detectan datos incompletos, jueces faltantes, escalas mal "
                                            "codificadas, duplicados y otros problemas antes de analizar.")
st.markdown(describe_design(prep, test_type))
for n in map_notes:
    st.warning(n)
counts = {lvl: sum(i.level == lvl for i in prep.issues) for lvl in (ERROR, WARNING, INFO)}
m1, m2, m3, m4 = st.columns(4)
m1.metric("Jueces", prep.profile.n_judges)
m2.metric("Muestras", prep.profile.k)
m3.metric("Datos válidos", prep.profile.n_obs)
m4.metric("Alertas", counts[ERROR] + counts[WARNING])
for issue in prep.issues:
    with st.container():
        LEVEL_UI[issue.level](f"**{issue.title}**" + (f" — {issue.detail}" if issue.detail else ""))
        if issue.table is not None and not issue.table.empty:
            with st.expander("Ver registros"):
                show_df(issue.table)
with st.expander("Datos depurados (los que entran al análisis)"):
    clean_view = prep.data.rename(columns={"valor": "valor (num.)", "categoria": "respuesta (texto)"})
    st.dataframe(clean_view, use_container_width=True, hide_index=True)
    st.download_button("Descargar datos depurados (CSV para validar en Excel)",
                       prep.data.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       "datos_depurados.csv", "text/csv")
if prep.has_errors:
    st.error("Corrige los errores marcados para poder continuar.")
    st.stop()

# ==========================================================================
# PASO 4 · Métodos
# ==========================================================================
st.divider()
step(4, "Métodos estadísticos recomendados",
     "Según la guía: conteos → binomial / χ² · notas → t / ANOVA (+Tukey) · órdenes → Friedman.")
recs = recommend(prep, test_type)
usable = [r for r in recs if r.status != NO_APLICA]
if not usable:
    st.warning("Ningún método aplica a estos datos con el tipo de prueba elegido. Revisa el tipo de prueba o la "
               "codificación de las respuestas.")
    st.stop()
badge = {RECOMENDADO: "b-rec", "alternativa": "b-alt", "complementario": "b-comp", NO_APLICA: "b-no"}
selected = []
for r in usable:
    with st.container(border=True):
        a, b = st.columns([0.06, 0.94])
        checked = a.checkbox(r.name, value=r.status == RECOMENDADO, key=f"m_{r.method_id}_{data_key}_{test_type}",
                             label_visibility="collapsed")
        b.markdown(f"**{r.name}** &nbsp; <span class='badge {badge[r.status]}'>{STATUS_LABEL[r.status]}</span>",
                   unsafe_allow_html=True)
        b.markdown(f"{r.reason}  \n<small>**Qué hace:** {r.what} · **Cómo se decide:** {r.decides}</small>",
                   unsafe_allow_html=True)
        if checked:
            selected.append(r.method_id)
with st.expander("Métodos que no aplican a estos datos y por qué"):
    for r in recs:
        if r.status == NO_APLICA:
            st.markdown(f"- **{r.name}**: {r.reason}")

params = AnalysisParams(alpha=alpha, scale=scale or (1, 9), ia_threshold=ia_threshold, protocol=protocol,
                        rank_one_is_best=rank_one_best, product_name=product or "el producto")
signature = (data_key, test_type, tuple(selected), str(params), str(spec), drop_out)
run = st.button("▶ Analizar", type="primary", disabled=not selected, use_container_width=True)
if not selected:
    st.caption("Selecciona al menos un método.")
if run:
    reset_results()
    with st.spinner("Calculando…"):
        results = run_methods(prep, selected, params, test_type)
    st.session_state["results"] = {"sig": signature, "results": results}

state = st.session_state.get("results")
if not state:
    st.stop()
if state["sig"] != signature:
    st.info("Cambiaste la configuración. Presiona **Analizar** para actualizar los resultados.")
    st.stop()
results = state["results"]

# ==========================================================================
# PASO 5 · Resultados
# ==========================================================================
st.divider()
step(5, "Resultados e interpretación")
conclusion = overall_conclusion(results, params.product_name)
st.markdown(f"<div class='decision'><b>Conclusión y decisión</b><br>{conclusion.replace(chr(10), '<br>')}</div>",
            unsafe_allow_html=True)
st.write("")
labels = [(" " if r.error else "") + r.title for r in results]
for res, tab in zip(results, st.tabs(labels)):
    with tab:
        if res.error:
            st.error(f"No se pudo calcular: {res.error}")
            continue
        if res.question:
            st.caption(f"Pregunta: {res.question}")
        if res.metrics:
            for col, (lab, val) in zip(st.columns(len(res.metrics)), res.metrics):
                col.metric(lab, val)
        if res.significant is not None:
            (st.success if res.significant else st.info)(
                "Resultado significativo" if res.significant else "Resultado no significativo")
        st.markdown("#### Interpretación")
        st.markdown(res.interpretation)
        st.markdown(f"<div class='decision'><b>Decisión para el producto:</b> {res.decision}</div>",
                    unsafe_allow_html=True)
        st.write("")
        if res.figures:
            fig_cols = st.columns(min(2, len(res.figures)))
            for i, fig in enumerate(res.figures):
                with fig_cols[i % len(fig_cols)]:
                    st.pyplot(fig, use_container_width=True)
        for cap, tdf in res.tables:
            st.markdown(f"**{cap}**")
            show_df(tdf)
        for n in res.notes:
            st.caption(f" {n}")

# ==========================================================================
# PASO 6 · Reporte
# ==========================================================================
st.divider()
step(6, "Reporte descargable")
try:
    pdf_bytes = build_report(product=params.product_name, test_type=test_type, author=author,
                             design_text=describe_design(prep, test_type), issues=prep.issues, recs=recs,
                             selected=selected, results=results, params=params, conclusion=conclusion,
                             data_preview=sample_summary(prep))
    st.download_button("⬇️ Descargar reporte PDF", pdf_bytes,
                       f"reporte_sensorial_{(params.product_name or 'producto').replace(' ', '_')}.pdf",
                       "application/pdf", type="primary", use_container_width=True)
except Exception as exc:  # noqa: BLE001
    logging.exception("Error generando PDF")
    st.error(f"No se pudo generar el PDF: {exc}")
