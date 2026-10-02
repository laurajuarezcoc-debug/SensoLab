"""Utilidades de redacción en lenguaje natural (formato numérico en español)."""
from __future__ import annotations

import math

from .constants import HEDONIC_9


def fmt(x: float | int | None, d: int = 2) -> str:
    """Número con coma decimal: 7.125 → '7,13'."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and d == 0):
        return f"{int(x):,}".replace(",", " ")
    return f"{x:.{d}f}".replace(".", ",")


def fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "p = —"
    if p < 0.001:
        return "p < 0,001"
    return f"p = {fmt(p, 3)}"


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{fmt(x, d)} %"


def sig_phrase(p: float, alpha: float) -> str:
    conf = fmt((1 - alpha) * 100, 0)
    if p < alpha:
        return f"la diferencia es estadísticamente significativa ({fmt_p(p)} < {fmt(alpha, 2)}; confianza {conf} %)"
    return f"no hay diferencia estadísticamente significativa ({fmt_p(p)} ≥ {fmt(alpha, 2)})"


def hedonic_label(mean: float, scale: tuple[float, float]) -> str | None:
    """Etiqueta verbal de la escala hedónica de 9 puntos más cercana a la media."""
    if tuple(int(s) for s in scale) != (1, 9):
        return None
    return HEDONIC_9.get(int(round(min(max(mean, 1), 9))))


def join_es(items: list[str]) -> str:
    items = [str(i) for i in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]
