"""Orquestación: ejecutar métodos de forma segura y redactar la conclusión global."""
from __future__ import annotations

import logging

import pandas as pd

from .constants import AFECTIVA, CATEGORICO
from .methods.catalog import METHODS, AnalysisParams, MethodResult
from .validation import PreparedData

log = logging.getLogger(__name__)


def run_methods(prep: PreparedData, method_ids: list[str], params: AnalysisParams,
                test_type: str) -> list[MethodResult]:
    """Ejecuta cada método; un fallo en uno no detiene a los demás."""
    results: list[MethodResult] = []
    for mid in method_ids:
        spec = METHODS[mid]
        try:
            results.extend(spec.run(prep, params, test_type))
        except Exception as exc:  # noqa: BLE001 - se informa al usuario sin romper la app
            log.exception("Fallo en el método %s", mid)
            results.append(MethodResult(mid, spec.name, error=f"Error inesperado: {exc}"))
    return results


def overall_conclusion(results: list[MethodResult], product: str) -> str:
    ok = [r for r in results if not r.error and r.decision]
    if not ok:
        return "No se obtuvieron resultados válidos; revisa las alertas de calidad de datos."
    lines = [f"Resumen de decisiones para {product}:"]
    seen = set()
    for r in ok:
        if r.decision in seen:
            continue
        seen.add(r.decision)
        lines.append(f"• {r.title}: {r.decision}")
    errors = [r for r in results if r.error]
    if errors:
        lines.append(f"({len(errors)} análisis no pudieron calcularse; ver detalle en la sección de resultados.)")
    return "\n".join(lines)


def sample_summary(prep: PreparedData) -> pd.DataFrame:
    """Tabla resumen por atributo y muestra para mostrar y para el PDF."""
    d = prep.data
    rows = []
    for attr in prep.profile.attributes:
        sub = d[d["atributo"] == attr.name]
        for s, g in sub.groupby("muestra", sort=False):
            if attr.kind == CATEGORICO:
                counts = g["categoria"].value_counts()
                rows.append({"Atributo": attr.name, "Muestra": s, "n": len(g),
                             "Conteos": ", ".join(f"{k}: {v}" for k, v in counts.items())})
            else:
                v = g["valor"]
                rows.append({"Atributo": attr.name, "Muestra": s, "n": int(v.notna().sum()),
                             "Media": v.mean(), "DE": v.std(), "Mín": v.min(), "Máx": v.max()})
    return pd.DataFrame(rows).dropna(axis=1, how="all")


__all__ = ["run_methods", "overall_conclusion", "sample_summary", "AnalysisParams", "AFECTIVA"]
