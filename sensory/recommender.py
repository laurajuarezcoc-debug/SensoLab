"""Recomendador de métodos según el diseño (tipo de prueba, n.º de muestras, escala, jueces).

Implementa la regla de 3 preguntas de la guía del curso:
  1. ¿Qué tipo de dato tengo?  conteo → binomial/χ² · notas → t/ANOVA · orden → Friedman
  2. ¿Cuántos productos comparo?  2 → t o binomial · 3+ → ANOVA (+Tukey) o Friedman
  3. ¿La ANOVA salió con diferencia?  sí → Tukey
"""
from __future__ import annotations

from dataclasses import dataclass

from .constants import (AFECTIVA, BINARIO, CATEGORICO, DESCRIPTIVA, DISCRIMINATIVA, ESCALA,
                        MIN_JUDGES, RANGO, RESPONSE_LABELS)
from .interpret import fmt, join_es
from .methods import core
from .methods.catalog import METHODS
from .validation import PreparedData

RECOMENDADO = "recomendado"
ALTERNATIVA = "alternativa"
COMPLEMENTO = "complementario"
NO_APLICA = "no_aplicable"

STATUS_LABEL = {
    RECOMENDADO: "★ Recomendado",
    ALTERNATIVA: "Alternativa válida",
    COMPLEMENTO: "Complemento útil",
    NO_APLICA: "No aplica",
}
_RANK = {RECOMENDADO: 0, ALTERNATIVA: 1, COMPLEMENTO: 2, NO_APLICA: 3}


@dataclass
class Recommendation:
    method_id: str
    name: str
    status: str
    reason: str
    what: str
    decides: str

    @property
    def selectable(self) -> bool:
        return self.status != NO_APLICA


def _non_normal(prep: PreparedData) -> bool:
    """Revisión rápida de normalidad de residuos para sugerir Friedman."""
    for attr in [a.name for a in prep.profile.attrs_of(ESCALA)]:
        sub = prep.data[prep.data["atributo"] == attr]
        try:
            an = core.anova_samples(sub)
        except Exception:  # noqa: BLE001 - solo es una sugerencia
            continue
        if an.p_shapiro is not None and an.p_shapiro < 0.05:
            return True
    return False


def describe_design(prep: PreparedData, test_type: str) -> str:
    p = prep.profile
    kinds = {}
    for a in p.attributes:
        kinds.setdefault(a.kind, []).append(a.name)
    parts = [f"**{p.n_judges} jueces**"]
    if p.samples == ["Prueba"]:
        parts.append("**1 prueba** (sin columna de muestra)")
    else:
        parts.append(f"**{p.k} muestra{'s' if p.k != 1 else ''}** ({join_es(p.samples[:8])}{'…' if p.k > 8 else ''})")
    for kind, names in kinds.items():
        txt = f"{len(names)} variable{'s' if len(names) > 1 else ''} de tipo **{RESPONSE_LABELS[kind][0].lower() + RESPONSE_LABELS[kind][1:]}**"
        if kind == ESCALA and prep.scale:
            txt += f" en escala {fmt(prep.scale[0], 0)}–{fmt(prep.scale[1], 0)}"
        if p.is_a_no_a and kind == CATEGORICO:
            txt += " (diseño A – no A)"
        parts.append(txt + f": {join_es(names[:6])}{'…' if len(names) > 6 else ''}")
    if p.k >= 2 and not p.is_a_no_a:
        parts.append("bloques completos (todos los jueces evaluaron todas las muestras)" if p.complete_blocks
                     else f"bloques incompletos ({len(p.incomplete_judges)} jueces no evaluaron todo)")
    txt = "Detecté " + join_es(parts) + "."
    min_j = MIN_JUDGES[test_type]
    if p.n_judges < min_j:
        txt += f" ⚠️ Para este tipo de prueba se recomiendan al menos {min_j} jueces."
    return txt


def recommend(prep: PreparedData, test_type: str) -> list[Recommendation]:
    p = prep.profile
    k = p.k
    has_scale = bool(p.attrs_of(ESCALA))
    has_rank = bool(p.attrs_of(RANGO))
    has_binary = bool(p.attrs_of(BINARIO))
    choice_attrs = [a for a in p.attrs_of(CATEGORICO) if not a.a_no_a]
    n_opts = max((len(a.categories) for a in choice_attrs), default=0)
    n_scale_attrs = len(p.attrs_of(ESCALA))
    non_normal = has_scale and k >= 2 and _non_normal(prep)

    status: dict[str, tuple[str, str]] = {}

    def put(mid: str, st: str, reason: str) -> None:
        # Conserva el mejor estado si se asigna más de una vez
        if mid not in status or _RANK[st] < _RANK[status[mid][0]]:
            status[mid] = (st, reason)

    if test_type == AFECTIVA:
        if has_scale:
            if k == 1:
                put("ia", RECOMENDADO, "1 producto con notas de agrado: el IA dice si gusta lo suficiente.")
            else:
                put("ia", COMPLEMENTO, "Describe la aceptación de cada muestra por separado (IA ≥ 70 %).")
            if k == 2:
                put("t", RECOMENDADO, "2 muestras con notas: compara sus promedios"
                    + (" (pareada: los mismos jueces probaron ambas)." if p.complete_blocks else "."))
                put("anova", ALTERNATIVA, "Con 2 muestras la ANOVA equivale a la t de Student.")
            elif k >= 3:
                put("anova", RECOMENDADO, f"{k} muestras con notas: la ANOVA dice si hay diferencia y Tukey cuáles.")
            if k >= 3:
                put("friedman", RECOMENDADO if non_normal else ALTERNATIVA,
                    "Los residuos no son normales: Friedman (rangos) es más robusto." if non_normal else
                    "Si prefieres tratar las notas como orden (no exige normalidad).")
            if n_scale_attrs >= 2 and k >= 3:
                put("pca", COMPLEMENTO, f"{n_scale_attrs} atributos de agrado: el mapa muestra qué impulsa la preferencia.")
        if has_rank:
            put("friedman", RECOMENDADO, f"Datos de orden (ranking) de {k} muestras.")
        if choice_attrs:
            if n_opts == 2:
                put("binomial", RECOMENDADO, "Eligieron una de 2 muestras (preferencia pareada, bilateral vs. ½).")
                put("chi2_gof", ALTERNATIVA, "Aproximación χ² de la binomial.")
            elif n_opts >= 3:
                put("chi2_gof", RECOMENDADO, f"Eligieron entre {n_opts} opciones: χ² contra reparto igual.")
        if has_binary:
            if k >= 2:
                put("cochran", RECOMENDADO, "Respuestas sí/no de los mismos jueces en varias muestras.")
                put("binomial", COMPLEMENTO, "Prueba si la mayoría acepta cada muestra.")
            else:
                put("binomial", RECOMENDADO, "Respuestas sí/no para 1 muestra: ¿la mayoría la acepta?")

    elif test_type == DISCRIMINATIVA:
        if has_binary:
            put("binomial", RECOMENDADO, "Datos de aciertos/fallos: se comparan con la probabilidad de azar "
                "(1/3 triangular, ½ dúo-trío/pareada, 1/10 dos de cinco).")
            put("chi2_gof", ALTERNATIVA, "Aproximación χ² (con Yates) de la binomial.")
            if k >= 2:
                put("cochran", COMPLEMENTO, "Varias pruebas/muestras con los mismos jueces: ¿la tasa de aciertos cambia?")
        if p.is_a_no_a:
            put("chi2_2x2", RECOMENDADO, "Diseño A – no A: tabla 2×2 de lo presentado vs. lo respondido.")
        if has_rank:
            put("friedman", RECOMENDADO, "Ordenamiento por intensidad: Friedman compara los lugares.")
        if has_scale:
            put("anova", NO_APLICA, "En pruebas discriminativas el dato es acierto/fallo; la ANOVA/Tukey no se usan aquí. "
                "Si tus datos son intensidades, cambia el tipo a «Descriptiva».")
            put("t", NO_APLICA, "No se usa en discriminativas. Para intensidades elige «Descriptiva».")
            if k >= 3:
                put("friedman", ALTERNATIVA, "Las notas se pueden tratar como orden de intensidad.")
        if choice_attrs and not has_binary:
            put("binomial", NO_APLICA, "Codifica cada respuesta como acierto (1) o fallo (0) para usar la binomial.")

    elif test_type == DESCRIPTIVA:
        if has_scale:
            if k == 2:
                put("t", RECOMENDADO, "2 productos: t de Student por atributo.")
                put("anova", ALTERNATIVA, "Con 2 muestras equivale a la t.")
            elif k >= 3:
                put("anova", RECOMENDADO, f"Perfil de {k} productos: una ANOVA por atributo y Tukey para ver cuáles difieren.")
                put("friedman", RECOMENDADO if non_normal else ALTERNATIVA,
                    "Residuos no normales: confirma con Friedman." if non_normal else
                    "Alternativa no paramétrica por atributo.")
            if n_scale_attrs >= 2 and k >= 3:
                put("pca", COMPLEMENTO if n_scale_attrs < 4 else RECOMENDADO,
                    f"{n_scale_attrs} atributos: mapa de sabores de tu producto vs. la competencia.")
            put("ia", NO_APLICA, "El índice de aceptabilidad es para consumidores (pruebas afectivas).")
        if has_rank:
            put("friedman", RECOMENDADO, "Datos de orden por intensidad.")

    # Razones genéricas para el resto
    generic = {
        "ia": "Necesita notas de agrado en escala (prueba afectiva).",
        "t": "Necesita notas en escala de exactamente 2 muestras.",
        "anova": "Necesita notas en escala de 2 o más muestras.",
        "friedman": "Necesita ordenamientos (o notas) de los mismos jueces en 2+ muestras.",
        "binomial": "Necesita conteos: aciertos/fallos o elección entre 2 muestras.",
        "chi2_gof": "Necesita conteos de elecciones o aciertos.",
        "chi2_2x2": "Necesita respuestas «A» / «no A» y la muestra presentada.",
        "cochran": "Necesita respuestas sí/no de los mismos jueces en 2+ muestras.",
        "pca": "Necesita 3+ muestras y 2+ atributos en escala.",
    }
    recs = []
    for mid, spec in METHODS.items():
        st, reason = status.get(mid, (NO_APLICA, generic[mid]))
        recs.append(Recommendation(mid, spec.name, st, reason, spec.what, spec.decides))
    recs.sort(key=lambda r: _RANK[r.status])
    return recs


def suggest_test_type(prep: PreparedData) -> tuple[str, str]:
    """Sugiere el tipo de prueba a partir de los datos (el usuario confirma)."""
    p = prep.profile
    names = " ".join(a.name.lower() for a in p.attributes)
    if p.is_a_no_a:
        return DISCRIMINATIVA, "las respuestas son «A» / «no A»"
    if p.attrs_of(BINARIO):
        if any(w in names for w in ("acept", "compra", "gusta")):
            return AFECTIVA, "las respuestas sí/no parecen de aceptación o intención de compra"
        return DISCRIMINATIVA, "las respuestas son aciertos/fallos"
    if [a for a in p.attrs_of(CATEGORICO) if not a.a_no_a]:
        return AFECTIVA, "los jueces eligieron una muestra (preferencia)"
    if p.attrs_of(ESCALA):
        if len(p.attrs_of(ESCALA)) >= 3 and p.n_judges <= 15:
            return DESCRIPTIVA, "varios atributos de intensidad evaluados por un panel pequeño"
        return AFECTIVA, "notas en escala evaluadas por un panel de consumidores"
    if p.attrs_of(RANGO):
        return AFECTIVA, "ordenamiento de muestras (elige «Discriminativa» si ordenaron por intensidad)"
    return AFECTIVA, ""
