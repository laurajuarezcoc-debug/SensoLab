"""Detección de columnas, formato canónico (largo) y perfil del diseño experimental."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import (A_VALUES, BINARIO, BINARY_FALSE, BINARY_TRUE, CATEGORICO,
                        COMMON_SCALES, ESCALA, NOT_A_VALUES, RANGO)
from .io_utils import is_text, normalize_text

JUDGE_PATTERNS = r"^(juez|jueces|panelista|consumidor|evaluador|catador|sujeto|participante|persona|boleta|nombre|judge|panelist|assessor)|^id(_?juez)?$"
SAMPLE_PATTERNS = r"^(muestra|producto|tratamiento|formulaci|codigo|cod|sample|product|prototipo|presentad|estimulo|variante)"
META_PATTERNS = r"^(_archivo|fecha|sesion|edad|sexo|genero|comentario|observaci|hora|orden_pres|orden de pres|repeticion|rep$|grupo)"
RANK_HINT = r"(rango|orden|posici|rank|lugar)"
CHOICE_HINT = r"(preferencia|prefier|eleccion|elige|eligi|escog|choice|prefer)"
BINARY_HINT = r"(acierto|correct|respuesta_correcta|identific|acepta|compraria|resultado)"

CANON = ["juez", "muestra", "atributo", "valor"]


@dataclass
class ColumnSpec:
    layout: str = "largo"             # "largo" o "ancho"
    judge_col: str | None = None
    sample_col: str | None = None
    value_cols: list[str] = field(default_factory=list)
    wide_attribute_name: str = "Respuesta"


@dataclass
class AttributeProfile:
    name: str
    kind: str
    n_valid: int
    vmin: float | None = None
    vmax: float | None = None
    is_integer: bool = False
    a_no_a: bool = False
    categories: list[str] = field(default_factory=list)


@dataclass
class DataProfile:
    n_judges: int
    samples: list[str]
    attributes: list[AttributeProfile]
    complete_blocks: bool
    incomplete_judges: list[str]
    n_obs: int

    @property
    def k(self) -> int:
        return len(self.samples)

    @property
    def main_kind(self) -> str:
        kinds = [a.kind for a in self.attributes]
        return max(set(kinds), key=kinds.count) if kinds else ESCALA

    @property
    def is_a_no_a(self) -> bool:
        return any(a.a_no_a for a in self.attributes)

    def attrs_of(self, kind: str) -> list[AttributeProfile]:
        return [a for a in self.attributes if a.kind == kind]


# --------------------------------------------------------------------------
# Detección automática de columnas
# --------------------------------------------------------------------------
def _match(col: str, pattern: str) -> bool:
    return re.search(pattern, normalize_text(col)) is not None


def guess_spec(df: pd.DataFrame) -> ColumnSpec:
    cols = list(df.columns)
    judge = next((c for c in cols if _match(c, JUDGE_PATTERNS)), None)
    candidates = [c for c in cols if c != judge and _match(c, SAMPLE_PATTERNS)]
    sample = None
    # Una sola columna «muestra» con valores repetidos (no sí/no) → formato largo.
    # Varias columnas «Prototipo_A, Prototipo_B…» → formato ancho (cada columna es una muestra).
    if len(candidates) == 1:
        col = df[candidates[0]].dropna()
        binary_like = col.map(parse_binary).notna().mean() > 0.9 and col.nunique() <= 2
        if not binary_like and col.nunique() < len(col):
            sample = candidates[0]
    meta = {c for c in cols if _match(c, META_PATTERNS)}
    values = [c for c in cols if c not in {judge, sample} and c not in meta]
    if judge is None and values:
        # Primera columna no numérica con valores únicos por fila → probablemente juez
        first = cols[0]
        if first not in {sample} and df[first].nunique() >= max(2, len(df) // 10) and not _match(first, CHOICE_HINT):
            if is_text(df[first]) or df[first].is_monotonic_increasing:
                judge = first
                values = [c for c in values if c != first]
    layout = "largo" if sample is not None or len(values) <= 1 else "ancho"
    # Varias columnas con nombres de atributos sensoriales sin columna muestra → ancho (cada columna = muestra)
    return ColumnSpec(layout=layout, judge_col=judge, sample_col=sample, value_cols=values)


# --------------------------------------------------------------------------
# Conversión a formato largo canónico
# --------------------------------------------------------------------------
def to_long(df: pd.DataFrame, spec: ColumnSpec) -> tuple[pd.DataFrame, list[str]]:
    """Devuelve DataFrame con columnas juez, muestra, atributo, valor (valor sin convertir)."""
    notes: list[str] = []
    if not spec.value_cols:
        raise ValueError("Selecciona al menos una columna de respuestas.")
    missing = [c for c in [spec.judge_col, spec.sample_col, *spec.value_cols] if c and c not in df.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas en los datos: {', '.join(missing)}")

    work = df.copy()
    if spec.judge_col is None:
        if spec.layout == "largo" and spec.sample_col:
            work["juez"] = work.groupby(spec.sample_col).cumcount() + 1
            notes.append("No se indicó columna de juez: se asumió que el orden de las filas dentro de cada "
                         "muestra corresponde al mismo juez. Verifícalo.")
        else:
            work["juez"] = np.arange(1, len(work) + 1)
            notes.append("No se indicó columna de juez: cada fila se tomó como un juez distinto.")
        judge_col = "juez"
    else:
        judge_col = spec.judge_col
    work[judge_col] = work[judge_col].map(_label)

    if spec.layout == "ancho":
        long = work.melt(id_vars=[judge_col], value_vars=spec.value_cols,
                         var_name="muestra", value_name="valor")
        long["atributo"] = spec.wide_attribute_name or "Respuesta"
    else:
        if spec.sample_col:
            work[spec.sample_col] = work[spec.sample_col].map(_label)
            id_vars = [judge_col, spec.sample_col]
        else:
            work["_muestra"] = "Prueba"
            id_vars = [judge_col, "_muestra"]
        long = work.melt(id_vars=id_vars, value_vars=spec.value_cols,
                         var_name="atributo", value_name="valor")
        long = long.rename(columns={id_vars[1]: "muestra"})
    long = long.rename(columns={judge_col: "juez"})
    long = long[CANON].copy()
    long["muestra"] = long["muestra"].map(_label)
    long["atributo"] = long["atributo"].astype(str)
    long = long[long["juez"].notna() & (long["juez"] != "nan")]
    return long.reset_index(drop=True), notes


def _label(v) -> str | None:
    """Etiqueta limpia: 347.0 → '347'."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


# --------------------------------------------------------------------------
# Clasificación del tipo de respuesta
# --------------------------------------------------------------------------
def parse_binary(v) -> float:
    """1.0 / 0.0 / NaN para respuestas tipo acierto/fallo o sí/no."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return np.nan
    t = normalize_text(_label(v) or "")
    if t in BINARY_TRUE:
        return 1.0
    if t in BINARY_FALSE:
        return 0.0
    return np.nan


def parse_a_no_a(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    t = normalize_text(_label(v) or "")
    if t in A_VALUES:
        return "A"
    if t in NOT_A_VALUES:
        return "no A"
    return None


def classify_attribute(long: pd.DataFrame, attr: str) -> AttributeProfile:
    sub = long[long["atributo"] == attr]
    raw = sub["valor"]
    nonnull = raw.dropna()
    nonnull = nonnull[nonnull.astype(str).str.strip() != ""]
    n = len(nonnull)
    if n == 0:
        return AttributeProfile(attr, ESCALA, 0)

    num = pd.to_numeric(nonnull, errors="coerce")
    frac_num = num.notna().mean()
    name = normalize_text(attr)

    # A / no A
    samples_norm = {normalize_text(s) for s in sub["muestra"].dropna().unique()}
    vals_norm = {normalize_text(_label(v)) for v in nonnull.unique()}
    ano = A_VALUES | NOT_A_VALUES
    if vals_norm and vals_norm <= ano and (samples_norm <= ano or "a" in vals_norm):
        return AttributeProfile(attr, CATEGORICO, n, a_no_a=True,
                                categories=sorted({parse_a_no_a(v) for v in nonnull.unique()} - {None}))

    if re.search(CHOICE_HINT, name):
        return AttributeProfile(attr, CATEGORICO, n, categories=sorted(map(_label, nonnull.unique())))

    if frac_num >= 0.8:
        vals = num.dropna()
        uniq = set(vals.unique())
        is_int = bool(np.all(np.isclose(vals, np.round(vals))))
        prof = AttributeProfile(attr, ESCALA, n, float(vals.min()), float(vals.max()), is_int)
        if uniq <= {0.0, 1.0}:
            prof.kind = BINARIO
            return prof
        # Códigos de muestra de 3 dígitos elegidos (preferencia) sin nombre explícito
        if len(uniq) <= 3 and min(uniq) >= 100 and is_int and sub["muestra"].nunique() == 1:
            return AttributeProfile(attr, CATEGORICO, n, categories=sorted(map(_label, nonnull.unique())))
        k = sub["muestra"].nunique()
        if re.search(RANK_HINT, name) or (is_int and k >= 2 and _looks_like_ranks(sub, k)):
            prof.kind = RANGO
        return prof

    parsed = nonnull.map(parse_binary)
    if parsed.notna().mean() >= 0.8 or re.search(BINARY_HINT, name):
        return AttributeProfile(attr, BINARIO, n)
    return AttributeProfile(attr, CATEGORICO, n, categories=sorted(map(_label, nonnull.unique())))


def _looks_like_ranks(sub: pd.DataFrame, k: int) -> bool:
    ok = 0
    groups = list(sub.groupby("juez"))
    for _, g in groups:
        v = pd.to_numeric(g["valor"], errors="coerce").dropna()
        if len(v) == k and sorted(v.astype(int)) == list(range(1, k + 1)):
            ok += 1
    return bool(groups) and ok / len(groups) >= 0.6


def build_profile(long: pd.DataFrame, attributes: list[AttributeProfile] | None = None) -> DataProfile:
    """Perfil del diseño. Si se pasan `attributes`, se conserva su clasificación."""
    if "categoria" in long.columns:
        valid = long[long["valor"].notna() | long["categoria"].notna()]
    else:
        valid = long.dropna(subset=["valor"])
    if attributes is None:
        attrs = [classify_attribute(long, a) for a in long["atributo"].unique()]
    else:
        present = set(valid["atributo"].unique())
        attrs = []
        for a in attributes:
            if a.name not in present:
                continue
            vals = pd.to_numeric(valid.loc[valid["atributo"] == a.name, "valor"], errors="coerce").dropna()
            attrs.append(AttributeProfile(a.name, a.kind, int((valid["atributo"] == a.name).sum()),
                                          float(vals.min()) if len(vals) else None,
                                          float(vals.max()) if len(vals) else None,
                                          a.is_integer, a.a_no_a, a.categories))
    samples = [str(x) for x in pd.unique(valid["muestra"].dropna())]
    judges = sorted(long["juez"].unique(), key=_natural_key)
    incomplete = []
    for j in judges:
        got = valid[valid["juez"] == j].groupby("atributo")["muestra"].nunique()
        if len(got) < len(attrs) or (got < len(samples)).any():
            incomplete.append(j)
    return DataProfile(n_judges=len(judges), samples=samples, attributes=attrs,
                       complete_blocks=not incomplete, incomplete_judges=incomplete,
                       n_obs=int(valid.shape[0]))


def robust_range(values: pd.Series) -> tuple[float | None, float | None]:
    """Rango típico de los datos ignorando valores aberrantes (p. ej. 77 en una escala 1–9)."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return None, None
    q1, q3 = v.quantile([0.25, 0.75])
    iqr = max(q3 - q1, 1.0)
    core_v = v[(v >= q1 - 3 * iqr) & (v <= q3 + 3 * iqr)]
    return float(core_v.min()), float(core_v.max())


def suggest_scale(vmin: float | None, vmax: float | None) -> tuple[float, float]:
    """Escala típica más pequeña que contiene los valores observados."""
    if vmin is None or vmax is None:
        return (1, 9)
    if vmin >= 1 and vmax <= 5:
        return (1, 5)
    for lo, hi in COMMON_SCALES:
        if (lo, hi) == (1, 5):
            continue
        if vmin >= lo and vmax <= hi:
            return (lo, hi)
    return (float(np.floor(vmin)), float(np.ceil(vmax)))


def _natural_key(s) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]
