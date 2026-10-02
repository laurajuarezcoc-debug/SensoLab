"""Validación y limpieza: datos incompletos, jueces faltantes, escalas mal codificadas…"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import (BINARIO, CATEGORICO, ESCALA, MIN_JUDGES, RANGO, TEST_TYPES,
                        DESCRIPTIVA)
from .io_utils import normalize_text
from .schema import (DataProfile, build_profile, parse_a_no_a, parse_binary, _label)

ERROR, WARNING, INFO = "error", "warning", "info"


@dataclass
class Issue:
    level: str
    title: str
    detail: str = ""
    table: pd.DataFrame | None = None


@dataclass
class PreparedData:
    data: pd.DataFrame            # juez, muestra, atributo, valor (float), categoria (str)
    profile: DataProfile
    issues: list[Issue] = field(default_factory=list)
    scale: tuple[float, float] | None = None

    @property
    def has_errors(self) -> bool:
        return any(i.level == ERROR for i in self.issues)


def _examples(values, n: int = 6) -> str:
    vals = list(dict.fromkeys(map(str, values)))
    more = f" … (+{len(vals) - n})" if len(vals) > n else ""
    return ", ".join(f"«{v}»" for v in vals[:n]) + more


def unify_labels(long: pd.DataFrame, issues: list[Issue]) -> pd.DataFrame:
    """Une etiquetas de muestra/juez que solo difieren en mayúsculas, tildes o espacios."""
    for col, nombre in (("muestra", "muestras"), ("juez", "jueces")):
        labels = long[col].dropna().unique()
        groups: dict[str, list[str]] = {}
        for lab in labels:
            groups.setdefault(normalize_text(lab), []).append(lab)
        mapping = {}
        merged = []
        for variants in groups.values():
            if len(variants) > 1:
                canon = max(variants, key=lambda v: (long[col] == v).sum())
                for v in variants:
                    mapping[v] = canon
                merged.append(" = ".join(variants))
        if mapping:
            long[col] = long[col].replace(mapping)
            issues.append(Issue(WARNING, f"Etiquetas de {nombre} escritas de distinta forma",
                                "Se unificaron: " + "; ".join(merged)))
    return long


def prepare(long_raw: pd.DataFrame, test_type: str,
            scale: tuple[float, float] | None = None,
            drop_out_of_scale: bool = True) -> PreparedData:
    """Convierte valores según su tipo, detecta problemas y devuelve datos listos."""
    issues: list[Issue] = []
    long = long_raw.copy()
    long = unify_labels(long, issues)
    profile0 = build_profile(long)

    long["categoria"] = None
    long["valor_num"] = np.nan
    for attr in profile0.attributes:
        mask = long["atributo"] == attr.name
        raw = long.loc[mask, "valor"]
        present = raw.notna() & (raw.astype(str).str.strip() != "")
        if attr.kind in (ESCALA, RANGO):
            num = pd.to_numeric(raw, errors="coerce")
            bad = present & num.isna()
            if bad.any():
                issues.append(Issue(
                    WARNING, f"Escala mal codificada en «{attr.name}»",
                    f"{int(bad.sum())} celda(s) no son números y se tratarán como faltantes: "
                    f"{_examples(raw[bad])}. Corrige la boleta (p. ej. «siete» → 7).",
                    long.loc[bad[bad].index, ["juez", "muestra", "valor"]]))
            long.loc[mask, "valor_num"] = num
        elif attr.kind == BINARIO:
            parsed = raw.map(parse_binary)
            bad = present & parsed.isna()
            if bad.any():
                issues.append(Issue(
                    WARNING, f"Respuestas no reconocidas en «{attr.name}»",
                    f"{int(bad.sum())} respuesta(s) no son acierto/fallo ni sí/no: {_examples(raw[bad])}. "
                    "Usa 1/0, sí/no o correcto/incorrecto.",
                    long.loc[bad[bad].index, ["juez", "muestra", "valor"]]))
            long.loc[mask, "valor_num"] = parsed
        else:  # CATEGORICO
            if attr.a_no_a:
                parsed = raw.map(parse_a_no_a)
                long.loc[mask, "muestra"] = long.loc[mask, "muestra"].map(
                    lambda s: parse_a_no_a(s) or s)
            else:
                parsed = raw.map(lambda v: _label(v) if pd.notna(v) else None)
            bad = present & parsed.isna()
            if bad.any():
                issues.append(Issue(WARNING, f"Respuestas no reconocidas en «{attr.name}»",
                                    f"Valores ignorados: {_examples(raw[bad])}"))
            long.loc[mask, "categoria"] = parsed

    # ---------------- Escala declarada ----------------
    scale_attrs = [a.name for a in profile0.attributes if a.kind == ESCALA]
    if scale and scale_attrs:
        lo, hi = scale
        m = long["atributo"].isin(scale_attrs) & long["valor_num"].notna()
        out = m & ((long["valor_num"] < lo) | (long["valor_num"] > hi))
        if out.any():
            action = "se excluyeron del análisis" if drop_out_of_scale else "se conservaron (¡revísalos!)"
            issues.append(Issue(
                WARNING, f"{int(out.sum())} valor(es) fuera de la escala {lo:g}–{hi:g}",
                f"Valores {_examples(long.loc[out, 'valor_num'].map(lambda x: f'{x:g}'))} {action}. "
                "Suelen ser errores de digitación (p. ej. 77 en vez de 7) o una escala declarada incorrecta.",
                long.loc[out, ["juez", "muestra", "atributo", "valor_num"]]))
            if drop_out_of_scale:
                long.loc[out, "valor_num"] = np.nan
        if (hi - lo) <= 15:
            m = long["atributo"].isin(scale_attrs) & long["valor_num"].notna()
            frac = m & ~np.isclose(long["valor_num"], np.round(long["valor_num"]))
            if frac.any() and frac.sum() / max(m.sum(), 1) < 0.2:
                issues.append(Issue(INFO, "Valores con decimales en una escala de puntos enteros",
                                    f"{int(frac.sum())} valor(es) con decimales: "
                                    f"{_examples(long.loc[frac, 'valor_num'].map(lambda x: f'{x:g}'))}. "
                                    "Se usarán tal cual."))

    # ---------------- Rangos ----------------
    for attr in profile0.attrs_of(RANGO):
        sub = long[long["atributo"] == attr.name]
        k = sub["muestra"].nunique()
        bad_judges = []
        for j, g in sub.groupby("juez"):
            v = g["valor_num"].dropna()
            if len(v) and (v.min() < 1 or v.max() > k or v.duplicated().any()
                           or not np.allclose(v, np.round(v))):
                bad_judges.append(f"{j} ({', '.join(f'{x:g}' for x in v)})")
        if bad_judges:
            issues.append(Issue(WARNING, f"Rangos mal codificados en «{attr.name}»",
                                f"Con {k} muestras cada juez debe usar los lugares 1 a {k} sin repetir. "
                                f"Revisar: {'; '.join(bad_judges[:8])}"
                                + (" …" if len(bad_judges) > 8 else "")
                                + ". Los empates se aceptan (rango promedio), pero valores fuera de 1–k se excluyen."))
            out = (long["atributo"] == attr.name) & ((long["valor_num"] < 1) | (long["valor_num"] > k))
            long.loc[out, "valor_num"] = np.nan

    # ---------------- Duplicados ----------------
    has_value = long["valor_num"].notna() | long["categoria"].notna()
    dup = long[has_value].duplicated(subset=["juez", "muestra", "atributo"], keep="first")
    single_sample_choice = (long["muestra"].nunique() == 1)
    if dup.any() and not single_sample_choice:
        idx = dup[dup].index
        issues.append(Issue(WARNING, f"{len(idx)} registro(s) duplicado(s) (mismo juez, muestra y atributo)",
                            "Se conservó la primera calificación de cada juez.",
                            long.loc[idx, ["juez", "muestra", "atributo", "valor"]]))
        long = long.drop(index=idx)
    elif dup.any() and single_sample_choice:
        issues.append(Issue(WARNING, "Jueces repetidos",
                            f"Los jueces {_examples(long.loc[dup[dup].index, 'juez'])} aparecen más de una vez. "
                            "Se conservó su primera respuesta.", None))
        long = long.drop(index=dup[dup].index)

    long["valor"] = long["valor_num"]
    long = long.drop(columns=["valor_num"])
    has_value = long["valor"].notna() | long["categoria"].notna()

    # ---------------- Datos incompletos ----------------
    n_missing = int((~has_value).sum())
    if n_missing:
        miss = long[~has_value].groupby(["juez"]).agg(
            faltantes=("muestra", lambda s: ", ".join(map(str, s.unique()))))
        issues.append(Issue(WARNING, f"{n_missing} dato(s) faltante(s)",
                            "Celdas vacías o inválidas; se excluyen del análisis.",
                            miss.reset_index().rename(columns={"faltantes": "muestras sin dato"})))
    clean = long[has_value].reset_index(drop=True)
    if clean.empty:
        issues.append(Issue(ERROR, "No hay datos válidos para analizar",
                            "Revisa las columnas seleccionadas y la codificación de respuestas."))
        return PreparedData(clean, profile0, issues, scale)

    profile = build_profile(clean, profile0.attributes)
    all_judges = sorted(long["juez"].unique(), key=str)
    lost = sorted(set(all_judges) - set(clean["juez"].unique()), key=str)
    if lost:
        issues.append(Issue(WARNING, "Jueces sin ninguna respuesta válida",
                            f"Se excluyen: {_examples(lost, 10)}"))

    # ---------------- Jueces faltantes / bloques incompletos ----------------
    if profile.k >= 2 and profile.incomplete_judges and not profile.is_a_no_a:
        rows = []
        for j in profile.incomplete_judges:
            done = set(clean.loc[clean["juez"] == j, "muestra"])
            rows.append({"juez": j, "muestras sin evaluar": ", ".join(s for s in profile.samples if s not in done)})
        issues.append(Issue(
            WARNING, f"{len(profile.incomplete_judges)} juez(es) no evaluaron todas las muestras",
            "Las pruebas de bloques (t pareada, ANOVA con juez como bloque, Friedman, Q de Cochran) "
            "solo usan a los jueces completos; se indicará en cada método.",
            pd.DataFrame(rows)))
    counts = clean.groupby("muestra")["juez"].nunique()
    if profile.k >= 2 and counts.max() - counts.min() > 0:
        issues.append(Issue(INFO, "Número de evaluaciones distinto por muestra",
                            ", ".join(f"{m}: {n}" for m, n in counts.items())))

    # ---------------- Tamaño del panel ----------------
    min_j = MIN_JUDGES.get(test_type, 1)
    if profile.n_judges < 2:
        issues.append(Issue(ERROR, "Se necesitan al menos 2 jueces", f"Hay {profile.n_judges}."))
    elif profile.n_judges < min_j:
        issues.append(Issue(WARNING, f"Panel pequeño: {profile.n_judges} jueces",
                            f"Para una prueba {TEST_TYPES[test_type].split(' ·')[0].lower()} se recomiendan "
                            f"al menos {min_j}. Los resultados tienen poca potencia; interprétalos con cautela."))

    # ---------------- Escala sin variación / jueces planos ----------------
    for attr in profile.attrs_of(ESCALA):
        sub = clean[clean["atributo"] == attr.name]
        if sub["valor"].nunique() <= 1:
            issues.append(Issue(WARNING, f"«{attr.name}» no tiene variación",
                                "Todos los valores son iguales; no se puede comparar."))
            continue
        if profile.k >= 3:
            flat = [j for j, g in sub.groupby("juez") if len(g) >= 3 and g["valor"].nunique() == 1]
            if flat:
                issues.append(Issue(INFO, f"Jueces que dieron la misma nota a todas las muestras en «{attr.name}»",
                                    f"{_examples(flat, 10)}. Puede indicar falta de atención o de discriminación."))

    # ---------------- Desempeño del panel (descriptiva) ----------------
    if test_type == DESCRIPTIVA and profile.k >= 3:
        perf = panel_consensus(clean, profile)
        if perf is not None and not perf.empty:
            low = perf[perf["r_consenso"] < 0.3]
            if not low.empty:
                issues.append(Issue(WARNING, "Jueces poco alineados con el panel",
                                    "Su perfil se correlaciona poco (r < 0,3) con el promedio del resto del panel en "
                                    "algún atributo; revisar entrenamiento.", low.round(2)))

    if not any(i.level in (ERROR, WARNING) for i in issues):
        issues.append(Issue(INFO, "Datos sin problemas", "No se detectaron errores de codificación."))
    return PreparedData(clean, profile, issues, scale)


def panel_consensus(clean: pd.DataFrame, profile: DataProfile) -> pd.DataFrame | None:
    """Correlación de cada juez con la media del resto del panel (por atributo)."""
    rows = []
    for attr in profile.attrs_of(ESCALA):
        sub = clean[clean["atributo"] == attr.name]
        pivot = sub.pivot_table(index="juez", columns="muestra", values="valor", aggfunc="mean")
        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            continue
        for j in pivot.index:
            own = pivot.loc[j]
            rest = pivot.drop(index=j).mean()
            ok = own.notna() & rest.notna()
            if ok.sum() >= 3 and own[ok].std() > 0 and rest[ok].std() > 0:
                r = float(np.corrcoef(own[ok], rest[ok])[0, 1])
                rows.append({"juez": j, "atributo": attr.name, "r_consenso": r})
    return pd.DataFrame(rows) if rows else None
