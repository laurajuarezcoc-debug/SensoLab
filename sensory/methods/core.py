"""Cálculos estadísticos puros (sin interfaz). Cada función es verificable a mano.

Los valores críticos se calculan con las distribuciones exactas (binomial,
F, t, χ², rango studentizado) en lugar de copiarse de tablas, para evitar
errores de transcripción.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ==========================================================================
# Binomial
# ==========================================================================
@dataclass
class BinomialResult:
    successes: int
    n: int
    p0: float
    alternative: str
    p_value: float
    min_required: int | None
    proportion: float
    pd_estimate: float | None   # proporción de discriminadores (Abbott)


def binomial_min_required(n: int, p0: float, alpha: float, alternative: str = "greater") -> int | None:
    """Mínimo de aciertos para significancia (equivale a la tabla binomial)."""
    a = alpha / 2 if alternative == "two-sided" else alpha
    for x in range(n + 1):
        if stats.binom.sf(x - 1, n, p0) <= a:
            return x
    return None


def binomial_test(successes: int, n: int, p0: float, alpha: float = 0.05,
                  alternative: str = "greater") -> BinomialResult:
    if n <= 0:
        raise ValueError("No hay respuestas para la prueba binomial.")
    if not 0 <= successes <= n:
        raise ValueError("El número de aciertos debe estar entre 0 y n.")
    p = stats.binomtest(successes, n, p0, alternative=alternative).pvalue
    prop = successes / n
    pd_est = max(0.0, (prop - p0) / (1 - p0)) if alternative == "greater" else None
    return BinomialResult(successes, n, p0, alternative, float(p),
                          binomial_min_required(n, p0, alpha, alternative), prop, pd_est)


# ==========================================================================
# Ji cuadrado
# ==========================================================================
def chi2_goodness_two_cells(successes: int, n: int, p0: float, alternative: str = "greater") -> dict:
    """χ² con corrección de Yates para aciertos vs. fallos (alternativa a la binomial)."""
    obs = np.array([successes, n - successes], dtype=float)
    exp = np.array([n * p0, n * (1 - p0)])
    chi2 = float(np.sum((np.abs(obs - exp) - 0.5).clip(min=0) ** 2 / exp))
    p_two = float(stats.chi2.sf(chi2, 1))
    if alternative == "greater":
        p = p_two / 2 if successes > exp[0] else 1 - p_two / 2
    else:
        p = p_two
    return {"chi2": chi2, "gl": 1, "p": p, "esperados": exp, "observados": obs,
            "min_esperado": float(exp.min())}


def chi2_goodness(counts: pd.Series) -> dict:
    """χ² de bondad de ajuste contra frecuencias iguales (elección entre k opciones)."""
    obs = counts.values.astype(float)
    exp = np.full_like(obs, obs.sum() / len(obs))
    chi2, p = stats.chisquare(obs, exp)
    return {"chi2": float(chi2), "gl": len(obs) - 1, "p": float(p), "esperados": exp,
            "min_esperado": float(exp.min())}


def chi2_2x2(table: pd.DataFrame) -> dict:
    """Tabla de contingencia 2×2 (A – no A). Devuelve Pearson, Yates y Fisher."""
    arr = table.values.astype(float)
    if arr.shape != (2, 2):
        raise ValueError("Se necesita una tabla 2×2 (presentada A/no A × respuesta A/no A).")
    if (arr.sum(axis=0) == 0).any() or (arr.sum(axis=1) == 0).any():
        raise ValueError("Una fila o columna de la tabla 2×2 está vacía; faltan presentaciones o respuestas.")
    chi2, p, dof, exp = stats.chi2_contingency(arr, correction=False)
    chi2_y, p_y, _, _ = stats.chi2_contingency(arr, correction=True)
    _, p_fisher = stats.fisher_exact(arr, alternative="greater")
    return {"chi2": float(chi2), "p": float(p), "gl": int(dof), "chi2_yates": float(chi2_y),
            "p_yates": float(p_y), "p_fisher": float(p_fisher), "esperados": exp,
            "min_esperado": float(exp.min())}


# ==========================================================================
# Escalas: aceptabilidad, t, ANOVA, Tukey
# ==========================================================================
def acceptability(values: pd.Series, scale_max: float, scale_min: float, threshold: float = 70.0) -> dict:
    v = values.dropna().astype(float)
    mean = float(v.mean())
    ia = mean / scale_max * 100
    # % de jueces en la mitad superior (p. ej. ≥ 6 en la hedónica de 9 puntos)
    midpoint = (scale_min + scale_max) / 2
    pct_like = float((v > midpoint).mean() * 100)
    target = threshold / 100 * scale_max
    if len(v) >= 3 and v.std(ddof=1) > 0:
        t_res = stats.ttest_1samp(v, target, alternative="greater")
        p_above = float(t_res.pvalue)
    else:
        p_above = float("nan")
    return {"n": len(v), "media": mean, "de": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "ia": ia, "pct_agrado": pct_like, "aceptado": ia >= threshold,
            "media_objetivo": target, "p_supera": p_above}


def ttest_two(df: pd.DataFrame, samples: list[str]) -> dict:
    """t de Student: pareada si los mismos jueces evaluaron ambas muestras."""
    a, b = samples
    pivot = df.pivot_table(index="juez", columns="muestra", values="valor", aggfunc="mean")
    paired = pivot[[a, b]].dropna() if {a, b} <= set(pivot.columns) else pd.DataFrame()
    xa = df.loc[df["muestra"] == a, "valor"].dropna()
    xb = df.loc[df["muestra"] == b, "valor"].dropna()
    if len(xa) < 2 or len(xb) < 2:
        raise ValueError("Cada muestra necesita al menos 2 calificaciones.")
    n_union = len(set(df.loc[df["muestra"] == a, "juez"]) | set(df.loc[df["muestra"] == b, "juez"]))
    out: dict = {"muestras": (a, b), "n_a": len(xa), "n_b": len(xb),
                 "media_a": float(xa.mean()), "media_b": float(xb.mean()),
                 "de_a": float(xa.std(ddof=1)), "de_b": float(xb.std(ddof=1))}
    if len(paired) >= 3 and len(paired) >= 0.7 * n_union:
        d = paired[a] - paired[b]
        if d.std(ddof=1) == 0:
            t_stat, p = (0.0, 1.0) if d.mean() == 0 else (float("inf"), 0.0)
        else:
            r = stats.ttest_rel(paired[a], paired[b])
            t_stat, p = float(r.statistic), float(r.pvalue)
        gl = len(paired) - 1
        out.update(tipo="pareada", n_pares=len(paired), t=t_stat, p=p, gl=gl,
                   dif_media=float(d.mean()), media_a=float(paired[a].mean()),
                   media_b=float(paired[b].mean()))
        if len(d) >= 3 and d.std(ddof=1) > 0:
            out["p_normalidad"] = float(stats.shapiro(d).pvalue)
    else:
        lev_p = float(stats.levene(xa, xb).pvalue) if xa.std() > 0 or xb.std() > 0 else 1.0
        equal = lev_p >= 0.05
        r = stats.ttest_ind(xa, xb, equal_var=equal)
        if equal:
            gl = len(xa) + len(xb) - 2
        else:
            va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb)
            gl = (va + vb) ** 2 / (va ** 2 / (len(xa) - 1) + vb ** 2 / (len(xb) - 1))
        out.update(tipo="independiente" if equal else "independiente (Welch)",
                   t=float(r.statistic), p=float(r.pvalue), gl=float(gl),
                   dif_media=float(xa.mean() - xb.mean()), p_levene=lev_p)
    out["t_critico"] = float(stats.t.ppf(0.975, out["gl"]))
    return out


@dataclass
class AnovaResult:
    table: pd.DataFrame
    f: float
    p: float
    f_crit: float
    mse: float
    df_error: float
    design: str
    means: pd.DataFrame
    p_shapiro: float | None
    p_levene: float | None
    n_judges_used: int


def anova_samples(df: pd.DataFrame, alpha: float = 0.05) -> AnovaResult:
    """ANOVA de las muestras. Si los jueces evaluaron varias muestras se usa el juez
    como bloque (diseño de bloques completos al azar): valor ~ muestra + juez."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    d = df[["juez", "muestra", "valor"]].dropna().copy()
    d = d.groupby(["juez", "muestra"], as_index=False)["valor"].mean()
    k = d["muestra"].nunique()
    if k < 2:
        raise ValueError("Se necesitan al menos 2 muestras para el ANOVA.")
    counts = d.groupby("muestra")["valor"].count()
    if (counts < 2).any():
        raise ValueError("Cada muestra necesita al menos 2 calificaciones para el ANOVA.")
    per_judge = d.groupby("juez")["muestra"].nunique()
    blocked = per_judge.mean() >= 2 and d["juez"].nunique() >= 2
    d["muestra_"] = d["muestra"].astype(str)
    d["juez_"] = d["juez"].astype(str)
    formula = "valor ~ C(muestra_) + C(juez_)" if blocked else "valor ~ C(muestra_)"
    model = smf.ols(formula, data=d).fit()
    if model.df_resid <= 0:
        raise ValueError("No hay grados de libertad para el error; se necesitan más jueces.")
    raw = anova_lm(model, typ=2)
    names = {"C(muestra_)": "Muestras", "C(juez_)": "Jueces (bloque)", "Residual": "Error"}
    table = pd.DataFrame({
        "Fuente": [names.get(i, i) for i in raw.index],
        "SC": raw["sum_sq"].values,
        "gl": raw["df"].values,
    })
    table["CM"] = table["SC"] / table["gl"]
    table["F"] = raw["F"].values
    table["p"] = raw["PR(>F)"].values
    df_err = float(model.df_resid)
    table["F crítico"] = [float(stats.f.ppf(1 - alpha, g, df_err)) if f == f else np.nan
                          for g, f in zip(table["gl"], table["F"])]
    total = pd.DataFrame([{"Fuente": "Total", "SC": table["SC"].sum(), "gl": table["gl"].sum()}])
    table = pd.concat([table, total], ignore_index=True)
    mse = float(model.mse_resid)
    row = table[table["Fuente"] == "Muestras"].iloc[0]
    resid = model.resid
    p_sh = float(stats.shapiro(resid).pvalue) if 3 <= len(resid) <= 5000 and resid.std() > 0 else None
    groups = [g["valor"].values for _, g in d.groupby("muestra")]
    p_lev = float(stats.levene(*groups).pvalue) if all(np.std(g) > 0 for g in groups) else None
    means = d.groupby("muestra")["valor"].agg(media="mean", de="std", n="count")
    # Medias ajustadas (mínimos cuadrados) si el diseño de bloques está desbalanceado
    if blocked and not per_judge.eq(k).all():
        means["media"] = [float(model.predict(pd.DataFrame({"muestra_": [m] * d["juez_"].nunique(),
                                                             "juez_": d["juez_"].unique()})).mean())
                          for m in means.index]
    means["ee"] = np.sqrt(mse / means["n"])
    return AnovaResult(table, float(row["F"]), float(row["p"]),
                       float(stats.f.ppf(1 - alpha, row["gl"], df_err)), mse, df_err,
                       "bloques (juez como bloque)" if blocked else "un factor", means,
                       p_sh, p_lev, int(d["juez"].nunique()))


def tukey_hsd(means: pd.Series, ns: pd.Series, mse: float, df_error: float,
              alpha: float = 0.05) -> tuple[pd.DataFrame, dict[str, str], float]:
    """Tukey HSD (Tukey-Kramer si los n difieren) con el CM del error del ANOVA."""
    k = len(means)
    q_crit = float(stats.studentized_range.ppf(1 - alpha, k, df_error))
    rows = []
    for a, b in combinations(means.index, 2):
        se = np.sqrt(mse / 2 * (1 / ns[a] + 1 / ns[b]))
        diff = float(means[a] - means[b])
        q = abs(diff) / se
        p = float(stats.studentized_range.sf(q, k, df_error))
        rows.append({"Muestra 1": a, "Muestra 2": b, "Diferencia": diff,
                     "DMS (HSD)": q_crit * se, "q": q, "p": min(p, 1.0),
                     "¿Difieren?": "Sí" if p < alpha else "No"})
    pairs = pd.DataFrame(rows)
    sig = {(r["Muestra 1"], r["Muestra 2"]): r["p"] < alpha for _, r in pairs.iterrows()}
    letters = compact_letters(means, sig)
    hsd = q_crit * np.sqrt(mse / ns.mean())
    return pairs, letters, float(hsd)


def compact_letters(values: pd.Series, sig: dict[tuple, bool], descending: bool = True) -> dict[str, str]:
    """Letras de significancia: muestras que comparten letra NO difieren."""
    order = list(values.sort_values(ascending=not descending).index)

    def differ(a, b):
        return sig.get((a, b), sig.get((b, a), False))

    groups: list[list[str]] = []
    for i, start in enumerate(order):
        grp = [start]
        for other in order[i + 1:]:
            if all(not differ(other, g) for g in grp):
                grp.append(other)
            else:
                break
        if not any(set(grp) <= set(g) for g in groups):
            groups.append(grp)
    letters = {m: "" for m in order}
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for idx, grp in enumerate(groups):
        for m in grp:
            letters[m] += alphabet[idx % 26]
    return letters


# ==========================================================================
# Rangos: Friedman
# ==========================================================================
@dataclass
class FriedmanResult:
    statistic: float
    p: float
    chi2_crit: float
    gl: int
    n_judges: int
    rank_sums: pd.Series
    lsd: float
    pairs: pd.DataFrame
    letters: dict[str, str]
    ties: bool


def friedman(matrix: pd.DataFrame, alpha: float = 0.05, higher_is_more: bool = True) -> FriedmanResult:
    """Friedman sobre una matriz jueces × muestras (rangos o notas).

    Las notas se convierten a rangos dentro de cada juez (1 = menor valor).
    `higher_is_more` solo afecta el orden de las letras.
    """
    m = matrix.dropna()
    J, k = m.shape
    if k < 2:
        raise ValueError("Friedman necesita al menos 2 muestras.")
    if J < 2:
        raise ValueError("Friedman necesita al menos 2 jueces con todas las muestras evaluadas.")
    ranks = m.rank(axis=1, method="average")
    R = ranks.sum(axis=0)
    t_stat = 12 / (J * k * (k + 1)) * float((R ** 2).sum()) - 3 * J * (k + 1)
    tie_term = 0.0
    for _, row in ranks.iterrows():
        _, cnt = np.unique(row.values, return_counts=True)
        tie_term += float(np.sum(cnt ** 3 - cnt))
    ties = tie_term > 0
    denom = 1 - tie_term / (J * (k ** 3 - k))
    if denom <= 0:
        t_stat, p = 0.0, 1.0
    else:
        t_stat = t_stat / denom
        p = float(stats.chi2.sf(t_stat, k - 1))
    z = float(stats.norm.ppf(1 - alpha / 2))
    lsd = z * np.sqrt(J * k * (k + 1) / 6)
    rows, sig = [], {}
    for a, b in combinations(R.index, 2):
        diff = float(R[a] - R[b])
        s = abs(diff) >= lsd
        sig[(a, b)] = s
        rows.append({"Muestra 1": a, "Muestra 2": b, "Diferencia de sumas": diff,
                     "DMS rangos": lsd, "¿Difieren?": "Sí" if s else "No"})
    if p >= alpha:
        sig = {key: False for key in sig}
    letters = compact_letters(R, sig, descending=higher_is_more)
    return FriedmanResult(float(t_stat), p, float(stats.chi2.ppf(1 - alpha, k - 1)), k - 1, J,
                          R, float(lsd), pd.DataFrame(rows), letters, ties)


# ==========================================================================
# Q de Cochran
# ==========================================================================
@dataclass
class CochranResult:
    q: float
    p: float
    gl: int
    q_crit: float
    n_judges: int
    totals: pd.Series
    proportions: pd.Series
    pairs: pd.DataFrame


def cochran_q(matrix: pd.DataFrame, alpha: float = 0.05) -> CochranResult:
    """Q de Cochran para respuestas binarias de los mismos jueces en k muestras."""
    m = matrix.dropna().astype(int)
    J, k = m.shape
    if k < 2:
        raise ValueError("Q de Cochran necesita al menos 2 muestras.")
    if J < 2:
        raise ValueError("Q de Cochran necesita al menos 2 jueces con todas las muestras.")
    C = m.sum(axis=0)
    Rr = m.sum(axis=1)
    N = float(C.sum())
    denom = k * N - float((Rr ** 2).sum())
    if denom == 0:
        q, p = 0.0, 1.0
    else:
        q = (k - 1) * (k * float((C ** 2).sum()) - N ** 2) / denom
        p = float(stats.chi2.sf(q, k - 1))
    n_pairs = k * (k - 1) // 2
    rows = []
    for a, b in combinations(m.columns, 2):
        b01 = int(((m[a] == 1) & (m[b] == 0)).sum())
        b10 = int(((m[a] == 0) & (m[b] == 1)).sum())
        nd = b01 + b10
        p_mc = float(stats.binomtest(b01, nd, 0.5).pvalue) if nd else 1.0
        p_adj = min(1.0, p_mc * n_pairs)
        rows.append({"Muestra 1": a, "Muestra 2": b, "% Muestra 1": m[a].mean() * 100,
                     "% Muestra 2": m[b].mean() * 100, "Discordantes": nd,
                     "p (McNemar, Bonferroni)": p_adj,
                     "¿Difieren?": "Sí" if p_adj < alpha else "No"})
    return CochranResult(float(q), p, k - 1, float(stats.chi2.ppf(1 - alpha, k - 1)), J, C,
                         C / J, pd.DataFrame(rows))


# ==========================================================================
# PCA (mapa sensorial)
# ==========================================================================
@dataclass
class PCAResult:
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained: np.ndarray


def pca(means: pd.DataFrame) -> PCAResult:
    """PCA normado (correlaciones) sobre la matriz productos × atributos."""
    X = means.dropna(axis=1, how="any")
    X = X.loc[:, X.std(ddof=1) > 0]
    if X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError("El PCA necesita al menos 3 productos y 2 atributos con variación.")
    Z = (X - X.mean()) / X.std(ddof=1)
    U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
    eig = S ** 2 / (len(X) - 1)
    explained = eig / eig.sum()
    n_comp = min(len(S), 3)
    scores = pd.DataFrame(U[:, :n_comp] * S[:n_comp], index=X.index,
                          columns=[f"CP{i + 1}" for i in range(n_comp)])
    loadings = pd.DataFrame(Vt[:n_comp].T * np.sqrt(eig[:n_comp]), index=X.columns,
                            columns=[f"CP{i + 1}" for i in range(n_comp)])
    # Orientación estable: el atributo de mayor carga absoluta en CP1 queda positivo
    for c in loadings.columns:
        if loadings[c].loc[loadings[c].abs().idxmax()] < 0:
            loadings[c] *= -1
            scores[c] *= -1
    return PCAResult(scores, loadings, explained[:n_comp])
