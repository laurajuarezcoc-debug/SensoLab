"""Pruebas: valores de tablas conocidos, métodos y manejo de errores."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sensory.constants import AFECTIVA, BINARIO, CATEGORICO, DESCRIPTIVA, DISCRIMINATIVA, ESCALA, RANGO
from sensory.io_utils import DataLoadError, read_table
from sensory.methods import core
from sensory.methods.catalog import AnalysisParams
from sensory.pipeline import run_methods
from sensory.recommender import RECOMENDADO, recommend
from sensory.schema import guess_spec, robust_range, suggest_scale, to_long
from sensory.validation import ERROR, prepare

EX = Path(__file__).parents[1] / "data" / "ejemplos"


def load(text: str, name="t.csv"):
    df = read_table(text.encode(), name).df
    long, _ = to_long(df, guess_spec(df))
    return long


# ------------------------------------------------------------------ tablas
@pytest.mark.parametrize("n,expected", [(12, 8), (18, 10), (24, 13), (30, 15), (36, 18)])
def test_triangular_minimum_matches_table(n, expected):
    assert core.binomial_min_required(n, 1 / 3, 0.05) == expected


@pytest.mark.parametrize("n,expected", [(18, 13), (24, 17), (30, 20)])
def test_duo_trio_minimum_matches_table(n, expected):
    assert core.binomial_min_required(n, 0.5, 0.05) == expected


@pytest.mark.parametrize("n,expected", [(20, 15), (30, 21), (40, 27)])
def test_paired_preference_two_tailed(n, expected):
    assert core.binomial_min_required(n, 0.5, 0.05, "two-sided") == expected


def test_guide_example_28_of_40():
    r = core.binomial_test(28, 40, 0.5, alternative="two-sided")
    assert round(r.p_value, 3) == 0.017  # la guía reporta p = 0,017


# ------------------------------------------------------------------ métodos
def test_anova_one_way_equals_scipy():
    d = pd.DataFrame({"juez": range(15), "muestra": list("AAAAABBBBBCCCCC"),
                      "valor": [5, 6, 5, 7, 6, 7, 8, 7, 6, 8, 4, 5, 4, 3, 5]})
    an = core.anova_samples(d)
    f, p = stats.f_oneway(*[g["valor"] for _, g in d.groupby("muestra")])
    assert an.design == "un factor"
    assert np.isclose(an.f, f) and np.isclose(an.p, p)


def test_tukey_one_way_equals_scipy():
    d = pd.DataFrame({"juez": range(15), "muestra": list("AAAAABBBBBCCCCC"),
                      "valor": [5, 6, 5, 7, 6, 7, 8, 7, 6, 8, 4, 5, 4, 3, 5]})
    an = core.anova_samples(d)
    pairs, letters, _ = core.tukey_hsd(an.means["media"], an.means["n"], an.mse, an.df_error)
    ref = stats.tukey_hsd(*[g["valor"].values for _, g in d.groupby("muestra")])
    assert np.isclose(pairs.loc[0, "p"], ref.pvalue[0, 1], atol=1e-3)
    assert letters["B"] != letters["C"]


def test_friedman_equals_scipy():
    m = pd.DataFrame({"A": [1, 1, 2, 1, 1, 2], "B": [2, 3, 1, 2, 3, 1], "C": [3, 2, 3, 3, 2, 3]})
    r = core.friedman(m)
    assert np.isclose(r.p, stats.friedmanchisquare(m.A, m.B, m.C).pvalue)


def test_cochran_known_value():
    from statsmodels.stats.contingency_tables import cochrans_q
    m = pd.DataFrame({"A": [1, 1, 0, 1, 1, 0, 1], "B": [0, 0, 0, 1, 0, 0, 0], "C": [1, 1, 1, 1, 0, 1, 1]})
    assert np.isclose(core.cochran_q(m).q, cochrans_q(m.values).statistic)


def test_compact_letters():
    means = pd.Series({"A": 7.4, "B": 6.8, "C": 5.7})
    sig = {("A", "B"): False, ("A", "C"): True, ("B", "C"): True}
    assert core.compact_letters(means, sig) == {"A": "a", "B": "a", "C": "b"}


# ------------------------------------------------------------------ detección
def test_examples_are_classified():
    expected = {"afectiva_hedonica_3_muestras": ESCALA, "afectiva_preferencia_pareada": CATEGORICO,
                "discriminativa_triangular": BINARIO, "ordenamiento_friedman": RANGO,
                "afectiva_compra_si_no_cochran": BINARIO}
    for name, kind in expected.items():
        long = load((EX / f"{name}.csv").read_text())
        assert prepare(long, AFECTIVA, (1, 9)).profile.main_kind == kind, name


def test_robust_scale_ignores_typo():
    vals = pd.Series([5, 6, 7, 8, 7, 6, 77, 4, 9, 3])
    assert suggest_scale(*robust_range(vals)) == (1, 9)


# ------------------------------------------------------------------ errores
def test_empty_file():
    with pytest.raises(DataLoadError):
        read_table(b"", "vacio.csv")


def test_single_column_file():
    with pytest.raises(DataLoadError):
        read_table(b"a\n1\n2\n", "x.csv")


def test_semicolon_and_decimal_comma():
    df = read_table("juez;muestra;nota\n1;A;7,5\n1;B;6,0\n".encode("latin-1"), "x.csv").df
    assert df["nota"].tolist() == [7.5, 6.0]


def test_miscoded_scale_and_missing_judge():
    text = "juez,muestra,agrado\n1,A,7\n1,B,siete\n2,A,8\n2,B,99\n3,A,6\n3,B,5\n4,A,7\n"
    prep = prepare(load(text), AFECTIVA, (1, 9))
    titles = " ".join(i.title for i in prep.issues)
    assert "mal codificada" in titles
    assert "fuera de la escala" in titles
    assert "no evaluaron todas" in titles
    assert 99 not in prep.data["valor"].tolist()


def test_bad_ranks_detected():
    text = "juez,A,B,C\n1,1,2,3\n2,1,1,3\n3,2,1,3\n4,3,2,1\n"
    prep = prepare(load(text), AFECTIVA)
    assert any("Rangos mal codificados" in i.title for i in prep.issues)


def test_no_valid_data_is_error():
    text = "juez,muestra,agrado\n1,A,\n2,A,\n"
    df = read_table(text.encode(), "x.csv").df
    long, _ = to_long(df, guess_spec(df))
    prep = prepare(long, AFECTIVA, (1, 9))
    assert any(i.level == ERROR for i in prep.issues)


def test_methods_never_crash_on_bad_input():
    """Todos los métodos sobre datos inadecuados devuelven error legible, no excepción."""
    text = "juez,muestra,agrado\n1,A,7\n2,A,8\n"
    prep = prepare(load(text), AFECTIVA, (1, 9))
    ids = ["ia", "t", "anova", "friedman", "binomial", "chi2_gof", "chi2_2x2", "cochran", "pca"]
    res = run_methods(prep, ids, AnalysisParams(), AFECTIVA)
    assert len(res) >= len(ids)


@pytest.mark.parametrize("name,tt,method", [
    ("afectiva_hedonica_3_muestras", AFECTIVA, "anova"),
    ("afectiva_2_muestras_ancho", AFECTIVA, "t"),
    ("afectiva_preferencia_pareada", AFECTIVA, "binomial"),
    ("discriminativa_triangular", DISCRIMINATIVA, "binomial"),
    ("discriminativa_a_no_a", DISCRIMINATIVA, "chi2_2x2"),
    ("ordenamiento_friedman", DISCRIMINATIVA, "friedman"),
    ("descriptiva_qda", DESCRIPTIVA, "anova"),
    ("afectiva_compra_si_no_cochran", AFECTIVA, "cochran"),
])
def test_recommender(name, tt, method):
    long = load((EX / f"{name}.csv").read_text())
    prep = prepare(long, tt, (0, 15) if "qda" in name else (1, 9))
    recs = {r.method_id: r.status for r in recommend(prep, tt)}
    assert recs[method] == RECOMENDADO
