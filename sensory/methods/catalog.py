"""Catálogo de métodos: cada uno calcula, grafica e interpreta para el producto."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .. import plots
from ..constants import (AFECTIVA, BINARIO, CATEGORICO, DESCRIPTIVA, DISCRIMINATIVA, ESCALA,
                         PROTOCOLS, RANGO)
from ..interpret import fmt, fmt_p, fmt_pct, hedonic_label, join_es, sig_phrase
from ..io_utils import normalize_text
from ..validation import PreparedData
from . import core


# ==========================================================================
# Estructuras
# ==========================================================================
@dataclass
class AnalysisParams:
    alpha: float = 0.05
    scale: tuple[float, float] = (1, 9)
    ia_threshold: float = 70.0
    protocol: str = "triangular"
    rank_one_is_best: bool = True
    product_name: str = "el producto"


@dataclass
class MethodResult:
    method_id: str
    title: str
    question: str = ""
    metrics: list[tuple[str, str]] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    significant: bool | None = None
    interpretation: str = ""
    decision: str = ""
    figures: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class MethodSpec:
    id: str
    name: str
    what: str            # qué hace (en fácil)
    decides: str         # cómo se decide
    run: Callable[[PreparedData, AnalysisParams, str], list[MethodResult]]


def _error(method_id: str, title: str, msg: str) -> MethodResult:
    return MethodResult(method_id, title, error=msg)


def _attrs(prep: PreparedData, *kinds: str) -> list[str]:
    return [a.name for a in prep.profile.attributes if a.kind in kinds]


def _sub(prep: PreparedData, attr: str) -> pd.DataFrame:
    return prep.data[prep.data["atributo"] == attr]


def _order(prep: PreparedData, sub: pd.DataFrame) -> list[str]:
    present = set(sub["muestra"].unique())
    return [s for s in prep.profile.samples if s in present]


def _attr_label(attr: str) -> str:
    return "agrado" if normalize_text(attr) in ("respuesta", "valor", "calificacion", "nota", "puntaje") else attr


def _matrix(sub: pd.DataFrame, order: list[str]) -> tuple[pd.DataFrame, int]:
    """Matriz jueces × muestras (solo jueces completos) y nº de jueces excluidos."""
    pivot = sub.pivot_table(index="juez", columns="muestra", values="valor", aggfunc="first")
    pivot = pivot.reindex(columns=order)
    complete = pivot.dropna()
    return complete, len(pivot) - len(complete)


def detect_protocol(prep: PreparedData, default: str) -> str:
    """Intenta reconocer el protocolo discriminativo por los nombres de columnas/muestras."""
    text = normalize_text(" ".join([*prep.profile.samples, *[a.name for a in prep.profile.attributes]]))
    rules = [(r"triang", "triangular"), (r"duo|trio", "duo_trio"), (r"2 ?de ?5|dos de cinco", "dos_de_cinco"),
             (r"tetrad", "tetrada"), (r"parea|pareada|direccional", "pareada_dir")]
    for pattern, key in rules:
        if re.search(pattern, text):
            return key
    return default


# ==========================================================================
# 1. Media + Índice de aceptabilidad
# ==========================================================================
def run_acceptability(prep: PreparedData, prm: AnalysisParams, test_type: str) -> list[MethodResult]:
    out = []
    lo, hi = prm.scale
    for attr in _attrs(prep, ESCALA):
        sub = _sub(prep, attr)
        order = _order(prep, sub)
        rows = []
        for s in order:
            r = core.acceptability(sub.loc[sub["muestra"] == s, "valor"], hi, lo, prm.ia_threshold)
            rows.append({"Muestra": s, "n": r["n"], "Media": r["media"], "DE": r["de"],
                         "IA (%)": r["ia"], f"% jueces > {fmt((lo + hi) / 2, 1)}": r["pct_agrado"],
                         "p (media > umbral)": r["p_supera"], "¿Aceptada?": "Sí" if r["aceptado"] else "No"})
        table = pd.DataFrame(rows)
        lines = []
        for _, r in table.iterrows():
            lab = hedonic_label(r["Media"], prm.scale)
            lab_txt = f" (≈ «{lab}»)" if lab else ""
            verdict = "ACEPTADA" if r["¿Aceptada?"] == "Sí" else "NO ACEPTADA"
            conf = ""
            if not np.isnan(r["p (media > umbral)"]):
                conf = (" y la media supera el umbral con significancia estadística"
                        if r["p (media > umbral)"] < prm.alpha else
                        ", aunque estadísticamente no se puede asegurar que supere el umbral")
            lines.append(f"• Muestra **{r['Muestra']}**: media {fmt(r['Media'])}{lab_txt} sobre {fmt(hi, 0)}, "
                         f"IA = {fmt_pct(r['IA (%)'])} → **{verdict}** (umbral {fmt(prm.ia_threshold, 0)} %){conf}. "
                         f"El {fmt_pct(r[table.columns[5]], 0)} de los jueces calificó en la mitad positiva de la escala.")
        accepted = table.loc[table["¿Aceptada?"] == "Sí", "Muestra"].tolist()
        rejected = table.loc[table["¿Aceptada?"] == "No", "Muestra"].tolist()
        interp = (f"El índice de aceptabilidad (IA = media ÷ puntaje máximo × 100) indica qué tanto gusta "
                  f"{prm.product_name} en «{_attr_label(attr)}».\n\n" + "\n".join(lines))
        if len(order) > 1:
            interp += ("\n\nNota: el IA describe cada muestra por separado; para saber si las diferencias entre "
                       "muestras son reales usa t de Student (2 muestras) o ANOVA + Tukey (3 o más).")
        if accepted and not rejected:
            decision = (f"{'La muestra cumple' if len(order) == 1 else 'Todas las muestras cumplen'} el criterio de "
                        f"aceptación: se puede avanzar a la siguiente etapa (vida útil, escalamiento o prueba de mercado).")
        elif rejected and not accepted:
            decision = ("Ninguna muestra alcanza el umbral de aceptación: reformular. Revisa los atributos con menor "
                        "agrado y los comentarios de los consumidores antes de una nueva prueba.")
        else:
            decision = (f"Continuar con {join_es(accepted)} (aceptadas) y reformular o descartar {join_es(rejected)}.")
        figs = [plots.acceptability_bars(table.set_index("Muestra")["IA (%)"], prm.ia_threshold,
                                         f"Índice de aceptabilidad · {_attr_label(attr)}")]
        dist = plots.score_distribution(sub, order, prm.scale, f"Distribución de calificaciones · {_attr_label(attr)}")
        if dist is not None:
            figs.append(dist)
        metrics = [(f"IA {r['Muestra']}", fmt_pct(r["IA (%)"])) for _, r in table.iterrows()][:4]
        out.append(MethodResult("ia", f"Media e índice de aceptabilidad · {attr}",
                                "¿El producto gusta lo suficiente?", metrics,
                                [("Aceptabilidad por muestra", table)], bool(accepted), interp, decision, figs,
                                [f"Criterio: aceptado si IA ≥ {fmt(prm.ia_threshold, 0)} %. "
                                 f"La columna «p (media > umbral)» es una t de una muestra contra "
                                 f"{fmt(prm.ia_threshold / 100 * hi)} puntos."]))
    return out or [_error("ia", "Índice de aceptabilidad", "No hay atributos con escala numérica.")]


# ==========================================================================
# 2. t de Student
# ==========================================================================
def run_ttest(prep: PreparedData, prm: AnalysisParams, test_type: str) -> list[MethodResult]:
    out = []
    for attr in _attrs(prep, ESCALA):
        sub = _sub(prep, attr)
        order = _order(prep, sub)
        title = f"t de Student · {attr}"
        if len(order) != 2:
            out.append(_error("t", title, f"La t de Student compara exactamente 2 muestras; hay {len(order)}. "
                                          "Usa ANOVA + Tukey."))
            continue
        try:
            r = core.ttest_two(sub, order)
        except ValueError as exc:
            out.append(_error("t", title, str(exc)))
            continue
        a, b = order
        sig = r["p"] < prm.alpha
        hi_s, lo_s = (a, b) if r["media_a"] >= r["media_b"] else (b, a)
        means = pd.Series({a: r["media_a"], b: r["media_b"]})
        tbl = pd.DataFrame({"Muestra": [a, b], "n": [r.get("n_pares", r["n_a"]), r.get("n_pares", r["n_b"])],
                            "Media": [r["media_a"], r["media_b"]], "DE": [r["de_a"], r["de_b"]],
                            "Grupo": ["a", "b" if sig else "a"] if r["media_a"] >= r["media_b"]
                            else ["b" if sig else "a", "a"]})
        stats_tbl = pd.DataFrame([{"Tipo": r["tipo"], "t": r["t"], "gl": r["gl"], "t crítico (bilateral)": r["t_critico"],
                                   "p": r["p"], "Diferencia de medias": r["dif_media"]}])
        label = _attr_label(attr)
        design = (f"Prueba **pareada**: los mismos {r['n_pares']} jueces evaluaron ambas muestras, por lo que se "
                  f"compara la diferencia dentro de cada juez." if r["tipo"] == "pareada" else
                  f"Prueba para **grupos independientes** ({r['tipo']}): jueces distintos evaluaron cada muestra.")
        interp = (f"{design}\n\nMedia de {a}: {fmt(r['media_a'])} · media de {b}: {fmt(r['media_b'])} "
                  f"(diferencia {fmt(abs(r['dif_media']))} puntos). t = {fmt(r['t'])} con {fmt(r['gl'], 0 if float(r['gl']).is_integer() else 1)} gl "
                  f"(t crítico = {fmt(r['t_critico'])}); {sig_phrase(r['p'], prm.alpha)}.")
        if test_type == AFECTIVA:
            if sig:
                interp += f"\n\n**{hi_s} gusta más que {lo_s}** en {label}."
                decision = (f"Elegir {hi_s}. Si {hi_s} es tu prototipo, supera a {lo_s} y está listo para avanzar; "
                            f"si {lo_s} es tu prototipo, debe mejorarse en {label} antes de competir con {hi_s}.")
            else:
                interp += f"\n\nLos consumidores **no distinguen en agrado** entre {a} y {b}."
                decision = ("Ambas muestras gustan igual: elige con criterios técnicos o económicos (costo, vida útil, "
                            "perfil nutricional). Si una es la versión mejorada (p. ej. menos azúcar), el cambio no "
                            "afectó el agrado.")
        else:
            if sig:
                interp += f"\n\n**{hi_s} tiene mayor intensidad de {label}** que {lo_s}."
                decision = f"Las muestras difieren en {label}; ajusta la formulación si buscas igualar a la referencia."
            else:
                interp += f"\n\nEl panel no encontró diferencia en {label} entre {a} y {b}."
                decision = f"Las muestras son equivalentes en {label}."
        notes = []
        if r.get("p_normalidad") is not None and r["p_normalidad"] < 0.05:
            notes.append(f"Las diferencias no siguen una distribución normal (Shapiro {fmt_p(r['p_normalidad'])}). "
                         "Con muestras pequeñas confirma con una prueba de rangos (Friedman/Wilcoxon).")
        if "p_levene" in r and r["p_levene"] < 0.05:
            notes.append("Las varianzas difieren (Levene); se aplicó la corrección de Welch.")
        figs = [plots.means_with_letters(means, sub.groupby("muestra")["valor"].sem().reindex(order),
                                         dict(zip(tbl["Muestra"], tbl["Grupo"])),
                                         f"Media ± EE · {label}", "Media", prm.scale),
                plots.boxplot(sub, order, f"Distribución por muestra · {label}", "Calificación")]
        out.append(MethodResult("t", title, "¿Una muestra gusta/es más intensa que la otra?",
                                [("t", fmt(r["t"])), ("p", fmt_p(r["p"]).replace("p ", "")),
                                 ("Diferencia", fmt(abs(r["dif_media"])))],
                                [("Medias", tbl), ("Prueba t", stats_tbl)], sig, interp, decision, figs, notes))
    return out or [_error("t", "t de Student", "No hay atributos con escala numérica.")]


# ==========================================================================
# 3. ANOVA + Tukey
# ==========================================================================
def run_anova(prep: PreparedData, prm: AnalysisParams, test_type: str) -> list[MethodResult]:
    out, summary_rows, profile_means = [], [], {}
    for attr in _attrs(prep, ESCALA):
        sub = _sub(prep, attr)
        order = _order(prep, sub)
        title = f"ANOVA + Tukey · {attr}"
        try:
            an = core.anova_samples(sub, prm.alpha)
        except ValueError as exc:
            out.append(_error("anova", title, str(exc)))
            continue
        means = an.means["media"].reindex(order)
        sig = an.p < prm.alpha
        tables = [("Tabla ANOVA", an.table)]
        if sig and len(order) >= 2:
            pairs, letters, hsd = core.tukey_hsd(means, an.means["n"].reindex(order), an.mse, an.df_error, prm.alpha)
            tables.append(("Comparaciones de Tukey (HSD)", pairs))
        else:
            letters, hsd, pairs = {s: "a" for s in order}, None, None
        mt = pd.DataFrame({"Muestra": order, "n": an.means["n"].reindex(order).values,
                           "Media": means.values, "DE": an.means["de"].reindex(order).values,
                           "Grupo Tukey": [letters[s] for s in order]}).sort_values("Media", ascending=False)
        tables.insert(1, ("Medias y grupos (misma letra = no difieren)", mt))
        label = _attr_label(attr)
        best = mt.iloc[0]
        worst = mt.iloc[-1]
        lab = hedonic_label(best["Media"], prm.scale) if test_type == AFECTIVA else None
        interp = (f"ANOVA de {'bloques al azar (el juez es un bloque, así se elimina la diferencia en el uso de la escala entre jueces)' if 'bloques' in an.design else 'un factor'} "
                  f"con {an.n_judges_used} jueces: F = {fmt(an.f)} (F crítico = {fmt(an.f_crit)}); "
                  f"{sig_phrase(an.p, prm.alpha)}.")
        if sig:
            top = mt.loc[mt["Grupo Tukey"].str.contains(best["Grupo Tukey"][0]), "Muestra"].tolist()
            interp += (f"\n\nComo la ANOVA detectó diferencia, Tukey (DMS ≈ {fmt(hsd)} puntos) indica cuáles difieren: "
                       f"**{best['Muestra']}** tiene la media más alta ({fmt(best['Media'])}"
                       f"{f', ≈ «{lab}»' if lab else ''}) y **{worst['Muestra']}** la más baja ({fmt(worst['Media'])}).")
            if len(top) > 1:
                interp += f" {join_es(top)} comparten la letra «{best['Grupo Tukey'][0]}»: estadísticamente son iguales."
            diff_pairs = pairs[pairs["¿Difieren?"] == "Sí"]
            if not diff_pairs.empty:
                interp += " Pares distintos: " + "; ".join(
                    f"{r['Muestra 1']} vs {r['Muestra 2']}" for _, r in diff_pairs.iterrows()) + "."
        else:
            interp += "\n\nNo se hace Tukey porque la ANOVA no encontró diferencias (todas las muestras comparten la letra «a»)."
        judge_row = an.table[an.table["Fuente"].str.startswith("Jueces")]
        if not judge_row.empty and judge_row["p"].iloc[0] < prm.alpha:
            interp += ("\n\nEl efecto juez también fue significativo: los jueces usan la escala de forma distinta "
                       "(unos califican más alto que otros). Es normal; el diseño de bloques ya lo controla.")
        if test_type == AFECTIVA:
            if sig:
                top = mt.loc[mt["Grupo Tukey"].str.contains(best["Grupo Tukey"][0]), "Muestra"].tolist()
                decision = (f"Seleccionar {join_es(top)} (mayor agrado). " +
                            ("Al ser estadísticamente iguales, decide entre ellas por costo, proceso o perfil nutricional. "
                             if len(top) > 1 else "") +
                            f"Descartar o reformular {worst['Muestra']}.")
            else:
                decision = ("Ninguna formulación gusta más que otra: elige la de menor costo o mejor perfil "
                            "técnico/nutricional; el cambio entre formulaciones no afecta el agrado.")
        else:
            decision = (f"El atributo {label} sí diferencia a las muestras: {best['Muestra']} es la de mayor intensidad "
                        f"y {worst['Muestra']} la de menor." if sig else
                        f"Las muestras tienen la misma intensidad de {label}; este atributo no las diferencia.")
        notes = [f"Diseño: {an.design}. α = {fmt(prm.alpha)}. CM error = {fmt(an.mse, 3)} con {fmt(an.df_error, 0)} gl."]
        if an.p_shapiro is not None and an.p_shapiro < 0.05:
            notes.append(f"Los residuos no son normales (Shapiro {fmt_p(an.p_shapiro)}): confirma el resultado con "
                         "Friedman (prueba de rangos, no exige normalidad).")
        if an.p_levene is not None and an.p_levene < 0.05:
            notes.append(f"Las varianzas entre muestras difieren (Levene {fmt_p(an.p_levene)}); interpreta con cautela.")
        figs = [plots.means_with_letters(means, an.means["ee"].reindex(order), letters,
                                         f"Media ± EE y grupos de Tukey · {label}", "Media", prm.scale),
                plots.boxplot(sub, order, f"Distribución por muestra · {label}", "Calificación")]
        out.append(MethodResult("anova", title, "¿Alguna muestra difiere? ¿Cuáles?",
                                [("F", fmt(an.f)), ("p", fmt_p(an.p).replace("p ", "")),
                                 ("Media más alta", str(best["Muestra"]))],
                                tables, sig, interp, decision, figs, notes))
        summary_rows.append({"Atributo": attr, "F": an.f, "p": an.p, "¿Difieren?": "Sí" if sig else "No",
                             **{s: f"{fmt(means[s])} {letters[s]}" for s in order}})
        profile_means[attr] = means

    if len(summary_rows) >= 2:
        summ = pd.DataFrame(summary_rows)
        diff_attrs = summ.loc[summ["¿Difieren?"] == "Sí", "Atributo"].tolist()
        pm = pd.DataFrame(profile_means)
        figs = []
        rad = plots.radar(pm, "Perfil sensorial (medias)", prm.scale)
        if rad is not None:
            figs.append(rad)
        interp = (f"Se analizaron {len(summ)} atributos con una ANOVA cada uno. "
                  + (f"Las muestras difieren en: **{join_es(diff_attrs)}**." if diff_attrs
                     else "En ningún atributo hubo diferencias significativas."))
        out.insert(0, MethodResult(
            "anova", "Resumen del perfil (ANOVA por atributo)", "¿En qué atributos difieren las muestras?",
            [("Atributos con diferencia", f"{len(diff_attrs)} de {len(summ)}")],
            [("Medias por atributo (letras de Tukey)", summ)], bool(diff_attrs), interp,
            ("Concentra la reformulación en los atributos que sí diferencian a las muestras: " + join_es(diff_attrs) + "."
             if diff_attrs else "Las muestras tienen un perfil sensorial equivalente."), figs))
    return out or [_error("anova", "ANOVA + Tukey", "No hay atributos con escala numérica.")]


# ==========================================================================
# 4. Friedman
# ==========================================================================
def run_friedman(prep: PreparedData, prm: AnalysisParams, test_type: str) -> list[MethodResult]:
    out = []
    attrs = _attrs(prep, RANGO) or _attrs(prep, ESCALA)
    for attr in attrs:
        sub = _sub(prep, attr)
        order = _order(prep, sub)
        kind = next(a.kind for a in prep.profile.attributes if a.name == attr)
        title = f"Friedman · {attr}"
        mat, dropped = _matrix(sub, order)
        is_rank = kind == RANGO
        higher_is_more = (not prm.rank_one_is_best) if is_rank else True
        try:
            fr = core.friedman(mat, prm.alpha, higher_is_more)
        except ValueError as exc:
            out.append(_error("friedman", title, str(exc)))
            continue
        sig = fr.p < prm.alpha
        R = fr.rank_sums.reindex(order)
        ranked = R.sort_values(ascending=not higher_is_more)
        label = _attr_label(attr)
        if is_rank:
            meaning = "mayor agrado/intensidad" if prm.rank_one_is_best else "menor agrado/intensidad"
            how = f"En la boleta el lugar 1 = {meaning}; "
            best_word = "menor" if prm.rank_one_is_best else "mayor"
        else:
            how = "Las notas se convirtieron en rangos dentro de cada juez (1 = nota más baja); "
            best_word = "mayor"
        first, last = ranked.index[0], ranked.index[-1]
        concept = "agrado" if test_type == AFECTIVA else ("intensidad" if label == "agrado" else label)
        tbl = pd.DataFrame({"Muestra": order, "Suma de rangos": R.values,
                            "Rango medio": (R / fr.n_judges).values,
                            "Grupo": [fr.letters[s] for s in order]})
        interp = (f"{how}la muestra con {best_word} suma de rangos es la que ocupa el primer lugar en {concept}.\n\n"
                  f"Con {fr.n_judges} jueces y {len(order)} muestras: T de Friedman = {fmt(fr.statistic)} "
                  f"(χ² crítico con {fr.gl} gl = {fmt(fr.chi2_crit)}); {sig_phrase(fr.p, prm.alpha)}.")
        if sig:
            interp += (f"\n\nEl orden no es producto del azar. **{first}** ocupa el primer lugar y **{last}** el último. "
                       f"Diferencia mínima significativa entre sumas de rangos: {fmt(fr.lsd, 1)} "
                       f"(muestras con la misma letra no difieren).")
            if test_type == AFECTIVA:
                decision = f"Preferir {first}; {last} es la menos aceptada y debería reformularse o descartarse."
            elif test_type == DISCRIMINATIVA:
                decision = (f"Los jueces sí perciben diferencias de {concept} entre las muestras ({first} > {last}). "
                            "Si el cambio debía pasar desapercibido, ajusta la formulación.")
            else:
                decision = f"Las muestras difieren en {concept}: {first} es la de mayor nivel."
        else:
            interp += "\n\nEl orden asignado por los jueces podría deberse al azar."
            decision = (f"No hay una muestra que destaque en {concept}; puedes elegir por criterios técnicos o de costo."
                        if test_type != DISCRIMINATIVA else
                        f"Los jueces no perciben diferencias de {concept}: el cambio pasa desapercibido.")
        notes = []
        if dropped:
            notes.append(f"Se excluyeron {dropped} juez(es) que no evaluaron todas las muestras (Friedman exige bloques completos).")
        if fr.ties:
            notes.append("Hubo empates dentro de algunos jueces: se usaron rangos promedio y la corrección por empates.")
        fig = plots.rank_sums(R, fr.letters, f"Suma de rangos · {label}",
                              f"{'Menor' if best_word == 'menor' else 'Mayor'} suma = primer lugar")
        out.append(MethodResult("friedman", title, "¿El orden entre las muestras es real o azar?",
                                [("T", fmt(fr.statistic)), ("p", fmt_p(fr.p).replace("p ", "")), ("Primer lugar", first)],
                                [("Sumas de rangos", tbl), ("Comparaciones (DMS de rangos)", fr.pairs)],
                                sig, interp, decision, [fig], notes))
    return out or [_error("friedman", "Friedman", "No hay datos de rangos ni de escala.")]


# ==========================================================================
# 5. Binomial
# ==========================================================================
def _choice_counts(sub: pd.DataFrame) -> pd.Series:
    return sub["categoria"].dropna().value_counts()


def run_binomial(prep: PreparedData, prm: AnalysisParams, test_type: str) -> list[MethodResult]:
    out = []
    # Preferencia (elección categórica de 2 opciones)
    for attr in [a.name for a in prep.profile.attributes if a.kind == CATEGORICO and not a.a_no_a]:
        sub = _sub(prep, attr)
        counts = _choice_counts(sub)
        title = f"Binomial de preferencia · {attr}"
        if len(counts) != 2:
            out.append(_error("binomial", title, f"La binomial de preferencia necesita 2 opciones; hay {len(counts)} "
                                                 f"({join_es(list(counts.index))}). Usa Ji² de bondad de ajuste."))
            continue
        winner, loser = counts.index[0], counts.index[1]
        n = int(counts.sum())
        r = core.binomial_test(int(counts.iloc[0]), n, 0.5, prm.alpha, "two-sided")
        sig = r.p_value < prm.alpha
        tbl = pd.DataFrame([{"Opción": k, "Elecciones": int(v), "%": v / n * 100} for k, v in counts.items()])
        stats_tbl = pd.DataFrame([{"n": n, "Mayor conteo": r.successes, "Mínimo tabla (bilateral)": r.min_required,
                                   "p (bilateral)": r.p_value}])
        interp = (f"{r.successes} de {n} consumidores ({fmt_pct(r.proportion * 100)}) prefirieron **{winner}**. "
                  f"Por azar se esperaría la mitad ({fmt(n / 2, 1)}). Con n = {n} y α = {fmt(prm.alpha)} el mínimo de "
                  f"la tabla (bilateral) es {r.min_required}; {sig_phrase(r.p_value, prm.alpha)}.")
        decision = (f"Sí hay preferencia por {winner}: es la opción a lanzar o el estándar a igualar." if sig else
                    f"No hay preferencia clara entre {winner} y {loser}: los consumidores las consideran equivalentes.")
        figs = [plots.binomial_bars(r.successes, n, r.min_required, n / 2,
                                    f"Prefieren {winner}", label_ok=f"eligieron {winner}")]
        out.append(MethodResult("binomial", title, "¿Prefieren una muestra sobre la otra?",
                                [("Prefieren " + str(winner), f"{r.successes}/{n}"),
                                 ("p", fmt_p(r.p_value).replace("p ", ""))],
                                [("Conteo de elecciones", tbl), ("Prueba binomial", stats_tbl)],
                                sig, interp, decision, figs))

    # Aciertos / fallos (discriminativas) o aceptación sí/no
    for attr in _attrs(prep, BINARIO):
        sub = _sub(prep, attr)
        for sample, g in sub.groupby("muestra", sort=False):
            vals = g["valor"].dropna()
            n, x = len(vals), int(vals.sum())
            label = f"{attr}" + (f" · {sample}" if sample != "Prueba" else "")
            if test_type == AFECTIVA:
                proto_name, p0, alt = "Aceptación (sí/no) vs. 50 %", 0.5, "greater"
            else:
                proto = PROTOCOLS[prm.protocol]
                proto_name, p0, alt = proto.name, proto.p0, proto.alternative
            try:
                r = core.binomial_test(x, n, p0, prm.alpha, alt)
            except ValueError as exc:
                out.append(_error("binomial", f"Binomial · {label}", str(exc)))
                continue
            sig = r.p_value < prm.alpha
            stats_tbl = pd.DataFrame([{"Protocolo": proto_name, "n": n, "Aciertos": x, "% aciertos": r.proportion * 100,
                                       "Esperado por azar": n * p0, "p azar": p0,
                                       "Mínimo tabla": r.min_required, "p": r.p_value,
                                       "Discriminadores estimados (%)": (r.pd_estimate or 0) * 100}])
            if test_type == AFECTIVA:
                interp = (f"{x} de {n} jueces ({fmt_pct(r.proportion * 100)}) respondieron «sí» para {sample}. "
                          f"{sig_phrase(r.p_value, prm.alpha).capitalize()} respecto a un 50 %.")
                decision = (f"La mayoría acepta {sample}." if sig else
                            f"No se puede afirmar que la mayoría acepte {sample}; revisar la formulación.")
            else:
                interp = (f"Prueba **{proto_name}** (probabilidad de acertar por azar = {fmt(p0, 3)}): "
                          f"{x} de {n} jueces acertaron ({fmt_pct(r.proportion * 100)}); por azar se esperaban "
                          f"{fmt(n * p0, 1)}. Con n = {n} y α = {fmt(prm.alpha)} la tabla binomial exige al menos "
                          f"**{r.min_required}** aciertos. {sig_phrase(r.p_value, prm.alpha).capitalize()}.")
                if sig:
                    interp += (f"\n\nLos jueces **sí perciben una diferencia** entre las muestras. Se estima que ≈ "
                               f"{fmt_pct((r.pd_estimate or 0) * 100, 0)} del panel la detecta realmente (corrección de Abbott).")
                    decision = ("La diferencia es perceptible. Si el objetivo era un cambio imperceptible (reducir azúcar, "
                                "cambiar proveedor o materia prima), ajusta la formulación y repite la prueba. Si buscabas "
                                "una mejora perceptible, el objetivo se cumple: continúa con una prueba afectiva.")
                else:
                    interp += ("\n\nLos aciertos están dentro de lo esperable por azar: **no se detectó diferencia** "
                               "perceptible.")
                    decision = ("El cambio puede implementarse sin que el consumidor lo note. Ojo: «no significativo» no "
                                f"prueba que sean idénticas; con {n} jueces la prueba detecta solo diferencias moderadas.")
            figs = [plots.binomial_bars(x, n, r.min_required, n * p0, f"{proto_name} · {label}",
                                        "sí" if test_type == AFECTIVA else "aciertos")]
            out.append(MethodResult("binomial", f"Binomial ({proto_name}) · {label}",
                                    "¿Los aciertos superan lo esperado por azar?",
                                    [("Aciertos", f"{x}/{n}"), ("Mínimo tabla", str(r.min_required)),
                                     ("p", fmt_p(r.p_value).replace("p ", ""))],
                                    [("Prueba binomial", stats_tbl)], sig, interp, decision, figs,
                                    [] if n >= 18 or test_type == AFECTIVA else
                                    ["Menos de 18 jueces: la prueba tiene poca potencia (ISO recomienda 24 o más)."]))
    return out or [_error("binomial", "Binomial", "No hay respuestas binarias (acierto/fallo) ni de preferencia.")]


# ==========================================================================
# 6. Ji² de bondad de ajuste
# ==========================================================================
def run_chi2_gof(prep: PreparedData, prm: AnalysisParams, test_type: str) -> list[MethodResult]:
    out = []
    for attr in [a.name for a in prep.profile.attributes if a.kind == CATEGORICO and not a.a_no_a]:
        sub = _sub(prep, attr)
        counts = _choice_counts(sub)
        if len(counts) < 2:
            out.append(_error("chi2_gof", f"Ji² · {attr}", "Se necesitan al menos 2 opciones elegidas."))
            continue
        r = core.chi2_goodness(counts)
        sig = r["p"] < prm.alpha
        n = int(counts.sum())
        tbl = pd.DataFrame({"Opción": counts.index, "Observado": counts.values, "Esperado": r["esperados"],
                            "%": counts.values / n * 100})
        crit = float(sstats.chi2.ppf(1 - prm.alpha, r["gl"]))
        interp = (f"Se compara cuántas veces se eligió cada opción con lo esperado si todas gustaran igual "
                  f"({fmt(n / len(counts), 1)} cada una). χ² = {fmt(r['chi2'])} (crítico {fmt(crit)}, {r['gl']} gl); "
                  f"{sig_phrase(r['p'], prm.alpha)}.")
        top = counts.index[0]
        decision = (f"Las elecciones no son al azar: {top} es la más elegida ({fmt_pct(counts.iloc[0] / n * 100)})."
                    if sig else "Las elecciones se reparten como al azar: no hay una opción preferida.")
        notes = ["Algún esperado < 5: el χ² es aproximado; prefiere la binomial."] if r["min_esperado"] < 5 else []
        out.append(MethodResult("chi2_gof", f"Ji² de bondad de ajuste · {attr}", "¿Las elecciones difieren del azar?",
                                [("χ²", fmt(r["chi2"])), ("p", fmt_p(r["p"]).replace("p ", ""))],
                                [("Frecuencias", tbl)], sig, interp, decision,
                                [plots.proportion_bars((counts / n * 100), f"% de elecciones · {attr}",
                                                       ref=100 / len(counts), ref_label="azar")], notes))

    for attr in _attrs(prep, BINARIO):
        sub = _sub(prep, attr)
        for sample, g in sub.groupby("muestra", sort=False):
            vals = g["valor"].dropna()
            n, x = len(vals), int(vals.sum())
            if n == 0:
                continue
            if test_type == AFECTIVA:
                p0, alt, pname = 0.5, "greater", "Aceptación vs. 50 %"
            else:
                proto = PROTOCOLS[prm.protocol]
                p0, alt, pname = proto.p0, proto.alternative, proto.name
            r = core.chi2_goodness_two_cells(x, n, p0, alt)
            sig = r["p"] < prm.alpha
            label = attr + (f" · {sample}" if sample != "Prueba" else "")
            tbl = pd.DataFrame({"": ["Aciertos", "Fallos"], "Observado": r["observados"], "Esperado": r["esperados"]})
            interp = (f"Prueba {pname}: {x} aciertos de {n} frente a {fmt(n * p0, 1)} esperados por azar. "
                      f"χ² con corrección de Yates = {fmt(r['chi2'])} (1 gl, {'unilateral' if alt == 'greater' else 'bilateral'}); "
                      f"{sig_phrase(r['p'], prm.alpha)}. Es la aproximación de la prueba binomial; "
                      "si ambas discrepan, manda la binomial (exacta).")
            decision = ("Sí hay diferencia perceptible entre las muestras." if sig else
                        "No se detectó diferencia perceptible.")
            notes = ["Algún esperado < 5: el χ² no es confiable; usa la binomial."] if r["min_esperado"] < 5 else []
            out.append(MethodResult("chi2_gof", f"Ji² (aciertos vs. azar) · {label}", "¿Los aciertos superan el azar?",
                                    [("χ²", fmt(r["chi2"])), ("p", fmt_p(r["p"]).replace("p ", ""))],
                                    [("Observado vs. esperado", tbl)], sig, interp, decision, [], notes))
    return out or [_error("chi2_gof", "Ji²", "No hay conteos para analizar.")]


# ==========================================================================
# 7. Ji² tabla 2×2 (A – no A)
# ==========================================================================
def run_chi2_2x2(prep: PreparedData, prm: AnalysisParams, test_type: str) -> list[MethodResult]:
    out = []
    for attr in [a.name for a in prep.profile.attributes if a.a_no_a]:
        sub = _sub(prep, attr)
        ct = pd.crosstab(sub["muestra"], sub["categoria"]).reindex(index=["A", "no A"], columns=["A", "no A"],
                                                                     fill_value=0)
        title = f"Ji² A – no A · {attr}"
        try:
            r = core.chi2_2x2(ct)
        except ValueError as exc:
            out.append(_error("chi2_2x2", title, str(exc)))
            continue
        hit_a = ct.loc["A", "A"] / ct.loc["A"].sum() * 100
        fa = ct.loc["no A", "A"] / ct.loc["no A"].sum() * 100
        direction_ok = hit_a > fa
        sig = r["p"] < prm.alpha and direction_ok
        crit = float(sstats.chi2.ppf(1 - prm.alpha, 1))
        stats_tbl = pd.DataFrame([{"χ² (Pearson)": r["chi2"], "χ² crítico": crit, "p": r["p"],
                                   "χ² (Yates)": r["chi2_yates"], "p (Yates)": r["p_yates"],
                                   "p (Fisher, unilateral)": r["p_fisher"]}])
        interp = (f"Cuando se presentó **A**, el {fmt_pct(hit_a)} la reconoció como A; cuando se presentó **no A**, "
                  f"el {fmt_pct(fa)} también dijo «A». χ² = {fmt(r['chi2'])} (crítico {fmt(crit)}, 1 gl); "
                  f"{sig_phrase(r['p'], prm.alpha)}.")
        if r["p"] < prm.alpha and not direction_ok:
            interp += " Sin embargo, los jueces dicen «A» más cuando NO es A: la asociación va en sentido contrario, no hay discriminación válida."
        if sig:
            interp += "\n\nLos jueces **sí distinguen** la muestra A (original) de la nueva."
            decision = ("El cambio es perceptible: si debía pasar desapercibido, ajusta la formulación; si es una mejora, "
                        "verifica con una prueba afectiva que guste más.")
        else:
            interp += "\n\nLos jueces **no distinguen** A de no A mejor que el azar."
            decision = "El cambio no se percibe: se puede implementar sin afectar la identidad del producto."
        notes = []
        if r["min_esperado"] < 5:
            notes.append("Hay frecuencias esperadas < 5: usa el p de Fisher (exacto) en lugar del χ².")
        out.append(MethodResult("chi2_2x2", title, "¿Reconocen la muestra original frente a la nueva?",
                                [("χ²", fmt(r["chi2"])), ("p", fmt_p(r["p"]).replace("p ", "")),
                                 ("Reconocen A", fmt_pct(hit_a, 0))],
                                [("Tabla 2×2 (presentada × respuesta)", ct.reset_index().rename(columns={"muestra": "Presentada"})),
                                 ("Prueba Ji²", stats_tbl)], sig, interp, decision,
                                [plots.contingency_heatmap(ct, "Tabla A – no A")], notes))
    return out or [_error("chi2_2x2", "Ji² A – no A", "No se encontraron respuestas A / no A.")]


# ==========================================================================
# 8. Q de Cochran
# ==========================================================================
def run_cochran(prep: PreparedData, prm: AnalysisParams, test_type: str) -> list[MethodResult]:
    out = []
    for attr in _attrs(prep, BINARIO):
        sub = _sub(prep, attr)
        order = _order(prep, sub)
        title = f"Q de Cochran · {attr}"
        mat, dropped = _matrix(sub, order)
        try:
            r = core.cochran_q(mat, prm.alpha)
        except ValueError as exc:
            out.append(_error("cochran", title, str(exc)))
            continue
        sig = r.p < prm.alpha
        props = (r.proportions * 100).reindex(order)
        tbl = pd.DataFrame({"Muestra": order, "Jueces": r.n_judges, "Respuestas «sí»/aciertos": r.totals.reindex(order).values,
                            "%": props.values})
        best, worst = props.idxmax(), props.idxmin()
        what = "aceptación" if test_type == AFECTIVA else "aciertos"
        interp = (f"Los mismos {r.n_judges} jueces respondieron sí/no (o acierto/fallo) para {len(order)} muestras. "
                  f"Q = {fmt(r.q)} (χ² crítico con {r.gl} gl = {fmt(r.q_crit)}); {sig_phrase(r.p, prm.alpha)}.")
        if sig:
            interp += (f"\n\nLa proporción de {what} cambia según la muestra: **{best}** tiene la más alta "
                       f"({fmt_pct(props[best])}) y **{worst}** la más baja ({fmt_pct(props[worst])}).")
            decision = (f"Priorizar {best}; {worst} tiene una {what} significativamente menor." if test_type == AFECTIVA
                        else f"Las muestras no son igual de fáciles de identificar: {best} es la más reconocida.")
        else:
            decision = f"La proporción de {what} es similar en todas las muestras."
        notes = [f"Se excluyeron {dropped} juez(es) incompletos."] if dropped else []
        notes.append("Comparaciones por pares: McNemar exacto con corrección de Bonferroni.")
        out.append(MethodResult("cochran", title, f"¿La proporción de {what} difiere entre muestras?",
                                [("Q", fmt(r.q)), ("p", fmt_p(r.p).replace("p ", "")), ("Mayor %", str(best))],
                                [("Proporciones", tbl), ("Comparaciones por pares", r.pairs)], sig, interp, decision,
                                [plots.proportion_bars(props, f"% de {what} por muestra · {attr}")], notes))
    return out or [_error("cochran", "Q de Cochran", "No hay respuestas binarias.")]


# ==========================================================================
# 9. PCA
# ==========================================================================
def run_pca(prep: PreparedData, prm: AnalysisParams, test_type: str) -> list[MethodResult]:
    attrs = _attrs(prep, ESCALA)
    data = prep.data[prep.data["atributo"].isin(attrs)]
    means = data.pivot_table(index="muestra", columns="atributo", values="valor", aggfunc="mean")
    means = means.reindex(index=[s for s in prep.profile.samples if s in means.index], columns=attrs)
    try:
        r = core.pca(means)
    except ValueError as exc:
        return [_error("pca", "Mapa sensorial (PCA)", str(exc))]
    ev = r.explained * 100
    cp1 = r.loadings["CP1"].sort_values()
    pos = cp1[cp1 > 0.5].index.tolist()[::-1]
    neg = cp1[cp1 < -0.5].index.tolist()
    sc = r.scores.iloc[:, :2]
    dists = pd.DataFrame(np.sqrt(((sc.values[:, None, :] - sc.values[None, :, :]) ** 2).sum(-1)),
                         index=sc.index, columns=sc.index)
    near = []
    for s in sc.index:
        other = dists.loc[s].drop(s).idxmin()
        near.append(f"{s} ↔ {other}")
    interp = (f"El mapa resume {len(r.loadings)} atributos en 2 ejes que explican "
              f"{fmt(ev[:2].sum(), 1)} % de la variación entre muestras (CP1 = {fmt(ev[0], 1)} %).\n\n"
              f"CP1 (eje horizontal) va de " + (join_es(neg) if neg else "valores bajos") + " (izquierda) a "
              + (join_es(pos) if pos else "valores altos") + " (derecha). Las muestras cercanas en el mapa tienen "
              f"perfiles parecidos; las flechas indican hacia dónde aumenta cada atributo.\n\n"
              f"Muestra más parecida a cada una: {'; '.join(near)}.")
    right = sc["CP1"].idxmax()
    left = sc["CP1"].idxmin()
    decision = (f"{right} se caracteriza por {join_es(pos) if pos else 'los atributos de la derecha'}; "
                f"{left} por {join_es(neg) if neg else 'los atributos de la izquierda'}. Usa el mapa para decidir qué "
                "atributos modificar para acercar tu producto a la referencia o diferenciarlo de la competencia.")
    load_tbl = r.loadings.reset_index().rename(columns={"index": "Atributo", "atributo": "Atributo"})
    var_tbl = pd.DataFrame({"Componente": r.loadings.columns, "Varianza explicada (%)": ev,
                            "Acumulada (%)": np.cumsum(ev)})
    return [MethodResult("pca", "Mapa sensorial (PCA)", "¿Cómo se ubican las muestras según sus atributos?",
                         [("CP1 + CP2", fmt_pct(ev[:2].sum()))],
                         [("Varianza explicada", var_tbl), ("Cargas (correlación atributo–componente)", load_tbl),
                          ("Coordenadas de las muestras", r.scores.reset_index())],
                         None, interp, decision,
                         [plots.pca_biplot(r.scores, r.loadings, r.explained, "Mapa sensorial (biplot PCA)")],
                         ["PCA normado sobre las medias por muestra (cada atributo estandarizado)."])]


# ==========================================================================
# Registro
# ==========================================================================
METHODS: dict[str, MethodSpec] = {m.id: m for m in [
    MethodSpec("ia", "Media + Índice de aceptabilidad (IA %)",
               "Promedia las notas de agrado y las expresa como % del puntaje máximo.",
               "Aceptada si IA ≥ 70 %", run_acceptability),
    MethodSpec("t", "t de Student", "Compara el PROMEDIO de 2 muestras.",
               "p < α → una muestra gusta más / es más intensa", run_ttest),
    MethodSpec("anova", "ANOVA + Tukey",
               "ANOVA: ¿hay ALGUNA diferencia entre 3 o más muestras? Tukey: ¿CUÁLES difieren?",
               "p < α en la ANOVA → Tukey; misma letra = no difieren", run_anova),
    MethodSpec("friedman", "Friedman (rangos)",
               "Compara 3+ muestras cuando los datos son órdenes (o notas tratadas como orden).",
               "T > χ² crítico (p < α) → el orden es real", run_friedman),
    MethodSpec("binomial", "Prueba binomial",
               "Cuenta cuántos aciertan o eligen algo y lo compara con la SUERTE.",
               "Aciertos ≥ mínimo de la tabla (p < α)", run_binomial),
    MethodSpec("chi2_gof", "Ji² (χ²) de bondad de ajuste",
               "Compara los conteos observados con los esperados por azar.",
               "χ² > crítico (p < α)", run_chi2_gof),
    MethodSpec("chi2_2x2", "Ji² (χ²) tabla 2×2 · A – no A",
               "Compara lo que era (A / no A) con lo que dijeron los jueces.",
               "p < α → sí distinguen", run_chi2_2x2),
    MethodSpec("cochran", "Q de Cochran",
               "Compara proporciones sí/no (o acierto/fallo) de los mismos jueces en 3+ muestras.",
               "Q > χ² crítico (p < α)", run_cochran),
    MethodSpec("pca", "Mapa sensorial (PCA)",
               "Resume muchos atributos en un mapa de productos y atributos.",
               "Lectura visual: cercanía = perfiles parecidos", run_pca),
]}
