"""Gráficos (matplotlib) con estilo sobrio y paleta apta para daltonismo."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .constants import (COLOR_BAD, COLOR_GOOD, COLOR_GRID, COLOR_MUTED, COLOR_PRIMARY,  # noqa: E402
                        COLOR_TEXT, COLOR_TEXT_2, SERIES_COLORS)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": COLOR_GRID,
    "axes.labelcolor": COLOR_TEXT_2,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": COLOR_TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.8,
    "xtick.color": COLOR_TEXT_2,
    "ytick.color": COLOR_TEXT_2,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "figure.max_open_warning": 0,
})


def color_map(samples: list[str]) -> dict[str, str]:
    """El color sigue a la muestra (orden fijo), nunca a su posición en el ranking."""
    return {s: SERIES_COLORS[i % len(SERIES_COLORS)] for i, s in enumerate(samples)}


def _fig(w: float = 7, h: float = 4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_axisbelow(True)
    return fig, ax


def means_with_letters(means: pd.Series, errors: pd.Series | None, letters: dict[str, str] | None,
                       title: str, ylabel: str, ylim: tuple[float, float] | None = None,
                       ref: float | None = None, ref_label: str = ""):
    fig, ax = _fig()
    order = list(means.index)
    cmap = color_map(order)
    x = np.arange(len(order))
    ax.bar(x, means.values, width=0.6, color=[cmap[s] for s in order], edgecolor="white", linewidth=2)
    if errors is not None:
        ax.errorbar(x, means.values, yerr=errors.reindex(order).values, fmt="none",
                    ecolor=COLOR_TEXT_2, elinewidth=1.2, capsize=4)
    top = means.values + (errors.reindex(order).values if errors is not None else 0)
    for xi, s, y, t in zip(x, order, means.values, top):
        label = f"{y:.2f}"
        if letters:
            label += f"  {letters.get(s, '')}"
        ax.text(xi, t + (ylim[1] - ylim[0] if ylim else max(top)) * 0.02, label,
                ha="center", va="bottom", color=COLOR_TEXT, fontsize=10, fontweight="bold")
    if ref is not None:
        ax.axhline(ref, color=COLOR_BAD, lw=1.4, ls="--")
        ax.text(len(order) - 0.5, ref, f" {ref_label}", color=COLOR_BAD, va="bottom", ha="right", fontsize=9)
    ax.set_xticks(x, order)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim[0], ylim[1] * 1.12)
    ax.set_title(title, loc="left")
    if letters:
        fig.text(0.01, -0.02, "Muestras con la misma letra no difieren significativamente.",
                 color=COLOR_TEXT_2, fontsize=8)
    return fig


def boxplot(data: pd.DataFrame, order: list[str], title: str, ylabel: str):
    fig, ax = _fig()
    cmap = color_map(order)
    groups = [data.loc[data["muestra"] == s, "valor"].dropna().values for s in order]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5, medianprops={"color": COLOR_TEXT, "lw": 1.5})
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(cmap[s])
        patch.set_alpha(0.35)
        patch.set_edgecolor(cmap[s])
    for i, (g, s) in enumerate(zip(groups, order), start=1):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(g))
        ax.scatter(np.full(len(g), i) + jitter, g, s=14, color=cmap[s], alpha=0.8, zorder=3)
    ax.set_xticks(range(1, len(order) + 1), order)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    return fig


def score_distribution(data: pd.DataFrame, order: list[str], scale: tuple[float, float], title: str):
    lo, hi = int(scale[0]), int(scale[1])
    if hi - lo > 15:
        return None
    levels = np.arange(lo, hi + 1)
    fig, ax = _fig(7.5, 4)
    cmap = color_map(order)
    width = 0.8 / len(order)
    for i, s in enumerate(order):
        v = data.loc[data["muestra"] == s, "valor"].dropna().round()
        pct = v.value_counts(normalize=True).reindex(levels, fill_value=0) * 100
        ax.bar(levels + (i - (len(order) - 1) / 2) * width, pct.values, width=width * 0.92,
               color=cmap[s], label=s)
    ax.set_xticks(levels)
    ax.set_xlabel("Calificación")
    ax.set_ylabel("% de jueces")
    ax.legend(ncol=min(len(order), 6), loc="upper left")
    ax.set_title(title, loc="left")
    return fig


def acceptability_bars(ia: pd.Series, threshold: float, title: str):
    fig, ax = _fig()
    order = list(ia.index)
    colors = [COLOR_GOOD if v >= threshold else COLOR_BAD for v in ia.values]
    ax.barh(order, ia.values, color=colors, height=0.55, edgecolor="white", linewidth=2)
    ax.axvline(threshold, color=COLOR_TEXT_2, ls="--", lw=1.2)
    ax.text(threshold, len(order) - 0.45, f" umbral {threshold:g}%", color=COLOR_TEXT_2, fontsize=9)
    for y, v in enumerate(ia.values):
        ax.text(v + 1, y, f"{v:.1f}%  {'✓ aceptada' if v >= threshold else '✗ no alcanza'}",
                va="center", color=COLOR_TEXT, fontsize=10)
    ax.set_xlim(0, 118)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Índice de aceptabilidad (%)")
    ax.set_title(title, loc="left")
    return fig


def binomial_bars(successes: int, n: int, min_required: int | None, expected: float, title: str,
                  label_ok: str = "Aciertos"):
    fig, ax = _fig(7, 3.2)
    sig = min_required is not None and successes >= min_required
    ax.barh([0], [n], color=COLOR_GRID, height=0.5)
    ax.barh([0], [successes], color=COLOR_GOOD if sig else COLOR_PRIMARY, height=0.5)
    ax.text(successes, 0, f"  {successes} {label_ok.lower()}", va="center", ha="left",
            color=COLOR_TEXT, fontweight="bold")
    ax.axvline(expected, color=COLOR_MUTED, ls=":", lw=1.5)
    ax.text(expected, 0.38, f"azar: {expected:.1f}", ha="center", color=COLOR_TEXT_2, fontsize=9)
    if min_required is not None:
        ax.axvline(min_required, color=COLOR_BAD, ls="--", lw=1.5)
        ax.text(min_required, -0.42, f"mínimo tabla: {min_required}", ha="center", color=COLOR_BAD, fontsize=9)
    ax.set_xlim(0, n * 1.05)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Número de jueces (n = {n})")
    ax.set_title(title, loc="left")
    return fig


def contingency_heatmap(table: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    arr = table.values.astype(float)
    ax.imshow(arr, cmap="Blues", vmin=0, vmax=arr.max() * 1.3)
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=16, fontweight="bold", color=COLOR_TEXT)
    ax.set_xticks(range(arr.shape[1]), [f"Respondió «{c}»" for c in table.columns])
    ax.set_yticks(range(arr.shape[0]), [f"Se presentó «{r}»" for r in table.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left")
    return fig


def rank_sums(R: pd.Series, letters: dict[str, str], title: str, xlabel: str):
    fig, ax = _fig()
    order = list(R.index)
    cmap = color_map(order)
    ax.bar(order, R.values, color=[cmap[s] for s in order], width=0.6, edgecolor="white", linewidth=2)
    for i, (s, v) in enumerate(R.items()):
        ax.text(i, v * 1.01, f"{v:g}  {letters.get(s, '')}", ha="center", va="bottom",
                fontweight="bold", color=COLOR_TEXT)
    ax.set_ylabel("Suma de rangos")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, R.max() * 1.18)
    ax.set_title(title, loc="left")
    return fig


def proportion_bars(props: pd.Series, title: str, ylabel: str = "% de jueces", ref: float | None = None,
                    ref_label: str = ""):
    fig, ax = _fig()
    order = list(props.index)
    cmap = color_map(order)
    ax.bar(order, props.values, color=[cmap[s] for s in order], width=0.6, edgecolor="white", linewidth=2)
    for i, v in enumerate(props.values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold", color=COLOR_TEXT)
    if ref is not None:
        ax.axhline(ref, color=COLOR_MUTED, ls=":", lw=1.5)
        ax.text(len(order) - 0.5, ref, f" {ref_label}", color=COLOR_TEXT_2, va="bottom", ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    return fig


def radar(means: pd.DataFrame, title: str, scale: tuple[float, float] | None = None):
    """Perfil sensorial (araña): filas = muestras, columnas = atributos."""
    attrs = list(means.columns)
    if len(attrs) < 3:
        return None
    angles = np.linspace(0, 2 * np.pi, len(attrs), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"polar": True})
    cmap = color_map(list(means.index))
    for s, row in means.iterrows():
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, color=cmap[s], lw=2, label=s)
        ax.fill(angles, vals, color=cmap[s], alpha=0.08)
    ax.set_xticks(angles[:-1], attrs, color=COLOR_TEXT)
    if scale:
        ax.set_ylim(scale[0], scale[1])
    ax.tick_params(axis="y", colors=COLOR_TEXT_2, labelsize=8)
    ax.grid(color=COLOR_GRID)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    ax.set_title(title, loc="left", pad=24)
    return fig


def pca_biplot(scores: pd.DataFrame, loadings: pd.DataFrame, explained: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0, color=COLOR_GRID, lw=1)
    ax.axvline(0, color=COLOR_GRID, lw=1)
    ax.grid(False)
    sc = scores.iloc[:, :2]
    if sc.shape[1] < 2:
        sc = sc.assign(CP2=0.0)
    ld = loadings.iloc[:, :2]
    if ld.shape[1] < 2:
        ld = ld.assign(CP2=0.0)
    factor = np.abs(sc.values).max() / max(np.abs(ld.values).max(), 1e-9) * 0.9
    for attr, (x, y) in ld.iterrows():
        ax.annotate("", xy=(x * factor, y * factor), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": COLOR_MUTED, "lw": 1.3})
        ax.text(x * factor * 1.08, y * factor * 1.08, attr, color=COLOR_TEXT_2, fontsize=9, ha="center")
    cmap = color_map(list(sc.index))
    for s, (x, y) in sc.iterrows():
        ax.scatter(x, y, s=90, color=cmap[s], edgecolor="white", linewidth=2, zorder=3)
        ax.text(x, y, f"  {s}", fontweight="bold", color=COLOR_TEXT, va="center")
    ev = list(explained) + [0, 0]
    ax.set_xlabel(f"CP1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"CP2 ({ev[1] * 100:.1f}%)")
    ax.set_title(title, loc="left")
    return fig


def close_all(figs) -> None:
    for f in figs:
        if f is not None:
            plt.close(f)
