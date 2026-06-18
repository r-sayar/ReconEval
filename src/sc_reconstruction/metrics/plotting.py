"""Plotting helpers for the metrics API.

The public entry point is :func:`funky_heatmap`, a reproduction of the
funky-map summary used in the paper (Fig 3, decoder × metric). The plot
has one row per method (sorted by overall rank-percentile, best at top)
and a column layout of:

    Overall | Stat. | <statistical metric columns> | Bio. | <biological metric columns>

The "Overall" / "Stat." / "Bio." columns are drawn as rounded rank bars
whose width scales with the family rank-percentile and whose fill colour
darkens with the score. The metric columns are drawn as colour-coded
circles whose area and colour both encode the within-column normalised
score (min-max normalisation per column; the orientation respects
``HIGHER_IS_BETTER``). The raw value is annotated inside each cell.

``matplotlib`` is imported lazily so the metrics core stays plot-free
for users who only score and aggregate.
"""

from __future__ import annotations

import colorsys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .api import HIGHER_IS_BETTER, aggregate_rank_percentile


__all__ = ["funky_heatmap"]


# Default grouping mirrors the paper's Fig 2 / Fig 3 summary panels.
_DEFAULT_FAMILIES: dict[str, tuple[str, ...]] = {
    "statistical": ("r2", "mse", "energy_distance", "mmd_rbf"),
    "biological": (
        "cellcycle_proportion_same_phase",
        "coexpression",
        "pathway",
        "deg_dice_at_50",
        "deg_dice_at_100",
        "deg_logfc_pearson",
        "deg_logfc_spearman",
        "cytokine",
    ),
    "perturbational": ("knn_purity",),
}

# Colours used for each rank bar / metric column group (matches fig3_clean).
_RANK_COLORS: dict[str, str] = {
    "Overall": "#7A3A61",
    "Statistical": "#233C66",
    "Biological": "#2D6B66",
    "Perturbational": "#6B5B2D",
}


def _hex_to_hls(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h: float, ll: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, max(0, min(1, ll)), max(0, min(1, s)))
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _family_cols(scores: pd.DataFrame, names: Sequence[str]) -> list[str]:
    return [c for c in scores.columns if c in set(names)]


def funky_heatmap(
    scores: pd.DataFrame,
    *,
    higher_is_better: Mapping[str, bool] | None = None,
    families: Mapping[str, Sequence[str]] | None = None,
    rank_colors: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ax=None,
):
    """Render the Fig-3-style funky map from a method × metric score table.

    Parameters
    ----------
    scores
        DataFrame indexed by method (rows) and metric name (columns).
    higher_is_better
        Per-metric direction; defaults to
        :data:`sc_reconstruction.metrics.HIGHER_IS_BETTER`. Missing keys
        default to ``True`` (higher = better).
    families
        Mapping ``{family_name: [metric_names]}`` deciding which columns
        belong to which family. Family names are case-insensitive; the
        canonical labels used in the plot are ``"Statistical"``,
        ``"Biological"``, ``"Perturbational"``. Defaults to the paper's
        grouping. Pass ``{}`` to disable family columns.
    rank_colors
        Optional ``{family_label: hex_color}`` overriding the default
        palette (Overall = purple, Stat. = blue, Bio. = teal).
    figsize
        Matplotlib figsize. Defaults to ``(7, 0.4 * n_methods + 1.5)``.
    title
        Optional title.
    ax
        Existing matplotlib axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn onto.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    if higher_is_better is None:
        higher_is_better = HIGHER_IS_BETTER
    if families is None:
        families = _DEFAULT_FAMILIES
    if rank_colors is None:
        rank_colors = _RANK_COLORS

    # Canonical family labels (capitalised) drive the column layout.
    fam_labels: dict[str, str] = {}
    for fam in families:
        fam_labels[fam] = fam.capitalize()

    # Per-column rank-percentile (orientation from `higher_is_better`).
    rp = aggregate_rank_percentile(scores, higher_is_better=higher_is_better)

    fam_cols: dict[str, list[str]] = {
        fam: _family_cols(scores, names) for fam, names in families.items()
    }
    fam_rank: dict[str, pd.Series] = {
        fam: (rp[cols].mean(axis=1) if cols else pd.Series(np.nan, index=rp.index))
        for fam, cols in fam_cols.items()
    }
    overall_rank = rp.mean(axis=1)

    # Sort methods best-first by overall rank.
    model_order = list(
        overall_rank.sort_values(ascending=False, na_position="last").index
    )

    # Column entries: (display_name, ctype, key, family_label).
    col_entries: list[tuple[str, str, str, str]] = []
    col_entries.append(("Overall", "rank_bar", "_overall_rank", "Overall"))
    # Per family: one summary rank bar followed by its metric circles.
    for fam, cols in fam_cols.items():
        if not cols:
            continue
        label = fam_labels[fam]
        col_entries.append((f"{label[:4]}.", "rank_bar", f"_{fam}_rank", label))
        for m in cols:
            col_entries.append((m, "metric_circle", m, label))

    # Column x-positions (small gaps between groups).
    col_x: list[float] = []
    x = 0.0
    spacing = 0.55
    gap_small = 0.08
    gap_large = 0.25
    for i, (name, ctype, key, group) in enumerate(col_entries):
        if i == 1:
            x += 0.15
        elif i == 2:
            x += gap_small
        elif ctype == "rank_bar" and i > 0 and col_entries[i - 1][3] != group:
            x += gap_large
        elif i > 0 and col_entries[i - 1][1] == "rank_bar" and col_entries[i - 1][3] == group:
            x += gap_small
        col_x.append(x)
        x += spacing
    col_x_arr = np.array(col_x)

    # Pre-compute the rank/value matrix the renderer needs.
    rp_extra = pd.DataFrame(
        {
            "_overall_rank": overall_rank,
            **{f"_{fam}_rank": fam_rank[fam] for fam in fam_cols},
        }
    )

    n_models = len(model_order)
    if ax is None:
        if figsize is None:
            figsize = (max(7.0, 0.55 * len(col_entries) + 2.0), 0.4 * n_models + 1.5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Striped row backgrounds.
    for i in range(n_models):
        if i % 2 == 0:
            ax.add_patch(
                Rectangle(
                    (col_x_arr[0] - 0.40, i - 0.45),
                    col_x_arr[-1] - col_x_arr[0] + 0.80,
                    0.9,
                    fc="#f5f5f5",
                    ec="none",
                    zorder=0,
                )
            )

    for i, model in enumerate(model_order):
        y = i
        for j, (name, ctype, key, group) in enumerate(col_entries):
            if ctype == "rank_bar":
                val = float(rp_extra.loc[model, key]) if model in rp_extra.index else float("nan")
                if np.isnan(val):
                    continue
                cell_left = col_x_arr[j] - 0.26
                bar_w = val * 0.52
                base_c = rank_colors.get(group, "#444444")
                hh, ll, ss = _hex_to_hls(base_c)
                bar_color = _hls_to_hex(hh, 0.35 + 0.5 * (1 - val), ss)
                ax.add_patch(
                    FancyBboxPatch(
                        (cell_left, y - 0.32),
                        bar_w,
                        0.64,
                        boxstyle="round,pad=0.03",
                        fc=bar_color,
                        ec="none",
                        zorder=2,
                    )
                )
                ax.text(
                    cell_left,
                    y,
                    f"{val:.2f}",
                    ha="left",
                    va="center",
                    fontsize=7,
                    color="black",
                    zorder=3,
                )

            elif ctype == "metric_circle":
                val = float(scores.loc[model, key]) if (model in scores.index and key in scores.columns) else float("nan")
                if np.isnan(val):
                    continue
                col_vals = scores[key].astype(float)
                hib = higher_is_better.get(key, True)
                vmin = float(col_vals.min())
                vmax = float(col_vals.max())
                if vmax > vmin:
                    raw = (val - vmin) / (vmax - vmin)
                    score = raw if hib else 1.0 - raw
                else:
                    score = 0.5
                score = max(0.0, min(1.0, score))

                base_c = rank_colors.get(group, "#444444")
                hh, ll, ss = _hex_to_hls(base_c)
                circle_color = _hls_to_hex(hh, 0.85 - 0.55 * score, ss)

                size = score * 550 + 180
                ax.scatter(
                    col_x_arr[j],
                    y,
                    s=size,
                    c=circle_color,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=3,
                )
                ax.text(
                    col_x_arr[j],
                    y,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                    zorder=4,
                )

    # Axis cosmetics.
    col_names = [e[0] for e in col_entries]
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_order, fontsize=11)
    ax.set_xticks(col_x_arr)
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=9)
    ax.set_xlim(col_x_arr[0] - 0.45, col_x_arr[-1] + 0.45)
    ax.set_ylim(-0.5, n_models - 0.5)
    ax.invert_yaxis()

    # Vertical dashed separators between family groups.
    prev_group = col_entries[0][3]
    for j in range(1, len(col_entries)):
        g = col_entries[j][3]
        if g != prev_group:
            x_sep = 0.5 * (col_x_arr[j - 1] + col_x_arr[j])
            ax.axvline(x_sep, color="0.82", ls="--", lw=0.5, alpha=0.6)
        prev_group = g

    # Group headers above the rank-bar columns.
    header_y = -0.85
    # Overall column header
    ax.text(
        col_x_arr[0],
        header_y,
        "Rank",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=rank_colors.get("Overall", "#7A3A61"),
    )
    # Family headers span their (summary bar + metric circles) columns
    for fam, cols in fam_cols.items():
        if not cols:
            continue
        label = fam_labels[fam]
        idx = [j for j, e in enumerate(col_entries) if e[3] == label]
        if idx:
            ax.text(
                float(np.mean(col_x_arr[idx])),
                header_y,
                label,
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=rank_colors.get(label, "#444444"),
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.tick_params(left=False, bottom=False)

    if title:
        ax.set_title(title, pad=25)

    return ax
