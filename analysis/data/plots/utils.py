from __future__ import annotations
from typing import Optional, Tuple
import torch
import anndata as ad

from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sc_reconstruction.utils.model_loader import create_model_from_cfg

try:
    from hydra import initialize_config_dir  # hydra>=1.2 typically
except Exception:
    from hydra.experimental import initialize_config_dir  # hydra<=1.1


def load_model_and_reconstruct(
    adata: ad.AnnData,
    *,
    config_dir: str,
    config_name: str = "base_pretrained_emb",
    model_key: str,
    eval_cfg_name: Optional[str] = None,
    ckpt_path: str,
    device: Optional[str] = None,
    clear_hydra: bool = True,
    predict_kwargs: Optional[dict] = None,
) -> Tuple[ad.AnnData, object]:
    """
    Loads model via Hydra config and runs model.predict(adata.X).
    Compatible with Hydra versions that do/don't support `version_base`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if clear_hydra and GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    if eval_cfg_name is None:
        eval_cfg_name = model_key

    predict_kwargs = predict_kwargs or {}

    try:
        ctx = initialize_config_dir(version_base=None, config_dir=config_dir)
    except TypeError:
        # older Hydra
        ctx = initialize_config_dir(config_dir=config_dir)

    with ctx:
        cfg = compose(
            config_name=config_name,
            overrides=[f"model=eval/{eval_cfg_name}"],
        )

    cfg.model.load.path = ckpt_path

    print(f"[{model_key}] device={device}")

    if cfg.get("dynamic_model_loading", False):
        model, checkpoint_path = create_model_from_cfg(cfg, device)
        print(f"[{model_key}] loaded (dynamic) from {checkpoint_path}")
    else:
        model = instantiate(cfg.model.model_args)
        model.load(cfg.model.load.path, map_location=device)
        print(f"[{model_key}] loaded (direct) from {cfg.model.load.path}")

    recon = model.predict(adata.X, **predict_kwargs)


    recon_adata = ad.AnnData(
        X=recon,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    recon_adata.obs["source"] = model_key
    return recon_adata, model

# ============================================================
# LogFC overlay helpers (ported from _fig2.ipynb cells 39/47/48)
# ============================================================


import os
import numpy as np
import pandas as pd
import zarr
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ranksums


# -----------------------------
# Utilities: dense + logFC + BH + DEGs
# -----------------------------
def _to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)

def _log2fc_means(X_treat, X_ctrl, eps=1.0):
    mu_t = np.asarray(X_treat).mean(axis=0)
    mu_c = np.asarray(X_ctrl).mean(axis=0)
    return np.log2((mu_t + eps) / (mu_c + eps))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

def wilcoxon_deg_mask(
    true_treat: ad.AnnData,
    true_ctrl: ad.AnnData,
    gene_names,
    *,
    alpha: float = 0.05,
    min_abs_logfc: float = 0.0,
    eps: float = 1.0,
    max_genes: int | None = None,
):
    gene_names = np.asarray(gene_names)
    Xt = _to_dense(true_treat.X)
    Xc = _to_dense(true_ctrl.X)

    if Xt.shape[1] != len(gene_names) or Xc.shape[1] != len(gene_names):
        raise ValueError("Gene dimension mismatch between X and gene_names.")

    true_logfc = _log2fc_means(Xt, Xc, eps=eps)

    pvals = np.empty(Xt.shape[1], dtype=float)
    for j in range(Xt.shape[1]):
        pvals[j] = ranksums(Xt[:, j], Xc[:, j]).pvalue

    qvals = bh_fdr(pvals)
    deg = (qvals <= alpha) & (np.abs(true_logfc) >= min_abs_logfc)

    df_stats = pd.DataFrame({
        "gene": gene_names,
        "pval": pvals,
        "qval": qvals,
        "true_logFC": true_logfc,
    }).sort_values(["qval", "pval"], ascending=True)

    if max_genes is not None and deg.sum() > max_genes:
        keep_idx = df_stats.index[:max_genes].to_numpy()
        deg2 = np.zeros_like(deg, dtype=bool)
        deg2[keep_idx] = True
        deg = deg2

    print(f"Found {int(deg.sum())} DEGs (FDR≤{alpha}, |logFC|≥{min_abs_logfc})")
    return deg, df_stats


def plot_logfc_overlay_models(
    *,
    true_treat: ad.AnnData,
    true_ctrl: ad.AnnData,
    recon_pairs: dict,            # {"AE": (pred_treat, pred_ctrl), ...}
    deg_mask: np.ndarray,
    gene_names=None,
    highlight_mask: np.ndarray | None = None,
    highlight_labels: list[str] | None = None,
    eps: float = 1.0,
    colors: dict | None = None,
    figsize=(3.6, 3.4),
    xlim=None,
    n_bins: int = 25,
    min_bin_n: int = 15,
    point_size: float = 4.0,
    point_alpha: float = 0.12,
    trend_lw: float = 1.8,
    title: str = "",
    out_path: str | None = None,
    highlight_size: float = 20.0,
    highlight_alpha: float = 1.0,
    highlight_edge: str = "black",          # default fallback
    highlight_lw: float = 0.4,
    highlight_edgecolors: list[str] | None = None,   # <-- NEW (per-gene stroke colors)
    annotate_highlights: bool = True,
    annotate_kwargs: dict | None = None,
):
    Xt = _to_dense(true_treat.X)
    Xc = _to_dense(true_ctrl.X)
    true_logfc_all = _log2fc_means(Xt, Xc, eps=eps)

    x_all = true_logfc_all
    x = x_all[deg_mask]

    if xlim is None:
        lim = float(np.nanmax(np.abs(x)) + 0.05) if len(x) else 1.0
        xlim = (-lim, lim)
    ylim = xlim

    if colors is None:
        colors = {"AE": "#2171b5", "scVI": "#7a0177", "PCA": "#ffbf00"}

    if annotate_kwargs is None:
        annotate_kwargs = dict(fontsize=7, xytext=(3, 3), textcoords="offset points")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]],
            linestyle="--", linewidth=1.0, alpha=0.5, color="grey")

    # ---- compute highlight indices
    if highlight_mask is not None:
        highlight_mask = np.asarray(highlight_mask, dtype=bool)
        if highlight_mask.shape != deg_mask.shape:
            raise ValueError("highlight_mask must have same shape as deg_mask (both in gene space).")

        idx_all = np.where(deg_mask & highlight_mask)[0]  # gene indices in ALL genes space

        deg_all_idx = np.where(deg_mask)[0]
        pos_in_deg = {g: i for i, g in enumerate(deg_all_idx)}
        idx_deg = np.array([pos_in_deg[g] for g in idx_all if g in pos_in_deg], dtype=int)
    else:
        idx_all = np.array([], dtype=int)
        idx_deg = np.array([], dtype=int)

    # ---- determine highlight labels aligned to idx_all / idx_deg
    # We want labels in the SAME ORDER as idx_deg (which corresponds to idx_all).
    if idx_deg.size > 0:
        if highlight_labels is None:
            labs = [str(np.asarray(gene_names)[g]) for g in idx_all] if gene_names is not None else [str(g) for g in idx_all]
        else:
            # assume user gives labels corresponding to the highlighted genes among DEGs
            # If lengths mismatch, we truncate to min to avoid crashing
            labs = list(highlight_labels)[: len(idx_deg)]
    else:
        labs = []

    # ---- per-gene edgecolor list
    # cycles if shorter than number of highlighted genes
    if highlight_edgecolors is None:
        edgecols = [highlight_edge] * len(idx_deg)
    else:
        if len(highlight_edgecolors) == 0:
            edgecols = [highlight_edge] * len(idx_deg)
        else:
            edgecols = [highlight_edgecolors[i % len(highlight_edgecolors)] for i in range(len(idx_deg))]

    # Plot each model
    for model_name, (pred_treat, pred_ctrl) in recon_pairs.items():
        col = colors.get(model_name, "grey")

        Xpt = _to_dense(pred_treat.X)
        Xpc = _to_dense(pred_ctrl.X)
        pred_logfc_all = _log2fc_means(Xpt, Xpc, eps=eps)
        y = pred_logfc_all[deg_mask]

        # base points
        ax.scatter(x, y, s=point_size, alpha=point_alpha, color=col, linewidths=0, zorder=1)

        # highlight points (per-gene, so each can have its own stroke color)
        if idx_deg.size > 0:
            for j, ec in zip(idx_deg, edgecols):
                ax.scatter(
                    [x[j]], [y[j]],
                    s=highlight_size,
                    alpha=highlight_alpha,
                    color=col,
                    edgecolors=ec,
                    linewidths=highlight_lw,
                    zorder=10,
                )

        # trend
        bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
        bin_id = np.digitize(x, bins) - 1
        centers = 0.5 * (bins[:-1] + bins[1:])

        y_med = np.full(n_bins, np.nan)
        for b in range(n_bins):
            m = bin_id == b
            if np.sum(m) >= min_bin_n:
                y_med[b] = np.nanmedian(y[m])
        ax.plot(centers, y_med, color=col, linewidth=trend_lw, zorder=3)

    # annotate highlights once (using first model's y for placement)
    if annotate_highlights and idx_deg.size > 0 and gene_names is not None:
        first_model = next(iter(recon_pairs.keys()))
        pred_treat, pred_ctrl = recon_pairs[first_model]
        y_first = _log2fc_means(_to_dense(pred_treat.X), _to_dense(pred_ctrl.X), eps=eps)[deg_mask]

        for j, lab, ec in zip(idx_deg, labs, edgecols):
            ax.annotate(str(lab), (x[j], y_first[j]), color=ec, **annotate_kwargs)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("True logFC")
    ax.set_ylabel("Predicted logFC")
    ax.set_title(title, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.show()

def logfc_overlay_for_cellline_drug(
    *,
    zarr_base: str,
    cell_line: str,
    drug: str,
    dose: str,
    control: str,
    split_group: str | None = None,
    config_dir: str,
    model_paths: dict,
    models: list[str] = ("AE", "scVI", "PCA"),
    eps: float = 1.0,
    alpha: float = 0.05,
    min_abs_logfc: float = 0.2,
    max_genes: int | None = 20,
    colors: dict | None = None,
    point_size: float = 20,
    point_alpha: float = 1.0,
    title: str | None = None,
    out_path: str | None = None,
    highlight_genes: list[str] | None = None,   # <-- NEW
    annotate_highlights: bool = True,           # <-- NEW
    highlight_colors: list[str] | None = None,  # <-- NEW
):
    def _join(*parts):
        return os.path.join(*[p for p in parts if p is not None and str(p) != ""])

    treat_key = f"{cell_line}-{drug}-{dose}"
    ctrl_key  = f"{cell_line}-{control}-0"

    treat_path = _join(zarr_base, split_group, treat_key)
    ctrl_path  = _join(zarr_base, split_group, ctrl_key)

    ptb_z = zarr.open(treat_path, mode="r")
    ref_z = zarr.open(ctrl_path, mode="r")

    ptb_adata = ad.AnnData(X=ptb_z["X"][:])
    ref_adata = ad.AnnData(X=ref_z["X"][:])

    zroot = zarr.open(_join(zarr_base, split_group), mode="r")
    gene_names = np.asarray(zroot.attrs["var_names"])

    for a in (ptb_adata, ref_adata):
        a.var_names = gene_names

    # reconstructions
    recon_pairs = {}
    for m in models:
        if m not in model_paths:
            raise KeyError(f"Missing model_paths['{m}']")
        recon_ptb, _ = load_model_and_reconstruct(
            ptb_adata, config_dir=config_dir, model_key=m, eval_cfg_name=m, ckpt_path=model_paths[m]
        )
        recon_ref, _ = load_model_and_reconstruct(
            ref_adata, config_dir=config_dir, model_key=m, eval_cfg_name=m, ckpt_path=model_paths[m]
        )
        recon_ptb.var_names = gene_names
        recon_ref.var_names = gene_names
        recon_pairs[m] = (recon_ptb, recon_ref)

    # DEGs on TRUE data only (possibly capped by max_genes)
    deg_mask, deg_table = wilcoxon_deg_mask(
        ptb_adata, ref_adata, gene_names,
        alpha=alpha, min_abs_logfc=min_abs_logfc, eps=eps, max_genes=max_genes
    )

    # Build highlight_mask: only genes that are BOTH in highlight list AND in deg_mask survive
    highlight_mask = None
    highlight_used = []
    if highlight_genes:
        highlight_set = {g.upper() for g in highlight_genes}
        gene_upper = pd.Index(gene_names.astype(str)).str.upper()
        highlight_mask = gene_upper.isin(highlight_set)

        # restrict to plotted DEGs (after max_genes cap)
        idx_all = np.where(deg_mask & highlight_mask)[0]
        highlight_used = [str(gene_names[i]) for i in idx_all]

    if title is None:
        title = f"{cell_line} · {drug} perturbed"

    plot_logfc_overlay_models(
        true_treat=ptb_adata,
        true_ctrl=ref_adata,
        recon_pairs=recon_pairs,
        deg_mask=deg_mask,
        gene_names=gene_names,
        highlight_mask=highlight_mask,
        highlight_labels=highlight_used if highlight_used else None,
        eps=eps,
        colors=colors,
        point_size=point_size,
        point_alpha=point_alpha,
        title=title,
        out_path=out_path,
        annotate_highlights=annotate_highlights,
        highlight_edgecolors=highlight_colors,
    )

    return {
        "ptb_adata": ptb_adata,
        "ref_adata": ref_adata,
        "recon_pairs": recon_pairs,
        "deg_mask": deg_mask,
        "deg_table": deg_table,
        "gene_names": gene_names,
        "highlight_used": highlight_used,  
        "treat_path": treat_path,
        "ctrl_path": ctrl_path,
    }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr

def _log2fc_from_mats(Xt, Xc, eps=1.0):
    mu_t = np.asarray(Xt).mean(axis=0)
    mu_c = np.asarray(Xc).mean(axis=0)
    return np.log2((mu_t + eps) / (mu_c + eps))

def _parse_key(key: str):
    # CVCL_0320-Dabrafenib-0_5  (robust to extra '-')
    parts = key.split("-")
    cell = parts[0] if len(parts) > 0 else "NA"
    drug = parts[1] if len(parts) > 1 else "NA"
    dose = "-".join(parts[2:]) if len(parts) > 2 else "NA"
    return cell, drug, dose

def build_gene_logfc_df_one_cellline(
    *,
    zarr_path: str,
    gene_names,
    gene: str,
    treated_keys: list[str],
    control_key: str,                 # e.g. "CVCL_0320-DMSO_TF-0"
    recon_predict_fns: dict,          # {"AE": fn, "scVI": fn, "PCA": fn}
    eps: float = 1.0,
    max_cells: int | None = None,     # optional subsample for speed
    seed: int = 0,
):
    """
    Returns df with columns:
      key, drug, dose, true_logFC, pred_logFC_<model>
    Each row = one treated key (drug/dose) compared to the SAME control_key.
    """
    gene_names = np.asarray(gene_names)
    if gene not in set(gene_names):
        raise ValueError(f"Gene {gene} not in gene_names")
    gi = int(np.where(gene_names == gene)[0][0])

    root = zarr.open(zarr_path, mode="r")
    if control_key not in root:
        raise KeyError(f"Control key not found in zarr: {control_key}")

    Xc_full = np.asarray(root[control_key]["X"][:])
    rng = np.random.default_rng(seed)

    rows = []
    for k in treated_keys:
        if k not in root:
            continue
        Xt_full = np.asarray(root[k]["X"][:])

        # optional subsample cells for speed (keeps gene-wise means stable enough)
        if max_cells is not None:
            if Xt_full.shape[0] > max_cells:
                idx = rng.choice(Xt_full.shape[0], size=max_cells, replace=False)
                Xt = Xt_full[idx]
            else:
                Xt = Xt_full
            if Xc_full.shape[0] > max_cells:
                idx = rng.choice(Xc_full.shape[0], size=max_cells, replace=False)
                Xc = Xc_full[idx]
            else:
                Xc = Xc_full
        else:
            Xt, Xc = Xt_full, Xc_full

        true_logfc = float(_log2fc_from_mats(Xt[:, [gi]], Xc[:, [gi]], eps=eps)[0])

        cell, drug, dose = _parse_key(k)
        row = dict(key=k, cell_line=cell, drug=drug, dose=dose, true_logFC=true_logfc)

        # model predictions (gene means after reconstruction)
        for model_name, fn in recon_predict_fns.items():
            Xthat = fn(Xt)  # must return (cells, genes)
            Xchat = fn(Xc)
            pred_logfc = float(_log2fc_from_mats(Xthat[:, [gi]], Xchat[:, [gi]], eps=eps)[0])
            row[f"pred_logFC_{model_name}"] = pred_logfc

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def plot_gene_true_vs_pred_across_drugs(
    df: pd.DataFrame,
    *,
    gene: str,
    model: str,                        # "AE"/"scVI"/"PCA"
    highlight_drugs: list[str] | None = None,
    figsize=(2.4, 2.4),
    out_path: str | None = None,
    title: str | None = None,
):
    ycol = f"pred_logFC_{model}"
    d = df.dropna(subset=["true_logFC", ycol]).copy()
    if d.empty:
        print("[WARN] nothing to plot")
        return

    x = d["true_logFC"].to_numpy()
    y = d[ycol].to_numpy()

    lim = np.nanmax(np.abs(np.r_[x, y]))
    lim = max(lim, 0.5)

    fig, ax = plt.subplots(figsize=figsize)

    # background points
    ax.scatter(x, y, s=18, color="0.6", alpha=0.75, linewidths=0, zorder=1)

    # highlight subset of drugs (e.g. Dabrafenib)
    if highlight_drugs:
        dh = d[d["drug"].isin(highlight_drugs)]
        if len(dh):
            ax.scatter(dh["true_logFC"], dh[ycol], s=24, alpha=0.95, linewidths=0, zorder=2)

    # reference lines
    ax.axhline(0, color="0.85", lw=1)
    ax.axvline(0, color="0.85", lw=1)
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color="0.6")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("True logFC")
    ax.set_ylabel("Predicted logFC")
    ax.set_title(title if title is not None else f"{gene} ({model})", fontsize=10, pad=2)

    ax.grid(True, linestyle="--", alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_gene_true_vs_pred_multi_models(
    df: pd.DataFrame,
    *,
    gene: str,
    models: list[str] = ("AE", "scVI", "PCA"),
    model_colors: dict | None = None,
    model_markers: dict | None = None,
    highlight_drugs: list[str] | None = None,
    highlight_alpha: float = 0.95,
    base_alpha: float = 0.50,
    s_base: float = 16,
    s_highlight: float = 22,
    figsize=(2.6, 2.6),
    out_path: str | None = None,
    title: str | None = None,
    show_legend: bool = True,
    legend_loc: str = "upper left",
    legend_bbox_to_anchor=(1.02, 1.0),
    lim_pad: float = 0.08,
):
    """
    One plot: x=true_logFC; y=pred_logFC_<model> for several models (colored).
    Each row in df corresponds to one drug/dose comparison.
    """
    if model_colors is None:
        model_colors = {
            "AE":   "#2171b5",
            "scVI": "#7a0177",
            "PCA":  "#ffbf00",
        }
    if model_markers is None:
        model_markers = {m: "o" for m in models}

    d = df.copy()
    if "true_logFC" not in d.columns:
        raise ValueError("df must contain column 'true_logFC'")

    # keep only rows that have at least one model's pred_logFC
    pred_cols = [f"pred_logFC_{m}" for m in models if f"pred_logFC_{m}" in d.columns]
    if len(pred_cols) == 0:
        raise ValueError(f"df has no pred columns for models={models}")

    d = d.dropna(subset=["true_logFC"]).copy()
    if d.empty:
        print("[WARN] nothing to plot (no true_logFC)")
        return

    # compute axis limits from all included models
    vals = [d["true_logFC"].to_numpy()]
    for m in models:
        c = f"pred_logFC_{m}"
        if c in d.columns:
            vals.append(d[c].to_numpy())
    vals = np.concatenate([v[np.isfinite(v)] for v in vals if v is not None])
    if vals.size == 0:
        print("[WARN] nothing to plot (all NaN)")
        return

    lim = float(np.nanmax(np.abs(vals)))
    lim = lim * (1.0 + lim_pad)

    fig, ax = plt.subplots(figsize=figsize)

    # reference lines
    ax.axhline(0, color="0.85", lw=1)
    ax.axvline(0, color="0.85", lw=1)
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color="0.6", zorder=0)

    # highlight mask
    if highlight_drugs is not None and "drug" in d.columns:
        hi_mask = d["drug"].isin(highlight_drugs).to_numpy()
    else:
        hi_mask = np.zeros(len(d), dtype=bool)

    x = d["true_logFC"].to_numpy()

    # plot each model
    for m in models:
        ycol = f"pred_logFC_{m}"
        if ycol not in d.columns:
            continue
        y = d[ycol].to_numpy()

        col = model_colors.get(m, "0.2")
        mk  = model_markers.get(m, "o")

        # base points (non-highlight)
        base = ~hi_mask
        ax.scatter(
            x[base], y[base],
            s=s_base,
            color=col,
            alpha=base_alpha,
            linewidths=0,
            marker=mk,
            label=m if show_legend else None,
            zorder=2,
        )

        # highlighted points
        if hi_mask.any():
            ax.scatter(
                x[hi_mask], y[hi_mask],
                s=s_highlight,
                color=col,
                alpha=highlight_alpha,
                linewidths=0,
                marker=mk,
                zorder=3,
            )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("True logFC")
    ax.set_ylabel("Predicted logFC")

    ax.set_title(title if title is not None else gene, fontsize=10, pad=2)

    ax.grid(True, linestyle="--", alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    if show_legend:
        ax.legend(
            frameon=False,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            borderaxespad=0.0,
            handletextpad=0.4,
        )

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.show()

import os

def list_treated_keys_from_fs(zarr_dir: str, cell_line: str, control_token: str = "DMSO_TF"):
    """
    zarr_dir: .../comb_w_obs_3000wCCM.zarr
    returns list of group names like 'CVCL_0320-Dabrafenib-0_5'
    """
    treated = []
    prefix = f"{cell_line}-"
    with os.scandir(zarr_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith(prefix) and (control_token not in name):
                treated.append(name)
    treated.sort()
    return treated
