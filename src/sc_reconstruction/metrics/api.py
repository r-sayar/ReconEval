"""Lightweight metrics API for ReconEval.

One call per metric on a `(true, predicted)` AnnData pair, plus a single
`compute_all_metrics` that runs everything for which the inputs are available.

The heavy `MetricsBatchEvaluator` family used by the paper's reproduction
pipeline lives in `sc_reconstruction.metrics.base_eval` and the per-family
modules; it is not re-exported here.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from . import utils as _u


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_dense(x):
    """Return a dense numpy array, regardless of whether `x` is sparse."""
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


def _check_pair(adata_true: AnnData, adata_pred: AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Extract dense matched `.X` arrays from a (true, predicted) pair."""
    if adata_true.n_vars != adata_pred.n_vars:
        raise ValueError(
            f"Gene-axis mismatch: true has {adata_true.n_vars} vars, "
            f"pred has {adata_pred.n_vars}. "
            "Subset to the same `.var_names` before calling the metrics."
        )
    return _as_dense(adata_true.X), _as_dense(adata_pred.X)


# ---------------------------------------------------------------------------
# Per-metric wrappers (lightweight, no batch evaluator needed)
# ---------------------------------------------------------------------------

# ---- Statistical ----------------------------------------------------------

def metric_r2(adata_true: AnnData, adata_pred: AnnData) -> float:
    """R^2 between summed expression per gene across cells.

    Wraps :func:`sc_reconstruction.metrics.utils.gene_r_squared`.
    """
    X, Y = _check_pair(adata_true, adata_pred)
    return float(_u.gene_r_squared(X, Y))


def metric_mse(adata_true: AnnData, adata_pred: AnnData) -> float:
    """Mean squared error between matched cells (cell-wise)."""
    X, Y = _check_pair(adata_true, adata_pred)
    if X.shape != Y.shape:
        raise ValueError(
            f"MSE requires matched shapes; got {X.shape} vs {Y.shape}."
        )
    return float(np.mean((X - Y) ** 2))


def metric_energy_distance(adata_true: AnnData, adata_pred: AnnData) -> float:
    """Energy distance between empirical distributions of cells.

    Implements the multivariate energy distance of
    :cite:`szekely:13`.
    """
    X, Y = _check_pair(adata_true, adata_pred)
    return float(_u.energy_distance(X, Y))


# ---- Biological -----------------------------------------------------------

def metric_cellcycle(
    adata_true: AnnData,
    adata_pred: AnnData,
    *,
    s_genes: Sequence[str],
    g2m_genes: Sequence[str],
    min_cells: int = 20,
) -> dict[str, float]:
    """Cell-cycle phase agreement between true and predicted expression.

    Cell-cycle scoring follows :cite:`tirosh:16` (the S- and G2M-phase
    gene lists shipped with most single-cell pipelines).
    Wraps :meth:`CellCycleCalculator.cell_cycle_labeling_similarity`.

    Parameters
    ----------
    s_genes, g2m_genes
        Gene symbols (matching ``adata_*.var_names``) defining S and G2M
        phases. Pass the standard :cite:`tirosh:16` lists or a custom split.
    min_cells
        Minimum number of cells per gene for inclusion.
    """
    from ._cellcycle import CellCycleCalculator

    cc_genes = list(s_genes) + list(g2m_genes)
    return CellCycleCalculator.cell_cycle_labeling_similarity(
        adata_true=adata_true.copy(),
        adata_recon=adata_pred.copy(),
        min_cells=min_cells,
        cell_cycle_genes=cc_genes,
        s_genes=list(s_genes),
        g2m_genes=list(g2m_genes),
    )


def metric_coexpression(
    adata_true: AnnData,
    adata_pred: AnnData,
    *,
    geneset_dict: Mapping[str, Sequence[str]] | None = None,
    correlation_measure: str = "spearman",
    overlap_threshold: int = 5,
    min_cells: int = 20,
) -> float:
    """Mean co-expression similarity over MSigDB Hallmark gene sets.

    Uses the MSigDB Hallmark collection of :cite:`subramanian:05`,
    fetched via OmniPath :cite:`turei:21`.
    Wraps :meth:`CoexpressionCalculator.gene_set_coexpression`.

    Parameters
    ----------
    geneset_dict
        Mapping from gene-set name to list of gene symbols. If ``None``,
        the Hallmark collection :cite:`subramanian:05` is fetched from
        MSigDB via ``omnipath`` :cite:`turei:21` (requires the
        ``omnipath`` package + internet on first call).
    correlation_measure
        One of ``"pearson"`` or ``"spearman"``.
    """
    from ._coexpression import CoexpressionCalculator

    if geneset_dict is None:
        try:
            import omnipath as op
            anns = op.requests.Annotations.get(resources="MSigDB")
            geneset_dict = CoexpressionCalculator._get_msigdb_collection(anns)
        except Exception as e:
            warnings.warn(
                f"metric_coexpression: could not fetch MSigDB via omnipath ({e}); "
                "pass `geneset_dict` explicitly to skip the network call."
            )
            return float("nan")

    return float(
        CoexpressionCalculator.gene_set_coexpression(
            adata_true=adata_true,
            adata_recon=adata_pred,
            overlap_threshold=overlap_threshold,
            min_cells=min_cells,
            correlation_measure=correlation_measure,
            pipeline_output=True,
            geneset_dict=dict(geneset_dict),
        )
    )


def metric_pathway(
    adata_true: AnnData,
    adata_pred: AnnData,
    *,
    progeny_model: pd.DataFrame | None = None,
    pathway_dict: Mapping[str, Sequence[str]] | None = None,
    correlation_measure: str = "spearman",
    overlap_threshold: int = 5,
    min_cells: int = 30,
) -> float:
    """Pathway activity correlation (PROGENy ULM scores).

    PROGENy pathway weights come from :cite:`schubert:18`; the
    univariate-linear-model scoring is provided by ``decoupler``
    :cite:`badia:22`.
    Wraps :meth:`PathwayCalculator.pathway_score_similarity`.

    Parameters
    ----------
    progeny_model
        PROGENy weight DataFrame (from ``decoupler.op.progeny()``,
        :cite:`schubert:18,badia:22`). If ``None``, it is fetched lazily.
    pathway_dict
        Optional explicit pathway → gene list mapping. Falls back to
        PROGENy coverage.
    """
    from ._pathway import PathwayCalculator

    if progeny_model is None:
        try:
            import decoupler as dc
            # decoupler>=2.1 removed get_progeny() in favor of op.progeny()
            # (same rename load_progeny() below already follows) -- the old
            # name is gone entirely, not deprecated, so this raised
            # AttributeError on every call and silently degraded to NaN via
            # the except-and-warn below instead of ever actually computing.
            progeny_model = dc.op.progeny(organism="human", top=500)
        except Exception as e:
            warnings.warn(
                f"metric_pathway: could not fetch PROGENy via decoupler ({e}); "
                "pass `progeny_model` explicitly."
            )
            return float("nan")

    if pathway_dict is None:
        pathway_dict = (
            progeny_model.groupby("source")["target"].apply(list).to_dict()
        )

    return float(
        PathwayCalculator.pathway_score_similarity(
            adata_true=adata_true,
            adata_recon=adata_pred,
            min_cells=min_cells,
            overlap_threshold=overlap_threshold,
            correlation_measure=correlation_measure,
            pipeline_output=True,
            progeny_model=progeny_model,
            pathway_dict=dict(pathway_dict),
        )
    )


def metric_deg(
    adata_true: AnnData,
    adata_pred: AnnData,
    *,
    ref_true: AnnData,
    ref_pred: AnnData,
    method: str = "wilcoxon",
    dice_k: Sequence[int] = (100,),
    min_cells: int = 20,
    set_neg_to_zero: bool = False,
    fdr_threshold: float | None = 0.05,
) -> dict[str, float]:
    """Differential-expression recovery vs a reference (e.g. control) condition.

    The Wilcoxon and t-test calls are forwarded to scanpy
    :cite:`wolf:18`. The metric computes DEGs of (true vs ref_true) and
    (pred vs ref_pred), then reports overlap (Dice@k) and
    rank-correlation on the log-fold-changes.
    Wraps :meth:`DegCalculator.compute_deg`.

    Parameters
    ----------
    ref_true, ref_pred
        Reference (e.g. control) condition's true and predicted expression.
    method
        ``"wilcoxon"`` or ``"t-test"``. Forwarded to scanpy
        :cite:`wolf:18`.
    dice_k
        K values at which to compute Dice overlap of top-K DEGs.
    """
    from ._deg import DegCalculator

    X_true, X_pred = _check_pair(adata_true, adata_pred)
    R_true, R_pred = _check_pair(ref_true, ref_pred)

    return DegCalculator.compute_deg(
        x=X_true, x_hat=X_pred,
        refer=R_true, refer_hat=R_pred,
        method=method,
        min_cells=min_cells,
        set_neg_to_zero=set_neg_to_zero,
        dice_k=list(dice_k),
        compute_mean_diff=True,
        compute_topk_corr=True,
        fdr_threshold=fdr_threshold,
    )


def metric_cytokine(
    adata_true: AnnData,
    adata_pred: AnnData,
    *,
    cytokine_dict: Mapping[str, Sequence[str]],
    correlation: str = "spearman",
    min_genes: int = 5,
    ctrl_size: int = 50,
    n_bins: int = 25,
    reducer: str = "mean",
) -> float:
    """Cytokine activity similarity (Immune Dictionary :cite:`cui:24`).

    Cytokine signatures are taken from the Immune Dictionary
    :cite:`cui:24`. The underlying calculator returns both Pearson and
    Spearman correlations; this wrapper picks one (default Spearman) and
    returns the scalar.
    Wraps :meth:`CytokineCalculator.cytokine_activity_similarity`.

    Parameters
    ----------
    cytokine_dict
        Mapping cytokine -> list of genes (Immune Dictionary signature,
        :cite:`cui:24`). Build one with :func:`load_cytokine_dict_from_csv`.
    correlation
        ``"pearson"`` or ``"spearman"`` — which of the two reported
        correlations to return.
    """
    from ._cytokine import CytokineCalculator

    pr, sr = CytokineCalculator.cytokine_activity_similarity(
        adata_true=adata_true,
        adata_recon=adata_pred,
        cytokine2genes=dict(cytokine_dict),
        min_genes=min_genes,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        reducer=reducer,
    )
    return float(pr["average"] if correlation == "pearson" else sr["average"])


# ---------------------------------------------------------------------------
# Resource loaders — small convenience helpers so users don't have to wire up
# the file formats themselves.
# ---------------------------------------------------------------------------

def load_cell_cycle_genes(path: str | Path) -> tuple[list[str], list[str]]:
    """Load S-phase and G2M-phase gene lists from the Regev-lab text file.

    The two lists are those of :cite:`tirosh:16`. Format: one gene symbol
    per line, first 43 are S-phase, the rest are G2M. The file ships
    with the paper artefacts at
    ``analysis/data/frozen/regev_lab_cell_cycle_genes.txt``.
    """
    with open(path) as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes[:43], genes[43:]


def load_cytokine_dict_from_csv(
    path: str | Path,
    *,
    celltype: str | None = None,
    fdr_threshold: float | None = 0.05,
    min_genes: int = 5,
) -> dict[str, list[str]]:
    """Build a cytokine -> gene-list mapping from the Immune Dictionary CSV.

    Cytokine signatures are taken from the Immune Dictionary of
    :cite:`cui:24`. The CSV at
    ``analysis/data/frozen/cytokine_act_merged.csv`` has columns
    ``Celltype_Str, Cytokine_Str, Gene, FDR, Avg_log2FC`` (plus a few
    others).

    Parameters
    ----------
    celltype
        If given, restrict to one cell-type (e.g. ``"B_cell"``). If
        ``None``, cytokine signatures are pooled across all cell types.
    fdr_threshold
        Drop genes with ``FDR > fdr_threshold`` before grouping. Set to
        ``None`` to keep all genes.
    min_genes
        Drop cytokines with fewer than this many surviving genes.
    """
    df = pd.read_csv(path)
    if celltype is not None:
        df = df[df["Celltype_Str"] == celltype]
    if fdr_threshold is not None:
        df = df[df["FDR"] <= fdr_threshold]

    out: dict[str, list[str]] = {}
    for cyto, group in df.groupby("Cytokine_Str"):
        genes = sorted(set(group["Gene"].astype(str).str.upper()))
        if len(genes) >= min_genes:
            out[cyto] = genes
    return out


def load_progeny(organism: str = "human") -> pd.DataFrame:
    """Fetch the PROGENy :cite:`schubert:18` weight DataFrame via decoupler
    :cite:`badia:22`. Cached on first call by decoupler.
    """
    import decoupler as dc
    return dc.op.progeny(organism=organism)


# ---- Perturbational -------------------------------------------------------

def metric_knn_purity(
    adata_pred: AnnData,
    adata_pert_true: AnnData,
    adata_ctrl: AnnData,
    *,
    k: int = 20,
    use_rep: str | None = None,
) -> float:
    """KNN purity of predicted perturbation in a (true-perturbed, control) pool.

    Wraps :func:`sc_reconstruction.metrics.utils.knn_purity`.

    Parameters
    ----------
    adata_pred
        Predicted perturbed cells (the query).
    adata_pert_true
        Ground-truth perturbed cells (the positive pool).
    adata_ctrl
        Control cells (the negative pool).
    k
        Number of neighbors. Returns a float in ``[0, 1]``: 1.0 = all neighbors
        are true-perturbed, 0.5 = random baseline.
    use_rep
        If given, use ``.obsm[use_rep]`` instead of ``.X`` (recommended; KNN in
        a learned latent space is far more meaningful than in raw expression).
    """
    def _get(a):
        if use_rep is None:
            return _as_dense(a.X)
        if use_rep not in a.obsm:
            raise KeyError(f"{use_rep!r} not in .obsm of provided AnnData")
        return np.asarray(a.obsm[use_rep])

    return float(
        _u.knn_purity(
            X_query=_get(adata_pred),
            X_pert=_get(adata_pert_true),
            X_ctrl=_get(adata_ctrl),
            k=k,
        )
    )


# ---------------------------------------------------------------------------
# Category bundles
# ---------------------------------------------------------------------------

#: Mapping ``{metric_name: higher_is_better}`` consumed by
#: ``aggregate_rank_percentile``. ``True`` = larger metric value is
#: better; ``False`` = smaller is better.
HIGHER_IS_BETTER: dict[str, bool] = {
    # statistical
    "r2": True,
    "mse": False,
    "energy_distance": False,
    # biological
    "cellcycle_proportion_same_phase": True,
    "coexpression": True,
    "pathway": True,
    "deg_dice_at_100": True,
    "deg_logfc_spearman": True,
    "cytokine": True,
    # perturbational
    "knn_purity": True,
}


def compute_statistical_metrics(
    adata_true: AnnData,
    adata_pred: AnnData,
) -> dict[str, float]:
    """Run the statistical metrics on a (true, predicted) pair.

    Returns
    -------
    dict
        Keys: ``r2``, ``mse``, ``energy_distance``.
    """
    return {
        "r2": metric_r2(adata_true, adata_pred),
        "mse": metric_mse(adata_true, adata_pred),
        "energy_distance": metric_energy_distance(adata_true, adata_pred),
    }


def compute_biological_metrics(
    adata_true: AnnData,
    adata_pred: AnnData,
    *,
    s_genes: Sequence[str] | None = None,
    g2m_genes: Sequence[str] | None = None,
    geneset_dict: Mapping[str, Sequence[str]] | None = None,
    progeny_model: pd.DataFrame | None = None,
    pathway_dict: Mapping[str, Sequence[str]] | None = None,
    cytokine_dict: Mapping[str, Sequence[str]] | None = None,
    deg_refs: tuple[AnnData, AnnData] | None = None,
    min_cells: int = 20,
) -> dict[str, float]:
    """Run the biological metrics that have their required resources.

    Each metric is skipped (and reported as ``NaN``) if the required resource
    is missing — so the function is safe to call with whichever inputs the
    user happens to have. A warning is emitted per skipped metric.

    Parameters
    ----------
    s_genes, g2m_genes
        Required for ``cellcycle_*``.
    geneset_dict
        Required for ``coexpression``. If ``None``, the wrapper will try to
        fetch MSigDB Hallmark via omnipath; pass explicitly to avoid network.
    progeny_model, pathway_dict
        Required for ``pathway``. If both ``None``, fetched via decoupler.
    cytokine_dict
        Required for ``cytokine``. No fetch fallback.
    deg_refs
        Optional ``(ref_true, ref_pred)`` AnnData pair for ``deg_*``.
    min_cells
        Minimum cells-per-gene cutoff forwarded to ``metric_cellcycle``,
        ``metric_coexpression``, ``metric_pathway`` and ``metric_deg``.
        Lower this on small test sets (e.g. 100-cell tutorial slices) where
        the default of 20 would filter most cell-cycle / signature genes.
    """
    out: dict[str, float] = {}

    # Cell-cycle
    if s_genes is not None and g2m_genes is not None:
        try:
            cc = metric_cellcycle(adata_true, adata_pred,
                                  s_genes=s_genes, g2m_genes=g2m_genes,
                                  min_cells=min_cells)
            out["cellcycle_proportion_same_phase"] = cc.get("proportion_same_phase", float("nan"))
        except Exception as e:
            warnings.warn(f"cellcycle metric failed: {e}")
            out["cellcycle_proportion_same_phase"] = float("nan")
    else:
        warnings.warn("metric_cellcycle skipped: provide `s_genes` and `g2m_genes`.")
        out["cellcycle_proportion_same_phase"] = float("nan")

    # Coexpression (auto-fetch if geneset_dict is None)
    try:
        out["coexpression"] = metric_coexpression(
            adata_true, adata_pred, geneset_dict=geneset_dict,
            min_cells=min_cells,
        )
    except Exception as e:
        warnings.warn(f"coexpression metric failed: {e}")
        out["coexpression"] = float("nan")

    # Pathway (auto-fetch if progeny_model is None)
    try:
        out["pathway"] = metric_pathway(
            adata_true, adata_pred,
            progeny_model=progeny_model,
            pathway_dict=pathway_dict,
            min_cells=min_cells,
        )
    except Exception as e:
        warnings.warn(f"pathway metric failed: {e}")
        out["pathway"] = float("nan")

    # DEG (requires a reference pair — typically control). We surface the
    # "pred_*" variant because it uses the predicted reference (i.e. it is
    # the realistic case where the user only has predictions to compare with).
    if deg_refs is not None:
        try:
            ref_true, ref_pred = deg_refs
            deg = metric_deg(adata_true, adata_pred,
                             ref_true=ref_true, ref_pred=ref_pred,
                             min_cells=min_cells)
            out["deg_dice_at_100"] = deg.get("pred_deg_dice_100", float("nan"))
            out["deg_logfc_spearman"] = deg.get("pred_mean_genediff_spearman", float("nan"))
        except Exception as e:
            warnings.warn(f"deg metric failed: {e}")
            out["deg_dice_at_100"] = float("nan")
            out["deg_logfc_spearman"] = float("nan")
    else:
        out["deg_dice_at_100"] = float("nan")
        out["deg_logfc_spearman"] = float("nan")

    # Cytokine
    if cytokine_dict is not None:
        try:
            out["cytokine"] = metric_cytokine(
                adata_true, adata_pred, cytokine_dict=cytokine_dict,
            )
        except Exception as e:
            warnings.warn(f"cytokine metric failed: {e}")
            out["cytokine"] = float("nan")
    else:
        out["cytokine"] = float("nan")

    return out


def compute_perturbational_metrics(
    adata_pred: AnnData,
    adata_pert_true: AnnData,
    adata_ctrl: AnnData,
    *,
    k: int = 20,
    use_rep: str | None = None,
) -> dict[str, float]:
    """Run perturbational metrics (currently KNN purity)."""
    return {
        "knn_purity": metric_knn_purity(
            adata_pred=adata_pred,
            adata_pert_true=adata_pert_true,
            adata_ctrl=adata_ctrl,
            k=k,
            use_rep=use_rep,
        ),
    }


def compute_all_metrics(
    adata_true: AnnData,
    adata_pred: AnnData,
    *,
    # Stat
    # Bio resources
    s_genes: Sequence[str] | None = None,
    g2m_genes: Sequence[str] | None = None,
    geneset_dict: Mapping[str, Sequence[str]] | None = None,
    progeny_model: pd.DataFrame | None = None,
    pathway_dict: Mapping[str, Sequence[str]] | None = None,
    cytokine_dict: Mapping[str, Sequence[str]] | None = None,
    deg_refs: tuple[AnnData, AnnData] | None = None,
    min_cells: int = 20,
    # Perturbational
    perturb: tuple[AnnData, AnnData] | None = None,
    knn_k: int = 20,
    use_rep: str | None = None,
) -> dict[str, float]:
    """Run every metric for which the required inputs are available.

    Returns one flat ``{metric_name: score}`` dict. Metrics whose required
    inputs are missing are returned as ``NaN``.

    Parameters
    ----------
    perturb
        ``(adata_pert_true, adata_ctrl)`` if KNN purity should be computed
        against the predicted perturbation in ``adata_pred``.
    min_cells
        Minimum cells-per-gene cutoff forwarded to the biological metrics.
        Lower this on small test sets (e.g. 100-cell tutorial slices) where
        the default of 20 would filter most cell-cycle / signature genes.
    """
    scores: dict[str, float] = {}
    scores.update(compute_statistical_metrics(adata_true, adata_pred))
    scores.update(compute_biological_metrics(
        adata_true, adata_pred,
        s_genes=s_genes, g2m_genes=g2m_genes,
        geneset_dict=geneset_dict,
        progeny_model=progeny_model, pathway_dict=pathway_dict,
        cytokine_dict=cytokine_dict, deg_refs=deg_refs,
        min_cells=min_cells,
    ))
    if perturb is not None:
        adata_pert_true, adata_ctrl = perturb
        scores.update(compute_perturbational_metrics(
            adata_pred=adata_pred,
            adata_pert_true=adata_pert_true,
            adata_ctrl=adata_ctrl,
            k=knn_k, use_rep=use_rep,
        ))
    return scores


# ---------------------------------------------------------------------------
# Rank-percentile aggregation (paper formula)
# ---------------------------------------------------------------------------

def aggregate_rank_percentile(
    scores: pd.DataFrame,
    higher_is_better: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """Convert raw per-method, per-metric scores to rank-percentiles.

    Implements the paper's formula::

        rp_i(m) = (n - rank_i(m)) / (n - 1)

    where ``n`` is the number of methods, ``rank 1`` is best, and the
    output range is ``[0, 1]`` — best method gets 1.0, worst gets 0.0.

    Parameters
    ----------
    scores
        DataFrame indexed by method (rows) and metric names (columns).
    higher_is_better
        Mapping ``{metric_name: bool}`` for each column of ``scores``.
        If ``None``, falls back to :data:`HIGHER_IS_BETTER`; missing keys
        default to ``True``.

    Returns
    -------
    pd.DataFrame
        Same shape as input, values in ``[0, 1]``; per-column NaN for ties.
    """
    if higher_is_better is None:
        higher_is_better = HIGHER_IS_BETTER

    out = pd.DataFrame(index=scores.index, columns=scores.columns, dtype=float)
    for col in scores.columns:
        s = scores[col].astype(float)
        valid = s.dropna()
        M = len(valid)
        if M < 2 or valid.nunique() <= 1:
            out[col] = float("nan")
            continue
        hib = higher_is_better.get(col, True) if higher_is_better is not None else True
        # rank 1 = best: ascending=False for higher-is-better, ascending=True for lower
        ranks = s.rank(method="average", ascending=(not hib))
        out[col] = (M - ranks) / (M - 1)
    return out
