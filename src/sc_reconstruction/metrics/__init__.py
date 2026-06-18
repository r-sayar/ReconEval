"""Public metrics API for ReconEval.

Only the lightweight wrapper in :mod:`sc_reconstruction.metrics.api` is
exported. The batch-evaluation infrastructure used by the paper's
reproduction pipeline lives at ``sc_reconstruction.metrics.base_eval``
(`MetricsBatchEvaluator`) and in the per-family modules (``_cellcycle``,
``_coexpression``, ``_cytokine``, ``_deg``, ``_pathway``,
``distributional``) — reach for those only when you need the
zarr/multiprocessing path.

Example
-------
>>> from sc_reconstruction.metrics import compute_all_metrics
>>> scores = compute_all_metrics(adata_true, adata_pred,
...                              s_genes=s_genes, g2m_genes=g2m_genes)
"""

from .api import (
    HIGHER_IS_BETTER,
    aggregate_rank_percentile,
    compute_all_metrics,
    compute_biological_metrics,
    compute_perturbational_metrics,
    compute_statistical_metrics,
    load_cell_cycle_genes,
    load_cytokine_dict_from_csv,
    load_progeny,
    metric_cellcycle,
    metric_coexpression,
    metric_cytokine,
    metric_deg,
    metric_energy_distance,
    metric_knn_purity,
    metric_mse,
    metric_pathway,
    metric_r2,
)
from .plotting import funky_heatmap

__all__ = [
    "HIGHER_IS_BETTER",
    "aggregate_rank_percentile",
    "compute_all_metrics",
    "compute_biological_metrics",
    "compute_perturbational_metrics",
    "compute_statistical_metrics",
    "funky_heatmap",
    "load_cell_cycle_genes",
    "load_cytokine_dict_from_csv",
    "load_progeny",
    "metric_cellcycle",
    "metric_coexpression",
    "metric_cytokine",
    "metric_deg",
    "metric_energy_distance",
    "metric_knn_purity",
    "metric_mse",
    "metric_pathway",
    "metric_r2",
]
