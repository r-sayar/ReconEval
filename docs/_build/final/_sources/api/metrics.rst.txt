metrics
=======

.. currentmodule:: sc_reconstruction.metrics

The metrics module scores a ``(true, reconstructed)`` AnnData pair across
three metric families (statistical, biological, perturbational), aggregates
per-method via the rank-percentile, and renders the Fig 3 funky map.

One-call wrappers and aggregation
---------------------------------

.. toctree::
    :maxdepth: 1

    _autosummary/sc_reconstruction.metrics.compute_all_metrics
    _autosummary/sc_reconstruction.metrics.compute_statistical_metrics
    _autosummary/sc_reconstruction.metrics.compute_biological_metrics
    _autosummary/sc_reconstruction.metrics.compute_perturbational_metrics
    _autosummary/sc_reconstruction.metrics.aggregate_rank_percentile
    _autosummary/sc_reconstruction.metrics.funky_heatmap
    _autosummary/sc_reconstruction.metrics.HIGHER_IS_BETTER

Individual metrics
------------------

.. toctree::
    :maxdepth: 1

    _autosummary/sc_reconstruction.metrics.metric_r2
    _autosummary/sc_reconstruction.metrics.metric_mse
    _autosummary/sc_reconstruction.metrics.metric_energy_distance
    _autosummary/sc_reconstruction.metrics.metric_cellcycle
    _autosummary/sc_reconstruction.metrics.metric_pathway
    _autosummary/sc_reconstruction.metrics.metric_coexpression
    _autosummary/sc_reconstruction.metrics.metric_deg
    _autosummary/sc_reconstruction.metrics.metric_cytokine
    _autosummary/sc_reconstruction.metrics.metric_knn_purity

Resource loaders
----------------

.. toctree::
    :maxdepth: 1

    _autosummary/sc_reconstruction.metrics.load_cell_cycle_genes
    _autosummary/sc_reconstruction.metrics.load_cytokine_dict_from_csv
    _autosummary/sc_reconstruction.metrics.load_progeny
