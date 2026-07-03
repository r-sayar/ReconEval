"""Shared helpers for OpenProblems ReconEval metric components."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def read_expression(adata: AnnData, layer: str) -> AnnData:
    """Return a view/copy of ``adata`` with the chosen matrix in ``.X``."""
    if layer == "X":
        return adata
    if layer in adata.layers:
        out = adata.copy()
        out.X = adata.layers[layer]
        return out
    if layer in adata.obsm:
        out = AnnData(
            X=np.asarray(adata.obsm[layer]),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )
        return out
    raise KeyError(
        f"Layer {layer!r} not found in .X, .layers, or .obsm of AnnData."
    )


def align_genes(
    solution: AnnData,
    prediction: AnnData,
    *,
    resolve_genes: str,
) -> tuple[AnnData, AnnData]:
    """Subset solution and prediction to a shared gene axis."""
    if resolve_genes == "solution":
        genes = solution.var_names
    elif resolve_genes == "intersection":
        genes = solution.var_names.intersection(prediction.var_names)
    else:
        raise ValueError(f"Unknown resolve_genes={resolve_genes!r}")

    if len(genes) == 0:
        raise ValueError("No shared genes between solution and prediction.")

    return solution[:, genes].copy(), prediction[:, genes].copy()


def write_score(
    *,
    dataset_id: str,
    normalization_id: str,
    method_id: str,
    metric_ids: list[str],
    metric_values: list[float],
    output_path: str,
) -> None:
    import anndata as ad

    output = ad.AnnData(
        uns={
            "dataset_id": dataset_id,
            "normalization_id": normalization_id,
            "method_id": method_id,
            "metric_ids": metric_ids,
            "metric_values": [float(v) for v in metric_values],
        }
    )
    output.write_h5ad(output_path, compression="gzip")
