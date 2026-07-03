#!/usr/bin/env python3
"""Generate minimal AnnData fixtures for OpenProblems Viash component tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData


def _load_gene_list(path: Path) -> list[str]:
    genes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            genes.append(line)
    return genes


def _make_expression(
    rng: np.random.Generator,
    n_obs: int,
    genes: list[str],
    *,
    shift_genes: list[str] | None = None,
    noise: float = 0.3,
) -> AnnData:
    n_genes = len(genes)
    x = rng.poisson(2.0, size=(n_obs, n_genes)).astype(np.float32)
    if shift_genes:
        idx = [genes.index(g) for g in shift_genes if g in genes]
        if idx:
            x[:, idx] += rng.poisson(3.0, size=(n_obs, len(idx)))
    var = pd.DataFrame(index=genes)
    return AnnData(x, var=var)


def _with_uns(adata: AnnData, *, dataset_id: str, normalization_id: str, method_id: str | None = None) -> AnnData:
    adata = adata.copy()
    adata.uns["dataset_id"] = dataset_id
    adata.uns["normalization_id"] = normalization_id
    if method_id is not None:
        adata.uns["method_id"] = method_id
    return adata


def _add_noise(true: AnnData, rng: np.random.Generator, scale: float = 0.35) -> AnnData:
    pred_x = true.X + rng.normal(0.0, scale, size=true.shape).astype(np.float32)
    pred_x = np.clip(pred_x, 0.0, None)
    return AnnData(pred_x, obs=true.obs.copy(), var=true.var.copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "resources_test" / "reconeval_demo",
    )
    parser.add_argument(
        "--cell-cycle-genes",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src"
        / "metrics"
        / "biological"
        / "resources"
        / "regev_lab_cell_cycle_genes.txt",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    cc_genes = _load_gene_list(args.cell_cycle_genes)
    extra_genes = [f"GENE_{i}" for i in range(200)]
    genes = list(dict.fromkeys(cc_genes + extra_genes))

    dataset_id = "reconeval_demo"
    normalization_id = "log1p"

    pert = _make_expression(rng, 120, genes, shift_genes=cc_genes[:10])
    ctrl = _make_expression(rng, 120, genes)
    pert_pred = _add_noise(pert, rng)
    ctrl_pred = _add_noise(ctrl, rng)

    solution = _with_uns(pert, dataset_id=dataset_id, normalization_id=normalization_id)
    prediction = _with_uns(
        _add_noise(pert, rng, scale=0.25),
        dataset_id=dataset_id,
        normalization_id=normalization_id,
        method_id="demo_method",
    )
    reference_solution = _with_uns(ctrl, dataset_id=dataset_id, normalization_id=normalization_id)
    reference_prediction = _with_uns(
        ctrl_pred,
        dataset_id=dataset_id,
        normalization_id=normalization_id,
        method_id="demo_method",
    )
    solution_perturbed = _with_uns(pert, dataset_id=dataset_id, normalization_id=normalization_id)
    solution_control = _with_uns(ctrl, dataset_id=dataset_id, normalization_id=normalization_id)
    prediction_perturbed = _with_uns(
        pert_pred,
        dataset_id=dataset_id,
        normalization_id=normalization_id,
        method_id="demo_method",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solution.write_h5ad(args.output_dir / "solution.h5ad", compression="gzip")
    prediction.write_h5ad(args.output_dir / "prediction.h5ad", compression="gzip")
    reference_solution.write_h5ad(args.output_dir / "reference_solution.h5ad", compression="gzip")
    reference_prediction.write_h5ad(args.output_dir / "reference_prediction.h5ad", compression="gzip")
    solution_perturbed.write_h5ad(args.output_dir / "solution_perturbed.h5ad", compression="gzip")
    solution_control.write_h5ad(args.output_dir / "solution_control.h5ad", compression="gzip")
    prediction_perturbed.write_h5ad(args.output_dir / "prediction_perturbed.h5ad", compression="gzip")
    print(f"Wrote test resources to {args.output_dir}")


if __name__ == "__main__":
    main()
