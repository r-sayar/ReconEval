#!/usr/bin/env python3
"""Run the ReconEval OpenProblems demo end-to-end on LuCA (no Viash required)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
OP = ROOT / "openproblems"
COMMON = OP / "resources_test" / "common" / "luca"
TASK = OP / "resources_test" / "reconeval" / "luca"
OUTPUT = OP / "output" / "demo"

sys.path.insert(0, str(ROOT / "src"))
from sc_reconstruction.metrics import compute_statistical_metrics  # noqa: E402


def _dense(x):
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


def run_process_dataset(
    input_path: Path,
    train_path: Path,
    test_path: Path,
    solution_path: Path,
    *,
    n_hvg: int = 500,
) -> None:
    print(">> process_dataset", flush=True)
    adata = ad.read_h5ad(input_path)
    x = _dense(adata.X)
    if adata.uns.get("normalization_id") == "counts":
        totals = x.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        x = x / totals * 1e4
        x = np.log1p(x)
        adata = adata.copy()
        adata.X = x

    gene_var = np.var(_dense(adata.X), axis=0)
    n_top = min(n_hvg, adata.n_vars - 1)
    top_ix = np.argsort(gene_var)[-n_top:]
    adata = adata[:, top_ix].copy()

    rng = np.random.default_rng(0)
    n_test = max(1, int(round(0.2 * adata.n_obs)))
    test_ix = rng.choice(adata.n_obs, size=n_test, replace=False)
    is_test = np.zeros(adata.n_obs, dtype=bool)
    is_test[test_ix] = True

    meta = {
        "dataset_id": adata.uns.get("dataset_id", "luca"),
        "normalization_id": "log1p_cp10k",
    }
    train = adata[~is_test].copy()
    test = adata[is_test].copy()
    for obj in (train, test):
        obj.uns.update(meta)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train.write_h5ad(train_path, compression="gzip")
    test.write_h5ad(test_path, compression="gzip")
    test.copy().write_h5ad(solution_path, compression="gzip")
    print(f"   train {train.shape}, test {test.shape}", flush=True)


def run_pca_method(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    *,
    n_components: int = 64,
) -> None:
    print(">> pca_reconstruction", flush=True)
    train = ad.read_h5ad(train_path)
    test = ad.read_h5ad(test_path)
    x_train = _dense(train.X)
    x_test = _dense(test.X)
    k = min(n_components, x_train.shape[0] - 1, x_train.shape[1])
    pca = PCA(n_components=k, random_state=0)
    pca.fit(x_train)
    x_pred = pca.inverse_transform(pca.transform(x_test))

    out = ad.AnnData(
        X=x_pred.astype(np.float32),
        obs=test.obs.copy(),
        var=test.var.copy(),
        uns={
            "dataset_id": train.uns["dataset_id"],
            "normalization_id": train.uns["normalization_id"],
            "method_id": "pca_reconstruction",
        },
    )
    out.obs_names = test.obs_names
    out.var_names = test.var_names
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output_path, compression="gzip")


def run_metrics(solution_path: Path, prediction_path: Path, output_path: Path) -> dict:
    print(">> statistical metrics", flush=True)
    solution = ad.read_h5ad(solution_path)
    prediction = ad.read_h5ad(prediction_path)
    genes = solution.var_names.intersection(prediction.var_names)
    solution = solution[:, genes].copy()
    prediction = prediction[:, genes].copy()
    scores = compute_statistical_metrics(solution, prediction)

    out = ad.AnnData(
        uns={
            "dataset_id": prediction.uns["dataset_id"],
            "normalization_id": prediction.uns["normalization_id"],
            "method_id": prediction.uns["method_id"],
            "metric_ids": list(scores.keys()),
            "metric_values": [float(v) for v in scores.values()],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output_path, compression="gzip")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=3000)
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    dataset_path = COMMON / "dataset.h5ad"
    if not args.skip_download:
        cmd = [
            sys.executable,
            str(OP / "scripts" / "download_luca.py"),
            "--output",
            str(dataset_path),
            "--n-cells",
            str(args.n_cells),
        ]
        if args.fallback:
            cmd.append("--fallback")
        subprocess.run(cmd, check=True)

    run_process_dataset(
        dataset_path,
        TASK / "train.h5ad",
        TASK / "test.h5ad",
        TASK / "solution.h5ad",
    )
    run_pca_method(
        TASK / "train.h5ad",
        TASK / "test.h5ad",
        TASK / "prediction.h5ad",
    )
    scores = run_metrics(
        TASK / "solution.h5ad",
        TASK / "prediction.h5ad",
        OUTPUT / "score.h5ad",
    )

    print("\n=== LuCA demo results (pca_reconstruction) ===", flush=True)
    for name, value in scores.items():
        print(f"  {name}: {value:.4f}", flush=True)
    print(f"\nArtifacts: {TASK}  |  {OUTPUT / 'score.h5ad'}", flush=True)


if __name__ == "__main__":
    main()
