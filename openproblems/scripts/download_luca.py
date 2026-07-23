#!/usr/bin/env python3
"""Download a LuCA subset from CZ CELLxGENE Census for the OpenProblems demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

LUCA_DATASET_ID = "232f6a5a-a04c-4758-a6e8-88ab2e3a6e69"
LUCA_COLLECTION_URL = (
    "https://cellxgene.cziscience.com/collections/"
    "edb893ee-4066-4128-9aec-5eb2b03f8287"
)
DEFAULT_CENSUS_VERSION = "2024-07-01"


def download_from_census(
    *,
    n_cells: int,
    census_version: str,
    seed: int,
) -> ad.AnnData:
    import cellxgene_census

    print(f"Opening CELLxGENE Census ({census_version})...", flush=True)
    with cellxgene_census.open_soma(census_version=census_version) as census:
        # Memory-safe sampling: read only the soma_joinids for LuCA (lightweight),
        # sample from them, and then pull expression for the sampled cells via
        # obs_coords. Pulling all ~892k LuCA cells into host RAM before
        # subsampling OOMs small machines.
        # LuCA is an aggregated atlas -- every cell has is_primary_data==False
        # (primary records live under constituent datasets) -- so we filter on
        # dataset_id only.
        print(f"Listing LuCA cells (dataset {LUCA_DATASET_ID})...", flush=True)
        obs = (
            census["census_data"]["homo_sapiens"]
            .obs.read(
                value_filter=f"dataset_id == '{LUCA_DATASET_ID}'",
                column_names=["soma_joinid"],
            )
            .concat()
            .to_pandas()
        )
        ids = obs["soma_joinid"].to_numpy()
        n_keep = min(n_cells, len(ids))
        print(f"LuCA has {len(ids)} cells; sampling {n_keep}...", flush=True)
        rng = np.random.default_rng(seed)
        coords = np.sort(rng.choice(ids, size=n_keep, replace=False)).tolist()
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            obs_coords=coords,
            obs_column_names=[
                "cell_type",
                "dataset_id",
                "assay",
                "donor_id",
                "tissue",
                "tissue_general",
                "disease",
            ],
            var_column_names=["feature_id", "feature_name"],
        )

    print(f"Downloaded {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    return adata


def make_fallback_luca(*, n_cells: int, n_genes: int, seed: int) -> ad.AnnData:
    """Synthetic LuCA-like dataset when Census is unavailable."""
    print("Census unavailable — generating synthetic LuCA-like data.", flush=True)
    rng = np.random.default_rng(seed)
    origins = rng.choice(["tumor_primary", "normal_adjacent"], size=n_cells, p=[0.6, 0.4])
    cell_types = rng.choice(
        ["AT2", "AT1", "Club", "Ciliated", "Macrophage"],
        size=n_cells,
    )
    datasets = rng.choice(["LuCA1", "LuCA2", "LuCA3"], size=n_cells)
    genes = [f"GENE_{i}" for i in range(n_genes)]
    import pandas as pd

    x = rng.poisson(1.5, size=(n_cells, n_genes)).astype(np.float32)
    tumor = origins == "tumor_primary"
    x[tumor, :50] += rng.poisson(2.0, size=(tumor.sum(), 50))

    obs = pd.DataFrame(
        {
            "cell_type": cell_types,
            "dataset": datasets,
            "origin": origins,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata


def prepare_common_dataset(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str)
        adata.var_names_make_unique()

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Derive a tumor/normal axis from the disease annotation so the dataset is
    # ready for the biological metric's perturbational (DEG) metrics.
    if "disease" in adata.obs.columns:
        dis = adata.obs["disease"].astype(str)
        adata.obs["origin"] = np.where(
            dis.str.contains("normal", case=False), "normal", "tumor"
        )

    adata.uns["dataset_id"] = "luca"
    adata.uns["dataset_name"] = "Human Lung Cancer Cell Atlas (LuCA)"
    adata.uns["dataset_url"] = LUCA_COLLECTION_URL
    adata.uns["dataset_summary"] = (
        "Subset of the Human Lung Cancer Cell Atlas from CZ CELLxGENE."
    )
    adata.uns["normalization_id"] = "counts"
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("openproblems/resources_test/common/luca/dataset.h5ad"),
    )
    parser.add_argument("--n-cells", type=int, default=3000)
    parser.add_argument("--census-version", default=DEFAULT_CENSUS_VERSION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Skip Census download and use synthetic data.",
    )
    args = parser.parse_args()

    if args.fallback:
        adata = make_fallback_luca(n_cells=args.n_cells, n_genes=2000, seed=args.seed)
    else:
        try:
            adata = download_from_census(
                n_cells=args.n_cells,
                census_version=args.census_version,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"Census download failed: {exc}", flush=True)
            adata = make_fallback_luca(n_cells=args.n_cells, n_genes=2000, seed=args.seed)

    adata = prepare_common_dataset(adata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    print(f"Wrote {args.output} ({adata.n_obs} x {adata.n_vars})", flush=True)


if __name__ == "__main__":
    main()
