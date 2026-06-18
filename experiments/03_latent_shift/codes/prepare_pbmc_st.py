"""Prepare unified PBMC State Transition h5ad files from zarr data.

Builds ONE h5ad per split (train/val/test) containing the HVG expression
matrix and model embeddings as separate obsm keys.  PBS controls are
included in every split.

Supports two modes:
  - Build (default): create h5ad from scratch with X, obs, and specified
    embeddings.
  - Append (--append): open existing h5ad, add new obsm keys via h5py
    without re-reading or rewriting X and obs.

Output:
  pbmc_st/train/train.h5ad
  pbmc_st/val/val.h5ad
  pbmc_st/test/test.h5ad

Usage:
  # First run — build with foundation models
  python prepare_pbmc_st.py --models SE scGPT scConcept scimilarity

  # Later — append end-to-end model embeddings
  python prepare_pbmc_st.py --models AE_2048 nlscVI_2048 PCA_2048 --append

  # Add a new dimension variant
  python prepare_pbmc_st.py --models AE_512 --append
"""

import argparse
import gc
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from tqdm import tqdm

DATA_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction"
HVG_ZARR = os.path.join(DATA_ROOT, "pbmc/comb_w_obs.zarr")
SPLIT_ZARR = os.path.join(DATA_ROOT, "pbmc/split03/split_metadata.zarr")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "pbmc_st")

EMBEDDING_CONFIGS = {
    "SE": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_SE/SE_emb.zarr"),
        "latent_dim": 2058,
    },
    "scGPT": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_scGPT/scGPT_emb.zarr"),
        "latent_dim": 512,
    },
    "scConcept": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_scConcept/scConcept_emb.zarr"),
        "latent_dim": 512,
    },
    "scimilarity": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_scimilarity/scimilarity_emb.zarr"),
        "latent_dim": 128,
    },
    "AE_2048": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_ae_2048/AE_emb.zarr"),
        "latent_dim": 2048,
    },
    "nlscVI_2048": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_nlscvi_2048/nlscVI_emb.zarr"),
        "latent_dim": 2048,
    },
    "PCA_2048": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_pca_2048/PCA_emb.zarr"),
        "latent_dim": 2048,
    },
    "AE_128": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_ae_128/AE_emb.zarr"),
        "latent_dim": 128,
    },
    "nlscVI_128": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_nlscvi_128/nlscVI_emb.zarr"),
        "latent_dim": 128,
    },
    "PCA_128": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_pca_128/PCA_emb.zarr"),
        "latent_dim": 128,
    },
    "AE_512": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_ae_512/AE_emb.zarr"),
        "latent_dim": 512,
    },
    "nlscVI_512": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_nlscvi_512/nlscVI_emb.zarr"),
        "latent_dim": 512,
    },
    "PCA_512": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_pca_512/PCA_emb.zarr"),
        "latent_dim": 512,
    },
    "AE_32": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_ae_32/AE_emb.zarr"),
        "latent_dim": 32,
    },
    "nlscVI_32": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_nlscvi_32/nlscVI_emb.zarr"),
        "latent_dim": 32,
    },
    "PCA_32": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_pca_32/PCA_emb.zarr"),
        "latent_dim": 32,
    },
    "AE_10": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_ae_10/AE_emb.zarr"),
        "latent_dim": 10,
    },
    "nlscVI_10": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_nlscvi_10/nlscVI_emb.zarr"),
        "latent_dim": 10,
    },
    "PCA_10": {
        "zarr_path": os.path.join(DATA_ROOT, "pbmc_pca_10/PCA_emb.zarr"),
        "latent_dim": 10,
    },
}


def load_split_metadata():
    root = zarr.open(SPLIT_ZARR, mode="r")
    train = list(root.attrs["train_combinations"])
    val = list(root.attrs["val_combinations"])
    test = list(root.attrs["test_combinations"])
    return train, val, test


def parse_combination(comb: str):
    """Parse '{cell_type}-{donor}-{cytokine}' with rsplit to handle hyphens in cell_type."""
    parts = comb.rsplit("-", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"Cannot parse combination (expected 2+ hyphens): {comb}")


def get_zarr_array(group):
    """Extract array from zarr group, handling group/X and direct-array layouts."""
    if isinstance(group, zarr.Array):
        return group[:]
    if "X" in group:
        return group["X"][:]
    arr_keys = list(group.array_keys())
    if arr_keys:
        return group[arr_keys[0]][:]
    raise ValueError(f"No array found in zarr group: {group.path}")


def get_pbs_combinations(all_combs):
    return [c for c in all_combs if parse_combination(c)[2] == "PBS"]


# ── Build mode ───────────────────────────────────────────────────────


def build_unified_h5ad(combs, hvg_root, emb_roots, model_names):
    """Build a single AnnData with X + all model embeddings in obsm."""
    x_chunks = []
    emb_chunks = {m: [] for m in model_names}
    obs_rows = []
    skipped = []

    for comb in tqdm(combs, desc="  combinations", leave=False):
        cell_type, donor, cytokine = parse_combination(comb)

        if comb not in hvg_root:
            skipped.append((comb, "missing in HVG zarr"))
            continue

        missing_emb = [m for m in model_names if comb not in emb_roots[m]]
        if missing_emb:
            skipped.append((comb, f"missing in emb zarr: {missing_emb}"))
            continue

        try:
            x = get_zarr_array(hvg_root[comb])
        except Exception as e:
            skipped.append((comb, f"X read error: {e}"))
            continue

        n = x.shape[0]
        if n == 0:
            print(f"  skip {comb}: zero cells")
            continue

        embs = {}
        try:
            for m in model_names:
                e = get_zarr_array(emb_roots[m][comb])
                if e.shape[0] != n:
                    raise ValueError(
                        f"cell count mismatch in '{m}': "
                        f"HVG has {n} cells, embedding has {e.shape[0]}"
                    )
                embs[m] = e
        except Exception as e:
            skipped.append((comb, f"emb error: {e}"))
            continue

        x_chunks.append(sp.csr_matrix(x.astype(np.float32)))
        for m in model_names:
            emb_chunks[m].append(embs[m].astype(np.float32))

        for _ in range(n):
            obs_rows.append({
                "cell_type": cell_type,
                "donor": donor,
                "target_gene": cytokine,
                "combination": comb,
            })

    if skipped:
        print(f"    skipped {len(skipped)} combos: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    if not x_chunks:
        return None

    X = sp.vstack(x_chunks)
    obs = pd.DataFrame(obs_rows)
    adata = ad.AnnData(X=X, obs=obs)

    for m in model_names:
        adata.obsm[f"X_{m}"] = np.concatenate(emb_chunks[m], axis=0)

    return adata


# ── Append mode ──────────────────────────────────────────────────────


def _read_obs_combination(h5ad_path):
    """Read the combination column from an h5ad file using h5py.

    Handles both categorical encodings:
      - AnnData >=0.9 style: obs["combination"] is a Group with "codes" and "categories" datasets
      - Legacy style: obs["combination"] is a Dataset with a "categories" attribute (HDF5 reference)
      - Plain strings: obs["combination"] is a Dataset of strings
    """
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        comb_node = obs["combination"]

        # AnnData >=0.9: categorical stored as Group with codes + categories datasets
        if isinstance(comb_node, h5py.Group):
            cats = comb_node["categories"][:]
            codes = comb_node["codes"][:]
            cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
            return cats[codes]

        # Legacy: Dataset with a "categories" attribute (HDF5 reference or string key)
        if "categories" in comb_node.attrs:
            cat_ref = comb_node.attrs["categories"]
            if isinstance(cat_ref, h5py.Reference):
                cats = f[cat_ref][:]
            else:
                cats = obs[cat_ref][:]
            cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
            codes = comb_node[:]
            return cats[codes]

        # Plain string dataset
        raw = comb_node[:]
        return np.array([v.decode() if isinstance(v, bytes) else v for v in raw])


def _get_existing_obsm_keys(h5ad_path):
    with h5py.File(h5ad_path, "r") as f:
        if "obsm" in f:
            return list(f["obsm"].keys())
        return []


def _build_comb_slices(combinations):
    """Map each combination to its row slice (contiguous block)."""
    slices = {}
    prev = None
    start = 0
    for i, c in enumerate(combinations):
        if c != prev:
            if prev is not None:
                slices[prev] = slice(start, i)
            start = i
            prev = c
    if prev is not None:
        slices[prev] = slice(start, len(combinations))
    return slices


def append_embeddings(h5ad_path, model_names, emb_roots, overwrite=False):
    """Append new obsm arrays to an existing h5ad via h5py."""
    existing_keys = _get_existing_obsm_keys(h5ad_path)

    if overwrite:
        # Delete existing keys so they get re-written from zarr
        with h5py.File(h5ad_path, "a") as f:
            for m in model_names:
                key = f"X_{m}"
                if key in existing_keys and f"obsm/{key}" in f:
                    del f[f"obsm/{key}"]
                    print(f"    Deleted existing {key} (--overwrite)")
        existing_keys = _get_existing_obsm_keys(h5ad_path)

    new_models = [m for m in model_names if f"X_{m}" not in existing_keys]

    if not new_models:
        print(f"    All requested models already in {h5ad_path}")
        return

    print(f"    Existing obsm: {existing_keys}")
    print(f"    Appending: {[f'X_{m}' for m in new_models]}")

    combinations = _read_obs_combination(h5ad_path)
    n_total = len(combinations)
    comb_slices = _build_comb_slices(combinations)

    with h5py.File(h5ad_path, "a") as f:
        if "obsm" not in f:
            f.create_group("obsm")

        for m in new_models:
            cfg = EMBEDDING_CONFIGS[m]
            emb_root = emb_roots[m]
            dim = cfg["latent_dim"]

            emb_array = np.full((n_total, dim), np.nan, dtype=np.float32)
            n_filled = 0
            n_missing = 0

            for comb, s in tqdm(comb_slices.items(), desc=f"    X_{m}", leave=False):
                n_cells = s.stop - s.start
                if comb not in emb_root:
                    n_missing += 1
                    continue
                try:
                    e = get_zarr_array(emb_root[comb]).astype(np.float32)
                except Exception:
                    n_missing += 1
                    continue
                n_use = min(n_cells, e.shape[0])
                emb_array[s.start : s.start + n_use] = e[:n_use]
                n_filled += n_use

            f.create_dataset(f"obsm/X_{m}", data=emb_array)

            pct = 100 * n_filled / n_total if n_total else 0
            print(f"    X_{m}: shape=({n_total}, {dim}), "
                  f"filled={n_filled:,} ({pct:.1f}%), "
                  f"missing_combs={n_missing}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare unified PBMC ST h5ad files from zarr data",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(EMBEDDING_CONFIGS.keys()),
        help=f"Embedding models to include (available: {list(EMBEDDING_CONFIGS.keys())})",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append new obsm keys to existing h5ad files instead of rebuilding",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="With --append, delete and re-write existing obsm keys instead of skipping them",
    )
    args = parser.parse_args()

    model_names = [m for m in args.models if m in EMBEDDING_CONFIGS]
    unknown = [m for m in args.models if m not in EMBEDDING_CONFIGS]
    if unknown:
        print(f"WARNING: unknown models ignored: {unknown}")
    if not model_names:
        print(f"No valid models. Available: {list(EMBEDDING_CONFIGS.keys())}")
        return

    print("Loading split03 metadata ...")
    train, val, test = load_split_metadata()
    all_combs = sorted(set(train + val + test))
    pbs = get_pbs_combinations(all_combs)

    donors, cts, cys = set(), set(), set()
    for c in all_combs[:200]:
        ct, d, cy = parse_combination(c)
        donors.add(d); cts.add(ct); cys.add(cy)

    print(f"split03: {len(train)} train, {len(val)} val, {len(test)} test")
    print(f"PBS combos: {len(pbs)}")
    print(f"donors (sample): {sorted(donors)}")
    print(f"cell types (sample): {sorted(cts)[:5]} ...")
    print(f"cytokines (sample): {sorted(cys)[:5]} ...")
    print(f"Models: {model_names}  (mode={'append' if args.append else 'build'})")

    # Open embedding zarrs
    emb_roots = {}
    for m in model_names:
        cfg = EMBEDDING_CONFIGS[m]
        emb_roots[m] = zarr.open(cfg["zarr_path"], mode="r")
        n_groups = len(list(emb_roots[m].group_keys()))
        print(f"  {m}: {n_groups} groups in {cfg['zarr_path']}")

    pbs_set = set(pbs)
    splits = {
        "train": sorted(set(train) | pbs_set),
        "val": sorted(set(val) | pbs_set),
        "test": sorted(set(test) | pbs_set),
    }

    hvg_root = zarr.open(HVG_ZARR, mode="r") if not args.append else None

    for split_name, combs in splits.items():
        out_dir = os.path.join(OUTPUT_ROOT, split_name)
        os.makedirs(out_dir, exist_ok=True)
        h5ad_path = os.path.join(out_dir, f"{split_name}.h5ad")

        print(f"\n{'=' * 60}")

        if args.append:
            if not os.path.exists(h5ad_path):
                print(f"  {h5ad_path} does not exist — skipping (run without --append first)")
                continue
            print(f"Appending to {split_name}.h5ad")
            print(f"{'=' * 60}")
            append_embeddings(h5ad_path, model_names, emb_roots, overwrite=args.overwrite)
        else:
            print(f"Building {split_name}.h5ad ({len(combs)} combos, {len(model_names)} embeddings)")
            print(f"{'=' * 60}")

            adata = build_unified_h5ad(combs, hvg_root, emb_roots, model_names)
            if adata is not None:
                adata.write_h5ad(h5ad_path)
                emb_summary = ", ".join(
                    f"X_{m}={adata.obsm[f'X_{m}'].shape[1]}d" for m in model_names
                )
                print(f"  {split_name}: {adata.n_obs:,} cells, X={adata.X.shape[1]} genes")
                print(f"  obsm: {emb_summary}")
                print(f"  -> {h5ad_path}")
                del adata
                gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()
