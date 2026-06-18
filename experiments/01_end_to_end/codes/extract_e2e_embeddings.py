from pathlib import Path
"""Extract cell embeddings from end-to-end models (AE, nlscVI, PCA).

Reads HVG expression from a zarr store, runs the specified model's encoder,
and saves embeddings in the same zarr layout as foundation model embeddings
(e.g. SE_emb.zarr/{combination}/X).

Usage:
    # AE 2048 (existing)
    python extract_e2e_embeddings.py \\
        --model AE \\
        --ckpt  weights/pbmc/split03/AE/Default/400_[4096]_2048.../epoch=116...ckpt \\
        --out   data/reconstruction/pbmc_ae_2048/AE_emb.zarr

    # AE 128 (new)
    python extract_e2e_embeddings.py \\
        --model AE \\
        --ckpt  weights/pbmc/split03/AE/Default/400_[2048]_128.../epoch=XX...ckpt \\
        --out   data/reconstruction/pbmc_ae_128/AE_emb.zarr

    # PCA 128
    python extract_e2e_embeddings.py \\
        --model PCA \\
        --pca_mean weights/pbmc/split03/PCA/Default/mean.zarr \\
        --pca_pc   weights/pbmc/split03/PCA/Default/pc_128.zarr \\
        --out      data/reconstruction/pbmc_pca_128/PCA_emb.zarr

    # nlscVI 2048
    python extract_e2e_embeddings.py \\
        --model nlscVI \\
        --ckpt  weights/.../epoch=65-step=88902-...pt \\
        --out   data/reconstruction/pbmc_nlscvi_2048/nlscVI_emb.zarr
"""

import argparse
import gc
import os
import sys

import numpy as np
import torch
import zarr
from numcodecs import Blosc
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

DEFAULT_HVG_ZARR = (
    "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/comb_w_obs.zarr"
)
_COMPRESSOR = Blosc(cname="zstd", clevel=3)


# ---------------------------------------------------------------------------
# Zarr helpers
# ---------------------------------------------------------------------------

def get_zarr_array(group):
    """Handle group/X and direct-array layouts."""
    if isinstance(group, zarr.Array):
        return group[:]
    if "X" in group:
        return group["X"][:]
    arr_keys = list(group.array_keys())
    if arr_keys:
        return group[arr_keys[0]][:]
    raise ValueError(f"No array found in zarr group: {group.path}")


def write_zarr(root, comb, emb):
    g = root.require_group(comb)
    g.array("X", emb, compressor=_COMPRESSOR, overwrite=True)


# ---------------------------------------------------------------------------
# Extractors — all return (n_cells_written: int)
# ---------------------------------------------------------------------------

def extract_ae(hvg_root, output_path, ckpt, device, batch_size):
    from sc_reconstruction.models.reconae import ReconAE

    model = ReconAE(input_dim=1, n_hidden=[1], n_latent=1)
    model.load(ckpt, map_location=device)
    print(f"  input_dim={model.module.input_dim}  n_latent={model.module.n_latent}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    root = zarr.open(output_path, mode="w")
    combs = sorted(hvg_root.group_keys())
    n_total = 0

    for comb in tqdm(combs, desc="AE"):
        try:
            X = get_zarr_array(hvg_root[comb]).astype(np.float32)
        except Exception as e:
            print(f"  skip {comb}: {e}")
            continue
        if X.shape[0] == 0:
            continue

        parts = [
            model.get_latent_representation(X[i : i + batch_size])
            for i in range(0, X.shape[0], batch_size)
        ]
        emb = np.concatenate(parts, axis=0)
        write_zarr(root, comb, emb)
        n_total += emb.shape[0]

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return n_total


def extract_nlscvi(hvg_root, output_path, ckpt, device, batch_size):
    from sc_reconstruction.models.reconnlscvi import ReconNLSCVI

    model = object.__new__(ReconNLSCVI)
    model.load(ckpt, map_location=device)
    print(f"  n_latent={model.module.n_latent}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    root = zarr.open(output_path, mode="w")
    combs = sorted(hvg_root.group_keys())
    n_total = 0

    for comb in tqdm(combs, desc="nlscVI"):
        try:
            X = get_zarr_array(hvg_root[comb]).astype(np.float32)
        except Exception as e:
            print(f"  skip {comb}: {e}")
            continue
        if X.shape[0] == 0:
            continue

        parts = [
            model.get_latent_representation(X[i : i + batch_size])
            for i in range(0, X.shape[0], batch_size)
        ]
        emb = np.concatenate(parts, axis=0)
        write_zarr(root, comb, emb)
        n_total += emb.shape[0]

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return n_total


def extract_pca(hvg_root, output_path, pca_mean, pca_pc, **_):
    from sc_reconstruction.models.reconpca import ReconPCA

    n_components = int(os.path.basename(pca_pc).split("_")[1].split(".")[0])
    model = ReconPCA(n_components=n_components)
    model.load([pca_mean, pca_pc])
    print(f"  mean={model.mean.shape}  pc={model.pc.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    root = zarr.open(output_path, mode="w")
    combs = sorted(hvg_root.group_keys())
    n_total = 0

    for comb in tqdm(combs, desc="PCA"):
        try:
            X = get_zarr_array(hvg_root[comb]).astype(np.float32)
        except Exception as e:
            print(f"  skip {comb}: {e}")
            continue
        if X.shape[0] == 0:
            continue

        z = (X - model.mean) @ model.pc
        write_zarr(root, comb, z)
        n_total += z.shape[0]

    return n_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from e2e models")
    parser.add_argument("--model", required=True, choices=["AE", "nlscVI", "PCA"],
                        help="Which model to extract embeddings from")
    parser.add_argument("--out", required=True,
                        help="Output zarr path, e.g. .../pbmc_ae_128/AE_emb.zarr")
    parser.add_argument("--hvg_zarr", default=DEFAULT_HVG_ZARR,
                        help="Input HVG zarr with per-combination groups")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)

    # AE / nlscVI
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path (required for AE and nlscVI)")

    # PCA only
    parser.add_argument("--pca_mean", default=None,
                        help="Path to mean.zarr (required for PCA)")
    parser.add_argument("--pca_pc", default=None,
                        help="Path to pc_{dim}.zarr (required for PCA)")

    args = parser.parse_args()

    # Validate
    if args.model in ("AE", "nlscVI") and not args.ckpt:
        parser.error(f"--ckpt is required for --model {args.model}")
    if args.model == "PCA" and (not args.pca_mean or not args.pca_pc):
        parser.error("--pca_mean and --pca_pc are required for --model PCA")

    hvg_root = zarr.open(args.hvg_zarr, mode="r")
    n_groups = len(list(hvg_root.group_keys()))
    print(f"HVG zarr: {n_groups} combination groups  ({args.hvg_zarr})")
    print(f"Model: {args.model}  ->  {args.out}")

    if args.model == "AE":
        n = extract_ae(hvg_root, args.out, args.ckpt, args.device, args.batch_size)
    elif args.model == "nlscVI":
        n = extract_nlscvi(hvg_root, args.out, args.ckpt, args.device, args.batch_size)
    elif args.model == "PCA":
        n = extract_pca(hvg_root, args.out, args.pca_mean, args.pca_pc)

    print(f"\nDone: {n:,} embeddings written to {args.out}")


if __name__ == "__main__":
    main()
