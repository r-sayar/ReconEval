#!/usr/bin/env python
"""
Regenerate metric-summary CSVs in notebooks/figs/results/.

  pbmc, luca   : exclude_cytokine = False   (cytokine column written if available)
  tahoe        : exclude_cytokine = True    (cytokine columns excluded)

Run from any cwd. Re-uses summary_handler / merge_model_metrics / clear_metrics from
notebooks/inspectresults/utils.py to keep behaviour identical to _fig2.ipynb except
for the cytokine flag.

Usage:
    python regenerate_summary_csvs.py [DATASET ...]   # default: pbmc luca tahoe
    python regenerate_summary_csvs.py luca            # only luca
"""
import os
import sys
import glob
import zarr
import pandas as pd
from .utils import (
    get_target_dict,
    merge_model_metrics,
    clear_metrics,
    parse_arch_ae,
    parse_arch_scvi,
    parse_arch_pca,
)

REPO = "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction"
RESULTS_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/results"
OUT_DIR = f"../frozen/figs/results"
PATHS_DIR = f"{REPO}/runs/paths"
DATA_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction"

LATENTS = [10, 32, 128, 512, 2048]
SPLITS = ["split01", "split02", "split03"]

DATASET_MODELS = {
    "pbmc":  ["AE", "olAE", "mlAE", "nlscVI", "scVI", "mlscVI"],
    "luca":  ["AE", "olAE", "mlAE", "nlscVI", "scVI", "mlscVI"],
    "tahoe": ["AE", "olAE", "mlAE", "nlscVI", "scVI", "mlscVI"],
}

# tahoe = exclude cytokine; pbmc, luca = include
EXCLUDE_CYTOKINE = {"pbmc": False, "luca": False, "tahoe": True}


def arch_parser(model_class, model_name):
    if "AE" in model_class:
        return parse_arch_ae(model_name)
    if "scVI" in model_class:
        return parse_arch_scvi(model_name)
    if "PCA" in model_class:
        return parse_arch_pca(model_name)
    raise ValueError(f"Unrecognized model class: {model_class}")


def summary_handler(model_list, zarr_root, split, exclude_cytokine):
    """Same logic as _fig2.ipynb's summary_handler but with configurable cytokine flag."""
    root_zarr = zarr.open(f"{zarr_root}/{split}/split_metadata.zarr", mode="r")
    test_comb = root_zarr.attrs["test_combinations"][:]
    key_fields_dict, keep_method_dict, set_negative_metrics = get_target_dict(
        exclude_cytokine=exclude_cytokine
    )

    rows = {}
    for model_metric_path in model_list:
        if not os.path.isdir(model_metric_path):
            print(f"  [skip] not a dir: {model_metric_path}")
            continue
        model_name = model_metric_path.split("/")[-1]
        model_class = model_metric_path.split("/")[-2]
        merged = merge_model_metrics(model_metric_path, key_fields_dict, keep_method_dict)
        if merged is None:
            print(f"  [skip] no metrics merged: {model_metric_path}")
            continue
        cleaned = clear_metrics(merged, test_comb, set_negative_metrics)
        cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
        arch = arch_parser(model_class, model_name)
        cleaned["latentsize"] = arch["latentsize"]
        cleaned["layers"] = arch["layers"]
        cleaned["hidden"] = arch["hidden"]
        rows[model_class] = cleaned

    return pd.DataFrame(rows).T


def build_model_list(dataset, latent, split):
    """Return list of model result dirs for a (dataset, latent, split)."""
    paths = []
    # Non-PCA models from best_*_paths.txt
    for cls in DATASET_MODELS[dataset]:
        pf = f"{PATHS_DIR}/{dataset}/best_{cls}_{latent}_paths.txt"
        if not os.path.exists(pf):
            continue
        with open(pf) as fh:
            for line in fh.read().splitlines():
                if not line:
                    continue
                # weight-file path -> derive model dir under results/{dataset}_data/{split}/sampled/{cls}/{model_name}
                # path layout (split is dir #10, model_name is dir #13 in the canonical 0-indexed split)
                parts = line.split("/")
                # find split & model_name positions
                try:
                    s_idx = next(i for i, p in enumerate(parts) if p in SPLITS)
                except StopIteration:
                    continue
                if parts[s_idx] != split:
                    continue
                model_name = parts[s_idx + 3]   # …/{split}/{cls}/Default/{model_name}/...
                model_dir = f"{RESULTS_ROOT}/{dataset}_data/{split}/sampled/{cls}/{model_name}"
                paths.append(model_dir)

    # PCA — direct
    pca_dir = f"{RESULTS_ROOT}/{dataset}_data/{split}/sampled/PCA/pc_{latent}"
    if os.path.isdir(pca_dir):
        paths.append(pca_dir)
    return paths


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["pbmc", "luca", "tahoe"]
    for ds in datasets:
        if ds not in DATASET_MODELS:
            print(f"[warn] unknown dataset: {ds}")
            continue
        zarr_root = f"{DATA_ROOT}/{'lucav2' if ds == 'luca' else ds}"
        excl = EXCLUDE_CYTOKINE[ds]
        print(f"\n=== {ds}  (exclude_cytokine={excl}) ===")
        for latent in LATENTS:
            for split in SPLITS:
                model_list = build_model_list(ds, latent, split)
                if not model_list:
                    print(f"  [skip] {ds} latent={latent} {split}: no models")
                    continue
                print(f"  {ds} latent={latent} {split}: {len(model_list)} models")
                df = summary_handler(model_list, zarr_root, split, exclude_cytokine=excl)
                out_csv = f"{OUT_DIR}/{ds}_{split}_metrics_summary_latent{latent}.csv"
                df.to_csv(out_csv)
                cyto_cols = [c for c in df.columns if "cytokine" in c]
                tag = f"  cytokine_cols={cyto_cols}" if cyto_cols else "  (no cytokine cols)"
                print(f"    -> {out_csv}{tag}")


if __name__ == "__main__":
    main()