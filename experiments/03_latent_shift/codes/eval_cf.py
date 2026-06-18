"""Evaluate CellFlow perturbation model on test/val (cell_type, donor, cytokine) combinations.

Same evaluator infrastructure as eval_st_pert.py — identical metrics, CSV format,
and worker pool — but replaces model loading and inference with CellFlow's predict() API.

All CellFlow predictions are pre-computed once via cf.predict(), then looked up
per combination during evaluation.

Usage:
  python eval_cf_pert.py \
      run_dir=/path/to/cf_run \
      [split=test] [ckpt=best] [device=cuda] \
      [test_fraction=1.0] [train_fraction=0.0] \
      [no_ctrl=false] [seed=42]
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
_CF_SRC = os.environ.get("CELLFLOW_SRC")
if _CF_SRC:
    sys.path.insert(0, _CF_SRC)

import gc
import glob
import json
import warnings
from os.path import join

import anndata as ad
import h5py
import hydra
import numpy as np
import pandas as pd
import torch
import zarr
from anndata.io import read_elem
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix

from cellflow.model import CellFlow
from sc_reconstruction.metrics.st_pert import (
    STPerturbationEvaluator,
    decode_gene,
    load_zarr_x,
)
from sc_reconstruction.utils.run_tools import sample_list

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)


# ── constants ─────────────────────────────────────────────────────────────────

DATA_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc_st"
WEIGHTS_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/weights/pbmc/split03"
MIN_CELLS = 10
CTRL_CAP = 2000

DECODER_CONFIGS = {
    "PCA_2048": {"type": "pca", "mean_path": join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"), "pc_path": join(WEIGHTS_ROOT, "PCA/Default/pc_2048.zarr")},
    "PCA_512":  {"type": "pca", "mean_path": join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"), "pc_path": join(WEIGHTS_ROOT, "PCA/Default/pc_512.zarr")},
    "PCA_128":  {"type": "pca", "mean_path": join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"), "pc_path": join(WEIGHTS_ROOT, "PCA/Default/pc_128.zarr")},
    "PCA_32":   {"type": "pca", "mean_path": join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"), "pc_path": join(WEIGHTS_ROOT, "PCA/Default/pc_32.zarr")},
    "PCA_10":   {"type": "pca", "mean_path": join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"), "pc_path": join(WEIGHTS_ROOT, "PCA/Default/pc_10.zarr")},
    "AE_2048":  {"type": "ae", "ckpt": join(WEIGHTS_ROOT, "AE/Default/400_[4096]_2048_20251009/epoch=116-val/loss_epoch=0.0205.ckpt")},
    "AE_512":   {"type": "ae", "ckpt": join(WEIGHTS_ROOT, "AE/Default/400_[4096, 4096, 4096, 4096]_512_20251009/epoch=146-val/loss_epoch=0.0578.ckpt")},
    "AE_128":   {"type": "ae", "ckpt": join(WEIGHTS_ROOT, "AE/Default/400_[4096, 4096, 4096]_128_20251009/epoch=253-val/loss_epoch=0.0904.ckpt")},
    "AE_32":    {"type": "ae", "ckpt": join(WEIGHTS_ROOT, "AE/Default/400_[4096, 4096, 4096, 4096]_32_20251009/epoch=182-val/loss_epoch=0.1167.ckpt")},
    "AE_10":    {"type": "ae", "ckpt": join(WEIGHTS_ROOT, "AE/Default/400_[4096, 4096, 4096, 4096]_10_20251009/epoch=73-val/loss_epoch=0.1317.ckpt")},
    "nlscVI_2048": {"type": "nlscvi", "ckpt": join(WEIGHTS_ROOT, "nlscVI/Default/400_4096_2048_1_0.1_20251016/2025-10-16_23-12-19_reconstruction_loss_validation/epoch=65-step=88902-reconstruction_loss_validation=-3203.762939453125.pt")},
    "nlscVI_512":  {"type": "nlscvi", "ckpt": join(WEIGHTS_ROOT, "nlscVI/Default/400_4096_512_3_0.1_20251017/2025-10-17_00-54-08_reconstruction_loss_validation/epoch=74-step=101025-reconstruction_loss_validation=-1493.1865234375.pt")},
    "nlscVI_128":  {"type": "nlscvi", "ckpt": join(WEIGHTS_ROOT, "nlscVI/Default/400_4096_128_4_0.1_20251017/2025-10-17_00-38-12_reconstruction_loss_validation/epoch=66-step=90249-reconstruction_loss_validation=-653.7234497070312.pt")},
    "nlscVI_32":   {"type": "nlscvi", "ckpt": join(WEIGHTS_ROOT, "nlscVI/Default/400_4096_32_4_0.1_20251017/2025-10-17_05-24-00_reconstruction_loss_validation/epoch=50-step=68697-reconstruction_loss_validation=-72.9731216430664.pt")},
    "nlscVI_10":   {"type": "nlscvi", "ckpt": join(WEIGHTS_ROOT, "nlscVI/Default/400_4096_10_4_0.1_20251016/2025-10-16_18-07-02_reconstruction_loss_validation/epoch=52-step=71391-reconstruction_loss_validation=169.5719451904297.pt")},
    "SE":         {"type": "mlp", "ckpt": "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/decoder/pbmc/hvg/SE/split03/MLP/500_3_4096_normal_none_20260319/epoch=117-val/loss_epoch=0.0882.ckpt"},
    "scGPT":      {"type": "mlp", "ckpt": "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/decoder/pbmc/hvg/scGPT/split03/MLP/500_2_4096_normal_none_20260318/epoch=68-val/loss_epoch=0.1337.ckpt"},
    "scConcept":  {"type": "mlp", "ckpt": "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/decoder/pbmc/hvg/scConcept/split03/MLP/500_2_4096_normal_none_20260320/epoch=105-val/loss_epoch=0.1326.ckpt"},
    "scimilarity": {"type": "mlp", "ckpt": "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/decoder/pbmc/hvg/scimilarity/split03/MLP/500_2_2048_normal_none_20260318/epoch=55-val/loss_epoch=0.1362.ckpt"},
}


# ── model loading ─────────────────────────────────────────────────────────────

def load_run_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(cfg_path) as f:
        return json.load(f)


def find_checkpoint(run_dir: str, ckpt_name: str = "best") -> str:
    """Find CellFlow .pkl checkpoint in run_dir."""
    pattern = os.path.join(run_dir, f"*_{ckpt_name}_CellFlow.pkl")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # Fallback: any CellFlow.pkl
    pattern = os.path.join(run_dir, "*_CellFlow.pkl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CellFlow.pkl found in {run_dir}")
    print(f"  Warning: '{ckpt_name}' ckpt not found, using {os.path.basename(matches[-1])}")
    return matches[-1]


def load_cf_model(run_dir: str, ckpt_name: str = "best"):
    """Load CellFlow model + read model_name from config.json."""
    ckpt_path = find_checkpoint(run_dir, ckpt_name)
    print(f"  Checkpoint: {ckpt_path}")
    cf = CellFlow.load(ckpt_path)
    if not cf._solver.is_trained:
        print("  WARNING: is_trained=False, forcing True")
        cf._solver.is_trained = True
    cfg = load_run_config(run_dir)
    model_name = cfg["model_name"]
    split_covs = list(cf._dm.split_covariates)
    print(f"  data_dim: {cf._data_dim}, split_covariates: {split_covs}")
    return cf, model_name


def load_decoder_module(model_name: str, device: str = "cpu"):
    """Load frozen decoder as a PyTorch module (latent -> gene)."""
    if model_name not in DECODER_CONFIGS:
        raise ValueError(f"No decoder config for {model_name}")
    cfg = DECODER_CONFIGS[model_name]
    dtype = cfg["type"]
    if dtype == "pca":
        from sc_reconstruction.adapters.e2e_decoder_adapters import FrozenPCADecoderAdapter
        decoder = FrozenPCADecoderAdapter(cfg["mean_path"], cfg["pc_path"])
    elif dtype == "ae":
        from sc_reconstruction.adapters.e2e_decoder_adapters import FrozenAEDecoderAdapter
        decoder = FrozenAEDecoderAdapter(cfg["ckpt"], map_location=device)
    elif dtype == "nlscvi":
        from sc_reconstruction.adapters.e2e_decoder_adapters import FrozenNLscVIDecoderAdapter
        decoder = FrozenNLscVIDecoderAdapter(cfg["ckpt"], map_location=device)
    elif dtype == "mlp":
        from sc_reconstruction.adapters.state_decoder_adapter import FrozenMLPDecoderAdapter
        decoder = FrozenMLPDecoderAdapter(cfg["ckpt"], map_location=device)
    else:
        raise ValueError(f"Unknown decoder type: {dtype}")
    decoder.eval()
    print(f"  Decoder ({dtype}): {decoder.n_input} -> {decoder.n_output}")
    return decoder


# ── CellFlow prediction ──────────────────────────────────────────────────────

def _comb_to_cf_key(comb):
    """Convert evaluator combo key 'ct-donor-cyto' to CF key 'donor_ct_cyto'."""
    parts = comb.rsplit("-", 2)
    if len(parts) == 3:
        ct, donor, cyto = parts
        return f"{donor}_{ct}_{cyto}"
    return None


def predict_all_conditions(cf, adata_test, obsm_key, comb_list=None):
    """Run cf.predict() for test conditions. Returns {cf_key: z_pred_np}.

    If comb_list is provided (evaluator-format: 'ct-donor-cyto'), only predict
    the matching conditions + their PBS controls.
    """
    split_covs = list(cf._dm.split_covariates)
    uses_ct_split = "cell_type" in split_covs

    rng = np.random.default_rng(42)
    adata_ctrl = adata_test[adata_test.obs["is_control"].to_numpy()]

    group_keys = ["donor", "cell_type"] if uses_ct_split else ["donor"]
    ctrl_parts = []
    for _, sub_idx in adata_ctrl.obs.groupby(group_keys).groups.items():
        sub = adata_ctrl[sub_idx]
        if sub.n_obs > CTRL_CAP:
            idx = rng.choice(sub.n_obs, CTRL_CAP, replace=False)
            ctrl_parts.append(sub[idx].copy())
        else:
            ctrl_parts.append(sub.copy())
    adata_ctrl = ad.concat(ctrl_parts)
    adata_ctrl.uns = adata_test.uns.copy()
    print(f"  Control: {adata_ctrl.n_obs} (cap={CTRL_CAP} per {group_keys})")

    if uses_ct_split:
        pert_obs = adata_test[~adata_test.obs["is_control"].to_numpy()].obs.copy()
        pert_obs["ct_condition"] = (
            pert_obs["donor"].astype(str) + "_"
            + pert_obs["cell_type"].astype(str) + "_"
            + pert_obs["target_gene"].astype(str)
        )
        ct_counts = pert_obs["ct_condition"].value_counts()
        valid = ct_counts[ct_counts >= MIN_CELLS].index
        pert_obs = pert_obs[pert_obs["ct_condition"].isin(valid)]
        covariate_pert = pert_obs.drop_duplicates(subset=["ct_condition"])
        condition_id_key = "ct_condition"

        ctrl_counts = adata_ctrl.obs.groupby(["donor", "cell_type"]).size()
        available = set((d, ct) for (d, ct), n in ctrl_counts.items() if n >= MIN_CELLS)
        keep = covariate_pert.apply(
            lambda r: (r["donor"], r["cell_type"]) in available, axis=1
        )
        if (~keep).sum() > 0:
            print(f"  Dropped {(~keep).sum()} conditions without matching controls")
        covariate_pert = covariate_pert[keep]
    else:
        covariate_pert = adata_test[
            ~adata_test.obs["is_control"].to_numpy()
        ].obs.drop_duplicates(subset=["condition"])
        condition_id_key = "condition"

    # Filter to only the combinations we need to evaluate
    if comb_list is not None:
        needed_cf_keys = set()
        for comb in comb_list:
            cf_key = _comb_to_cf_key(comb)
            if cf_key is not None:
                needed_cf_keys.add(cf_key)
            # Also need the PBS ctrl for this (donor, ct) for X_ctrl_pred
            parts = comb.rsplit("-", 2)
            if len(parts) == 3:
                ct, donor, _ = parts
                needed_cf_keys.add(f"{donor}_{ct}_PBS")
        n_before = len(covariate_pert)
        covariate_pert = covariate_pert[
            covariate_pert[condition_id_key].isin(needed_cf_keys)
        ]
        print(f"  Filtered to {len(covariate_pert)}/{n_before} conditions (from comb_list)")

    print(f"  Conditions to predict: {len(covariate_pert)}")
    preds = cf.predict(
        adata=adata_ctrl, sample_rep=obsm_key,
        condition_id_key=condition_id_key,
        covariate_data=covariate_pert,
    )
    preds = {k: np.asarray(v, dtype=np.float32) for k, v in preds.items()}
    sizes = [v.shape[0] if v.ndim > 1 else 1 for v in preds.values()]
    print(f"  Predicted {len(preds)} conditions, "
          f"cells/cond: min={min(sizes)}, max={max(sizes)}, median={np.median(sizes):.0f}")
    return preds


# ── embedding loader ──────────────────────────────────────────────────────────

def _read_h5_categorical(ds) -> np.ndarray:
    if "categories" in ds:
        cats = ds["categories"][:].astype(str)
        codes = ds["codes"][:]
        return cats[codes]
    return ds[:].astype(str)


def load_split_embeddings(model_name: str, split: str, h5ad_path: str):
    """Return (embeddings, emb_idx) from split h5ad (obsm only, no X loaded)."""
    embed_key = f"X_{model_name}"
    print(f"\n  Loading embeddings from {h5ad_path}")

    with h5py.File(h5ad_path, "r") as f:
        donors    = _read_h5_categorical(f["obs"]["donor"])
        cytokines = _read_h5_categorical(f["obs"]["target_gene"])
        cts       = _read_h5_categorical(f["obs"]["cell_type"])

        if embed_key not in f["obsm"]:
            avail = list(f["obsm"].keys())
            raise KeyError(f"'{embed_key}' not in obsm. Available: {avail}")

        print(
            f"  Loading obsm['{embed_key}'] "
            f"({len(donors):,} cells × {f['obsm'][embed_key].shape[1]} dims) ...",
            flush=True,
        )
        embeddings = f["obsm"][embed_key][:]

    combo_idx: dict = {}
    for i, (ct, d, cyto) in enumerate(zip(cts, donors, cytokines)):
        key = f"{ct}-{d}-{cyto}"
        combo_idx.setdefault(key, []).append(i)
    combo_idx = {k: np.asarray(v) for k, v in combo_idx.items()}
    print(f"  {len(combo_idx)} unique (cell_type, donor, cytokine) combinations in {split}")
    return embeddings, combo_idx


# ── CellFlow evaluator ───────────────────────────────────────────────────────

class _CellFlowModelStub:
    """Minimal stub satisfying STPerturbationEvaluator.__init__."""
    def __init__(self, decoder):
        self.gene_decoder = decoder
        self.cell_sentence_len = 512


class CellFlowPerturbationEvaluator(STPerturbationEvaluator):
    """STPerturbationEvaluator subclass using pre-computed CellFlow predictions.

    Overrides only _prepare_combination: instead of calling run_inference
    (model.predict_step), it looks up the pre-computed prediction dict.
    All metric computation, CSV writing, and worker pool logic are inherited.
    """

    def __init__(self, cf_predictions, decoder, embeddings, emb_idx, comb_list, **kwargs):
        self._cf_predictions = cf_predictions
        dummy_pert_map = {"PBS": torch.zeros(1)}
        model_stub = _CellFlowModelStub(decoder)
        super().__init__(
            model=model_stub,
            embeddings=embeddings,
            emb_idx=emb_idx,
            pert_map=dummy_pert_map,
            batch_oh_map={},
            comb_list=comb_list,
            **kwargs,
        )

    def _prepare_combination(self, comb: str):
        parts = comb.rsplit("-", 2)
        if len(parts) != 3:
            return None
        ct, donor, cyto = parts
        is_ctrl = cyto == "PBS"
        ctrl_key = f"{ct}-{donor}-PBS"

        # Pert cell count
        pert_idx = self.emb_idx.get(comb)
        if pert_idx is None or len(pert_idx) < self.min_cells:
            return None
        n_pert = len(pert_idx)
        if self.max_cells is not None and n_pert > self.max_cells:
            pert_idx = self.rng.choice(pert_idx, self.max_cells, replace=False)
            n_pert = self.max_cells

        # Ctrl embeddings (same donor, same cell type)
        ctrl_idx = self.emb_idx.get(ctrl_key, np.array([], dtype=int))
        if len(ctrl_idx) < self.min_cells:
            return None
        ctrl_sample = self.rng.choice(ctrl_idx, n_pert, replace=len(ctrl_idx) < n_pert)
        ctrl_embs = self.embeddings[ctrl_sample]

        # CellFlow prediction lookup: combo is ct-donor-cyto, CF key is donor_ct_cyto
        cf_key = f"{donor}_{ct}_{cyto}"
        z_pred_raw = self._cf_predictions.get(cf_key)

        if z_pred_raw is None:
            if is_ctrl:
                z_pred_raw = ctrl_embs  # identity for PBS
            else:
                return None

        z_pred = np.asarray(z_pred_raw, dtype=np.float32)
        if z_pred.ndim == 1:
            z_pred = z_pred[np.newaxis]

        # Match prediction count to n_pert
        if len(z_pred) > n_pert:
            z_pred = z_pred[self.rng.choice(len(z_pred), n_pert, replace=False)]
        elif len(z_pred) < n_pert:
            z_pred = z_pred[self.rng.choice(len(z_pred), n_pert, replace=True)]

        # Decode to gene space
        X_pred = decode_gene(self.decoder, z_pred, device=self.device)
        # ctrl_pred = decoded ctrl embeddings (identity baseline for delta-corrected DEG)
        X_ctrl_pred = decode_gene(self.decoder, ctrl_embs, device=self.device)

        # True gene expression from zarr
        X_true = load_zarr_x(self.root_zarr, comb)
        X_ctrl = load_zarr_x(self.root_zarr, ctrl_key)
        if X_ctrl is None:
            return None

        X_true_sub = X_true
        if X_true_sub is not None and self.max_cells is not None and len(X_true_sub) > self.max_cells:
            X_true_sub = X_true_sub[self.rng.choice(len(X_true_sub), self.max_cells, replace=False)]
        n_t = len(X_true_sub) if X_true_sub is not None else 0
        n_eval = n_t if n_t > 0 else n_pert

        X_ctrl_sub = X_ctrl[self.rng.choice(len(X_ctrl), n_eval, replace=len(X_ctrl) < n_eval)]
        n_c = len(X_ctrl_sub)

        # Log first combination
        if not self._inference_logged:
            self._inference_logged = True
            def _stats(name, arr):
                if arr is None:
                    print(f"  {name}: None")
                else:
                    pct_nz = float(np.count_nonzero(arr) / arr.size * 100)
                    print(
                        f"  {name}: shape={arr.shape}  "
                        f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
                        f"min={arr.min():.4f}  max={arr.max():.4f}  "
                        f"nonzero={pct_nz:.1f}%"
                    )
            print(f"\n[Inference sanity — first combo: {comb}]")
            print(f"  ctrl_embs   : shape={ctrl_embs.shape}  mean={ctrl_embs.mean():.4f}  std={ctrl_embs.std():.4f}")
            print(f"  z_pred      : shape={z_pred.shape}  mean={z_pred.mean():.4f}  std={z_pred.std():.4f}")
            _stats("X_pred     ", X_pred)
            _stats("X_ctrl_pred", X_ctrl_pred)
            _stats("X_true     ", X_true_sub)
            _stats("X_ctrl     ", X_ctrl_sub)
            print(f"  n_eval={n_eval}  n_c={n_c}  is_ctrl={is_ctrl}\n")

        return (
            comb, ctrl_key, ct, donor, cyto, is_ctrl,
            X_pred[:n_eval], X_ctrl_pred[:n_eval],
            X_true_sub[:n_eval] if X_true_sub is not None else None,
            X_ctrl_sub,
            n_eval, n_c,
        )


# ── Hydra entry point ─────────────────────────────────────────────────────────

@hydra.main(
    config_path="../configs",
    config_name="base_eval_cf",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    print(f'Measuring {cfg.metric.meta.name} for CellFlow perturbation model')
    print(f"Run dir : {cfg.run_dir}")
    print(f"Split   : {cfg.split}  |  Ckpt: {cfg.ckpt}  |  Device: {cfg.device}")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        print(f"  GPU not available — falling back to {device}")

    # ── Load CellFlow model ──────────────────────────────────────────────────
    print("\n=== Loading CellFlow model ===")
    cf, model_name = load_cf_model(cfg.run_dir, cfg.ckpt)
    obsm_key = f"X_{model_name}"

    # ── Load test data + embeddings for CellFlow prediction ──────────────────
    print("\n=== Loading test data for CellFlow prediction ===")
    h5ad_path = cfg.h5ad_paths[cfg.split]

    with h5py.File(h5ad_path, "r") as f:
        obs = read_elem(f["obs"])
        emb = np.asarray(read_elem(f["obsm"][obsm_key]), dtype=np.float32)
    n = len(obs)
    adata_test = ad.AnnData(
        X=csr_matrix((n, 100), dtype=np.float32), obs=obs, obsm={obsm_key: emb}
    )
    adata_test.obs["is_control"] = adata_test.obs["target_gene"] == "PBS"
    adata_test.obs["condition"] = (
        adata_test.obs["donor"].astype(str) + "_" + adata_test.obs["target_gene"].astype(str)
    )
    print(f"  {cfg.split}: {adata_test.shape}")

    # ESM2 + donor + cell_type embeddings
    from huggingface_hub import hf_hub_download

    hf_path = hf_hub_download(
        repo_id="theislab/cellflow-datasets",
        filename="pbmc_parse.h5ad",
        repo_type="dataset",
    )
    with h5py.File(hf_path, "r") as f:
        raw_esm = {k: f[f"uns/esm2_embeddings/{k}"][:] for k in f["uns/esm2_embeddings"].keys()}
    esm2 = {k.replace("-", "_"): v for k, v in raw_esm.items()}
    tg = adata_test.obs["target_gene"].unique().tolist()
    esm2 = {g: esm2[g] for g in tg if g != "PBS" and g in esm2}

    # Build one-hots from TRAIN for consistency with training
    train_path = join(DATA_ROOT, "train", "train.h5ad")
    print(f"  Loading train obs for one-hot encoding: {train_path}")
    with h5py.File(train_path, "r") as f:
        train_obs = read_elem(f["obs"])
    donors = sorted(train_obs["donor"].unique())
    donor_emb = {d: np.eye(len(donors), dtype=np.float32)[i] for i, d in enumerate(donors)}
    all_cts = sorted(train_obs["cell_type"].unique())
    ct_emb = {ct: np.eye(len(all_cts), dtype=np.float32)[i] for i, ct in enumerate(all_cts)}
    del train_obs

    adata_test.uns["esm2_embeddings"] = esm2
    adata_test.uns["donor_embeddings"] = donor_emb
    adata_test.uns["cell_type_embeddings"] = ct_emb

    # ── Load split combinations from zarr metadata ───────────────────────────
    print("\n=== Loading split metadata ===")
    split_zarr = zarr.open(cfg.split_zarr_path, mode="r")
    zarr_key = f"{cfg.split}_combinations"
    split_combs = list(split_zarr.attrs.get(zarr_key, []))
    train_combs = list(split_zarr.attrs.get("train_combinations", []))
    print(f"  Zarr key '{zarr_key}': {len(split_combs)} combinations")
    print(f"  Zarr key 'train_combinations': {len(train_combs)} combinations")
    if not split_combs:
        raise ValueError(
            f"No combinations found under '{zarr_key}' in {cfg.split_zarr_path}. "
            f"Available keys: {list(split_zarr.attrs.keys())}"
        )

    # Sample combinations BEFORE prediction so we only predict what we need
    rng = np.random.default_rng(cfg.seed)
    train_comb_list = sample_list(train_combs, cfg.train_fraction, rng) if cfg.train_fraction > 1e-4 else []
    test_comb_list = sample_list(split_combs, cfg.test_fraction, rng)
    comb_list = list(test_comb_list) + list(train_comb_list)

    # ── Load embeddings ──────────────────────────────────────────────────────
    print("\n=== Loading embeddings ===")
    embeddings, emb_idx = load_split_embeddings(model_name, cfg.split, h5ad_path)

    n_before = len(comb_list)
    comb_list = [c for c in comb_list if c in emb_idx]
    if cfg.no_ctrl:
        comb_list = [c for c in comb_list if not c.endswith("-PBS")]

    print(f"\n  Sampled test={cfg.test_fraction} train={cfg.train_fraction} seed={cfg.seed}: {n_before} total")
    print(f"  Found in h5ad        : {len(comb_list)}")
    print(
        f"  Evaluating {len(comb_list)} combinations "
        f"({'excluding' if cfg.no_ctrl else 'including'} PBS control)"
    )
    print(f"  First 5: {comb_list[:5]}")

    # ── Pre-compute CellFlow predictions (only for sampled combinations) ────
    print("\n=== Pre-computing CellFlow predictions ===")
    cf_predictions = predict_all_conditions(cf, adata_test, obsm_key, comb_list=comb_list)
    del cf
    gc.collect()

    # ── Load decoder (PyTorch, after freeing JAX model) ──────────────────────
    print("\n=== Loading decoder ===")
    decoder = load_decoder_module(model_name, device=device)
    decoder.to(device)

    # ── Instantiate evaluator ────────────────────────────────────────────────
    run_name = os.path.basename(os.path.normpath(cfg.run_dir))
    save_dir = os.path.join(cfg.output.results_root, run_name, cfg.split, cfg.ckpt)
    file_name = cfg.output.file_name
    print(f"\n  Run name : {run_name}")
    print(f"  Save dir : {save_dir}")
    print(f"  File name: {file_name}")

    # Extract evaluator kwargs from metric config (reuses st_pert.yaml)
    eval_cfg = OmegaConf.to_container(cfg.metric.evaluator, resolve=True)
    eval_cfg.pop("_target_", None)
    eval_cfg.pop("_partial_", None)

    evaluator = CellFlowPerturbationEvaluator(
        cf_predictions=cf_predictions,
        decoder=decoder,
        embeddings=embeddings,
        emb_idx=emb_idx,
        comb_list=comb_list,
        device=device,
        seed=cfg.seed,
        **eval_cfg,
    )
    evaluator.run(save_dir, file_name)
    print(f"\nSuccessfully saved {cfg.metric.meta.name} results under: {save_dir}/")


if __name__ == "__main__":
    main()
