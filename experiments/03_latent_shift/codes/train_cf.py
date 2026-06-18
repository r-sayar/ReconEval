"""Train CellFlow on PBMC with configurable embedding + cell_type covariate.

Uses split_covariates=["donor", "cell_type"] so OT only pairs
control↔perturbed within the same (donor, cell_type).
Saves best model (by test e-distance) and last model.

Usage:
    python train_cf.py --model PCA_128
    python train_cf.py --model AE_128
    python train_cf.py --model AE_512 --num_iters 1000000
"""

import argparse
import gc
import json
import os
import sys
import traceback
from os.path import join

import functools
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
import flax.linen as nn
import optax
from anndata.io import read_elem
from scipy.sparse import csr_matrix

CELLFLOW_SRC = os.environ.get('CELLFLOW_SRC')
if CELLFLOW_SRC: sys.path.insert(0, CELLFLOW_SRC)
import cellflow
from cellflow.model import CellFlow
from cellflow.utils import match_linear
from cellflow.training._callbacks import ComputationCallback


# ---------------------------------------------------------------------------
# Embedding registry (same as train_st.py)
# ---------------------------------------------------------------------------
EMB_DIMS = {
    "SE": 2058, "scGPT": 512, "scConcept": 512, "scimilarity": 128,
    "PCA_2048": 2048, "AE_2048": 2048, "nlscVI_2048": 2048,
    "AE_512": 512, "nlscVI_512": 512, "PCA_512": 512,
    "AE_128": 128, "nlscVI_128": 128, "PCA_128": 128,
    "AE_32": 32,   "nlscVI_32": 32,   "PCA_32": 32,
    "AE_10": 10,   "nlscVI_10": 10,   "PCA_10": 10,
}

DATA_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc_st"
OUT_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/cf_ct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_split(split: str, obsm_key: str) -> ad.AnnData:
    path = join(DATA_ROOT, split, f"{split}.h5ad")
    print(f"Loading {path} ...")
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        emb = np.asarray(read_elem(f["obsm"][obsm_key]), dtype=np.float32)
    n_cells = len(obs)
    X_pseudo = csr_matrix((n_cells, 100), dtype=np.float32)
    adata = ad.AnnData(X=X_pseudo, obs=obs, obsm={obsm_key: emb})
    adata.obs["is_control"] = adata.obs["target_gene"] == "PBS"
    adata.obs["condition"] = adata.obs["donor"].astype(str) + "_" + adata.obs["target_gene"].astype(str)
    print(f"  {split}: {adata.shape}, obsm['{obsm_key}'] = {emb.shape}")
    print(f"  conditions: {adata.obs['condition'].nunique()} unique")
    return adata


def subsample_conditions(adata, n_per_cond, seed=42):
    rng = np.random.default_rng(seed)
    parts = []
    for cond in adata.obs["condition"].unique():
        sub = adata[adata.obs["condition"] == cond]
        if sub.n_obs > n_per_cond:
            idx = rng.choice(sub.n_obs, n_per_cond, replace=False)
            parts.append(sub[idx].copy())
        else:
            parts.append(sub.copy())
    return ad.concat(parts)


def filter_missing_controls(adata_val, label):
    """Remove perturbed cells whose (donor, cell_type) has no matching control."""
    ctrl = adata_val[adata_val.obs["is_control"].to_numpy()]
    pert = adata_val[~adata_val.obs["is_control"].to_numpy()]
    ctrl_dct = set(zip(ctrl.obs["donor"], ctrl.obs["cell_type"]))
    pert_dct = set(zip(pert.obs["donor"], pert.obs["cell_type"]))
    missing = pert_dct - ctrl_dct
    if missing:
        print(f"  {label}: removing {len(missing)} (donor,cell_type) with no control: {missing}")
        keep = ~pert.obs.apply(lambda r: (r["donor"], r["cell_type"]) in missing, axis=1).values
        adata_val = ad.concat([ctrl, pert[keep]])
        print(f"  {label} after filter: {adata_val.shape}")
    return adata_val


def save_training_curves(logs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if "train_e_distance_mean" in logs and len(logs["train_e_distance_mean"]) > 0:
        axes[0].plot(range(len(logs["train_e_distance_mean"])), logs["train_e_distance_mean"], 'b-', label='train')
    if "test_e_distance_mean" in logs and len(logs["test_e_distance_mean"]) > 0:
        axes[0].plot(range(len(logs["test_e_distance_mean"])), logs["test_e_distance_mean"], 'r-', label='test')
    axes[0].set_xlabel('Validation step'); axes[0].set_ylabel('Energy distance')
    axes[0].set_title('Energy distance'); axes[0].legend(); axes[0].grid(True)

    for key, color in [("train_r_squared_mean", 'b'), ("test_r_squared_mean", 'r')]:
        if key in logs and len(logs[key]) > 0:
            axes[1].plot(range(len(logs[key])), logs[key], f'{color}-', label=key.split("_")[0])
    axes[1].set_xlabel('Validation step'); axes[1].set_ylabel('R-squared')
    axes[1].set_title('R-squared'); axes[1].legend(); axes[1].grid(True)

    if "loss" in logs and len(logs["loss"]) > 0:
        loss = logs["loss"]
        step = max(1, len(loss) // 500)
        axes[2].plot(range(len(loss[::step])), loss[::step], 'g-', alpha=0.7)
        axes[2].set_xlabel(f'Iteration (x{step})'); axes[2].set_ylabel('Loss')
        axes[2].set_title('Training loss'); axes[2].grid(True)

    plt.tight_layout()
    fig.savefig(join(out_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training_curves.png")


class BestModelCheckpoint(ComputationCallback):
    """Save best model (lowest test e-distance) and last model during training."""

    def __init__(self, cf_model, out_dir, prefix, monitor="test_e_distance_mean"):
        self.cf = cf_model
        self.out_dir = out_dir
        self.prefix = prefix
        self.monitor = monitor
        self.best_val = float("inf")
        self.best_iter = -1

    def on_train_begin(self, *args, **kwargs):
        pass

    def on_log_iteration(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        logs = self.cf.trainer.training_logs
        if self.monitor in logs and len(logs[self.monitor]) > 0:
            val = logs[self.monitor][-1]
            iteration = len(logs["loss"])
            print(f"  [ckpt] iter={iteration}, {self.monitor}={val:.4f} (best={self.best_val:.4f})")
            if val < self.best_val:
                self.best_val = val
                self.best_iter = iteration
                # Mark as trained so the checkpoint is usable for prediction
                self.cf._solver.is_trained = True
                self.cf.save(self.out_dir, file_prefix=f"{self.prefix}_best", overwrite=True)
                print(f"  [ckpt] New best model saved (iter={iteration}, {self.monitor}={val:.4f})")
        return {}

    def on_train_end(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        self.cf.save(self.out_dir, file_prefix=f"{self.prefix}_last", overwrite=True)
        print(f"  [ckpt] Last model saved")
        print(f"  [ckpt] Best was at iter={self.best_iter}, {self.monitor}={self.best_val:.4f}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(EMB_DIMS.keys()),
                        help="Embedding name (e.g. PCA_128, AE_128, AE_512)")
    parser.add_argument("--num_iters", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--valid_freq", type=int, default=50_000)
    parser.add_argument("--config", type=str, default="repro", choices=["repro", "paper"],
                        help="Hyperparameter config: 'repro' (reproducibility code) or 'paper' (paper defaults)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir (default: <out_root>/<model>_ct_s<seed>)")
    parser.add_argument("--out_root", type=str, default=OUT_ROOT,
                        help=f"Output root (default: {OUT_ROOT})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (used for data subsample + model init RNG)")
    args = parser.parse_args()

    model_name = args.model
    obsm_key = f"X_{model_name}"
    n_dims = EMB_DIMS[model_name]
    # Path: keep legacy `{model}_ct/` only when (seed=42 AND default root); otherwise
    # include the training seed in the dir so new seeds never collide with the frozen runs.
    legacy = (args.seed == 42 and args.out_root == OUT_ROOT)
    suffix = "" if legacy else f"_s{args.seed}"
    out_dir = args.out_dir or join(args.out_root, f"{model_name}_ct{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"CellFlow training: {model_name} (dim={n_dims})")
    print(f"Output: {out_dir}")
    print(f"{'=' * 60}")

    # --- 1. Load data ---
    adata_train = load_split("train", obsm_key)

    # --- 2. ESM2 + donor one-hot + cell_type one-hot ---
    from huggingface_hub import hf_hub_download
    hf_path = hf_hub_download(repo_id="theislab/cellflow-datasets", filename="pbmc_parse.h5ad", repo_type="dataset")
    with h5py.File(hf_path, "r") as f:
        raw_esm = {k: f[f"uns/esm2_embeddings/{k}"][:] for k in f["uns/esm2_embeddings"].keys()}
    esm2 = {k.replace("-", "_"): v for k, v in raw_esm.items()}
    target_genes = adata_train.obs["target_gene"].unique().tolist()
    esm2 = {g: esm2[g] for g in target_genes if g != "PBS" and g in esm2}

    all_donors = sorted(adata_train.obs["donor"].unique())
    donor_emb = {d: np.eye(len(all_donors), dtype=np.float32)[i] for i, d in enumerate(all_donors)}

    all_cell_types = sorted(adata_train.obs["cell_type"].unique())
    ct_emb = {ct: np.eye(len(all_cell_types), dtype=np.float32)[i] for i, ct in enumerate(all_cell_types)}

    adata_train.uns["esm2_embeddings"] = esm2
    adata_train.uns["donor_embeddings"] = donor_emb
    adata_train.uns["cell_type_embeddings"] = ct_emb
    print(f"ESM2: {len(esm2)} cytokines, Donors: {len(all_donors)}, Cell types: {len(all_cell_types)}")

    # --- 3. Prepare CellFlow ---
    cf = CellFlow(adata_train, solver="otfm")
    cf.prepare_data(
        sample_rep=obsm_key, control_key="is_control",
        perturbation_covariates={"cytokine_treatment": ["target_gene"]},
        perturbation_covariate_reps={"cytokine_treatment": "esm2_embeddings"},
        sample_covariates=["donor", "cell_type"],
        sample_covariate_reps={
            "donor": "donor_embeddings",
            "cell_type": "cell_type_embeddings",
        },
        split_covariates=["donor", "cell_type"],
        max_combination_length=1, null_value=0.0,
    )
    print(f"Data dim: {cf._data_dim}")

    # --- 4. Validation data ---
    N_VAL = 200
    adata_train_val = subsample_conditions(adata_train, N_VAL, seed=args.seed)
    print(f"Train val: {adata_train_val.shape}")

    adata_test = load_split("val", obsm_key)
    adata_test.uns["esm2_embeddings"] = esm2
    adata_test.uns["donor_embeddings"] = donor_emb
    adata_test.uns["cell_type_embeddings"] = ct_emb
    adata_test_val = subsample_conditions(adata_test, N_VAL, seed=args.seed)
    print(f"Test val: {adata_test_val.shape}")
    del adata_test; gc.collect()

    adata_train_val = filter_missing_controls(adata_train_val, "Train val")
    adata_test_val = filter_missing_controls(adata_test_val, "Test val")

    adata_train_val.uns = adata_train.uns.copy()
    adata_test_val.uns = adata_train.uns.copy()
    cf.prepare_validation_data(adata_train_val, name="train",
                               n_conditions_on_log_iteration=10, n_conditions_on_train_end=10)
    cf.prepare_validation_data(adata_test_val, name="test",
                               n_conditions_on_log_iteration=20, n_conditions_on_train_end=None)

    # --- 5. Prepare model ---
    if args.config == "paper":
        # Paper defaults (Methods Section 1.6)
        hp = dict(
            hidden_dims=[4096, 4096, 4096],
            time_encoder_dims=[2048, 2048, 2048],
            constant_noise=0.1,
            match_epsilon=1.0,
            cond_output_dropout=0.9,
        )
    else:
        # Reproducibility code values (fig_2/runs_cellflow/conf/model/pbmc_new_cytokine.yaml)
        hp = dict(
            hidden_dims=[2048, 2048, 2048],
            time_encoder_dims=[1024, 1024, 1024],
            constant_noise=0.5,
            match_epsilon=0.5,
            cond_output_dropout=0.5,
        )
    print(f"  Config: {args.config} -> hidden_dims={hp['hidden_dims']}, "
          f"noise={hp['constant_noise']}, epsilon={hp['match_epsilon']}, "
          f"cond_dropout={hp['cond_output_dropout']}")

    match_fn = functools.partial(match_linear, epsilon=hp["match_epsilon"], tau_a=1.0, tau_b=1.0)
    cf.prepare_model(
        condition_mode="deterministic", regularization=0.0,
        pooling="attention_token", pooling_kwargs={},
        layers_before_pool={
            "cytokine_treatment": {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
            "donor": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
            "cell_type": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
        },
        layers_after_pool={"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.0},
        condition_embedding_dim=256, cond_output_dropout=hp["cond_output_dropout"],
        condition_encoder_kwargs={}, pool_sample_covariates=True,
        time_freqs=1024,
        time_encoder_dims=hp["time_encoder_dims"], time_encoder_dropout=0.0,
        hidden_dims=hp["hidden_dims"], hidden_dropout=0.0,
        conditioning="concatenation",
        decoder_dims=[4096, 4096, 4096],
        vf_act_fn=nn.silu, vf_kwargs=None,
        probability_path={"constant_noise": hp["constant_noise"]},
        match_fn=match_fn,
        optimizer=optax.MultiSteps(optax.adam(5e-5), 20),
        solver_kwargs={},
        layer_norm_before_concatenation=False,
        linear_projection_before_concatenation=False,
        seed=args.seed,
    )
    print(f"Model ready: {model_name}, split_covariates=['donor','cell_type'], seed={args.seed}")

    # --- 6. Train ---
    prefix = f"{model_name}_ct{suffix}"
    ckpt_cb = BestModelCheckpoint(cf, out_dir, prefix=prefix, monitor="test_e_distance_mean")
    callbacks = [cellflow.training.Metrics(["r_squared", "e_distance"]), ckpt_cb]
    print(f"Training: {args.num_iters} iters, batch={args.batch_size}, valid_freq={args.valid_freq}")
    cf.train(num_iterations=args.num_iters, batch_size=args.batch_size,
             callbacks=callbacks, valid_freq=args.valid_freq)

    # --- 7. Save training curves + logs ---
    logs = cf.trainer.training_logs
    save_training_curves(logs, out_dir)

    logs_serializable = {}
    for k, v in logs.items():
        if isinstance(v, list) and len(v) > 0:
            logs_serializable[k] = [float(x) if isinstance(x, (int, float, np.floating)) else x for x in v]
    with open(join(out_dir, "training_logs.json"), "w") as f:
        json.dump(logs_serializable, f, indent=2)

    config = {
        "model_name": model_name,
        "config": args.config,
        "seed": args.seed,
        "hyperparameters": hp,
        "obsm_key": obsm_key,
        "n_dims": n_dims,
        "num_iterations": args.num_iters,
        "batch_size": args.batch_size,
        "valid_freq": args.valid_freq,
        "split_covariates": ["donor", "cell_type"],
        "best_iter": ckpt_cb.best_iter,
        "best_test_ed": float(ckpt_cb.best_val),
    }
    with open(join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for key in sorted(logs.keys()):
        if key != "loss" and len(logs[key]) > 0:
            v = logs[key]
            print(f"  {key}: {v[0]:.4f} -> {v[-1]:.4f}")

    print(f"\nDone. Best: {join(out_dir, f'{prefix}_best_CellFlow.pkl')}")
    print(f"      Last: {join(out_dir, f'{prefix}_last_CellFlow.pkl')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
