"""Evaluate ST perturbation model on test/val (cell_type, donor, cytokine) combinations.

Entry script following the biological_pbmc.py / Hydra pattern:
  - Config is driven by base_eval_st.yaml + metric/st_pert.yaml
  - Model is loaded here (requires train_st.inject_frozen_decoder)
  - Evaluator is instantiated via Hydra and called with runtime args
  - Results are saved to {results_root}/{run_name}/{split}/{ckpt}/{metric}/sampled_*.csv

Usage:
  python eval_st_pert.py \\
      run_dir=/path/to/run \\
      [split=test] [ckpt=last] [device=cuda] \\
      [test_fraction=1.0] [train_fraction=0.0] \\
      [no_ctrl=false] [seed=42]
"""

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import pickle
import warnings
import yaml
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import zarr
from hydra.utils import instantiate
from omegaconf import DictConfig

from sc_reconstruction.utils.run_tools import sample_list


# ── model loading ─────────────────────────────────────────────────────────────

def load_run_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    with open(cfg_path) as f:
        return yaml.unsafe_load(f)


def _parse_step(p: Path) -> int:
    """Extract step number from checkpoint filename like step=step=128000-val_loss=...ckpt"""
    import re
    m = re.search(r"step=(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def _parse_val_loss(p: Path) -> float:
    """Extract val_loss from checkpoint filename."""
    try:
        return float(p.stem.split("val_loss=")[-1])
    except ValueError:
        return float("inf")


def find_checkpoint(run_dir: str, ckpt_name: str = "best", min_step: int = 0) -> str:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if ckpt_name in ("best", "best_latent"):
        # pick the step=*-val_loss=*.ckpt with the lowest val_loss in its filename
        candidates = [p for p in Path(ckpt_dir).glob("step=*val_loss=*.ckpt")
                      if "last" not in p.name and "final" not in p.name]
        if min_step > 0:
            candidates = [p for p in candidates if _parse_step(p) >= min_step]
        if candidates:
            best = min(candidates, key=_parse_val_loss)
            print(f"  Best checkpoint (min_step={min_step}): {best.name}")
            return str(best)
        print("  Warning: no val_loss checkpoints found, falling back to last.ckpt")
        ckpt_name = "last"
    named = {
        "last":  os.path.join(ckpt_dir, "last.ckpt"),
        "final": os.path.join(ckpt_dir, "final.ckpt"),
    }
    if ckpt_name in named and os.path.exists(named[ckpt_name]):
        return named[ckpt_name]
    ckpts = sorted(Path(ckpt_dir).glob("*.ckpt"), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    print(f"  Warning: '{ckpt_name}.ckpt' not found, using {ckpts[-1].name}")
    return str(ckpts[-1])


def load_model(run_dir: str, ckpt_name: str = "last", device: str = "cuda", min_step: int = 0):
    """Load ST model from checkpoint + inject frozen decoder per config.yaml."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    from train_st import inject_frozen_decoder

    cfg = load_run_config(run_dir)
    model_name   = cfg["model_name"]
    decoder_mode = cfg.get("decoder_mode", "frozen")

    ckpt_path = find_checkpoint(run_dir, ckpt_name, min_step=min_step)
    print(f"  Checkpoint: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp  = raw["hyper_parameters"]
    sd  = raw["state_dict"]

    decoder_weight = hp.get("decoder_weight", hp.get("decoder_loss_weight", 0.0))

    model = StateTransitionPerturbationModel(
        input_dim=hp["input_dim"],
        gene_dim=hp.get("gene_dim", hp["input_dim"]),
        hvg_dim=hp.get("hvg_dim", 0),
        output_dim=hp["output_dim"],
        pert_dim=hp["pert_dim"],
        batch_dim=hp.get("batch_dim"),
        basal_mapping_strategy=hp.get("basal_mapping_strategy", "batch"),
        cell_set_len=hp["cell_set_len"],
        hidden_dim=hp["hidden_dim"],
        n_encoder_layers=hp["n_encoder_layers"],
        n_decoder_layers=hp["n_decoder_layers"],
        use_basal_projection=hp.get("use_basal_projection", True),
        transformer_backbone_key=hp.get("transformer_backbone_key", "llama"),
        transformer_backbone_kwargs=hp.get("transformer_backbone_kwargs", {}),
        output_space=hp.get("output_space", "gene"),
        embed_key=hp.get("embed_key", ""),
        gene_names=hp.get("gene_names", []),
        batch_size=hp.get("batch_size", 16),
        control_pert=hp.get("control_pert", "PBS"),
        lr=hp.get("lr", 1e-4),
        max_steps=hp.get("max_steps", 40000),
        train_seed=hp.get("train_seed", 42),
        weight_decay=hp.get("weight_decay", 0.0),
        wandb_track=False,
        gradient_clip_val=hp.get("gradient_clip_val", 10),
        blur=hp.get("blur", 0.05),
        loss=hp.get("loss", "energy"),
        distributional_loss=hp.get("distributional_loss", "energy"),
        predict_residual=hp.get("predict_residual", True),
        batch_encoder=hp.get("batch_encoder", True),
        use_batch_token=hp.get("use_batch_token", False),
        decoder_weight=decoder_weight,
        decoder_cfg=hp.get("decoder_cfg"),
        confidence_token=hp.get("confidence_token", False),
        freeze_pert_backbone=hp.get("freeze_pert_backbone", False),
        finetune_vci_decoder=hp.get("finetune_vci_decoder", False),
        residual_decoder=hp.get("residual_decoder", False),
        nb_decoder=hp.get("nb_decoder", False),
        mask_attn=hp.get("mask_attn", False),
        regularization=hp.get("regularization", 0.0),
        init_from=None,
    )

    if decoder_mode == "frozen":
        inject_frozen_decoder(
            model, model_name, device="cpu", decoder_weight=decoder_weight
        )
        model._decoder_externally_configured = True

    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_total:,} params ({n_trainable:,} trainable)")
    print(f"  Decoder: {type(model.gene_decoder).__name__ if model.gene_decoder else 'None'}")
    return model, model_name


# ── embedding loader ──────────────────────────────────────────────────────────

def _read_h5_categorical(ds) -> np.ndarray:
    if "categories" in ds:
        cats  = ds["categories"][:].astype(str)
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


# ── Hydra entry point ─────────────────────────────────────────────────────────

@hydra.main(
    config_path="../configs",
    config_name="base_eval_st",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    print(f'Measuring {cfg.metric.meta.name} for ST perturbation model')
    print(f"Run dir : {cfg.run_dir}")
    print(f"Split   : {cfg.split}  |  Ckpt: {cfg.ckpt}  |  Device: {cfg.device}")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        print(f"  GPU not available — falling back to {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n=== Loading model ===")
    min_step = cfg.get("min_step", 0)
    model, model_name = load_model(cfg.run_dir, cfg.ckpt, device, min_step=min_step)

    # ── Load perturbation one-hot map ─────────────────────────────────────────
    pert_map = torch.load(
        os.path.join(cfg.run_dir, "pert_onehot_map.pt"),
        map_location="cpu",
        weights_only=False,
    )
    print(f"  Perturbation map: {len(pert_map)} entries")

    with open(os.path.join(cfg.run_dir, "batch_onehot_map.pkl"), "rb") as fh:
        batch_oh_map = pickle.load(fh)
    print(f"  Batch (donor) map: {len(batch_oh_map)} entries: {list(batch_oh_map.keys())}")

    # ── Load split combinations from zarr metadata ────────────────────────────
    # The zarr uses "test_combinations" / "val_combinations" / "train_combinations"
    # keys, giving the curated split. The h5ad may contain extra combos (e.g.
    # additional control groups) that are not part of the official split.
    print("\n=== Loading split metadata ===")
    split_zarr  = zarr.open(cfg.split_zarr_path, mode="r")
    zarr_key    = f"{cfg.split}_combinations"
    split_combs = list(split_zarr.attrs.get(zarr_key, []))
    train_combs = list(split_zarr.attrs.get("train_combinations", []))
    print(f"  Zarr key '{zarr_key}': {len(split_combs)} combinations")
    print(f"  Zarr key 'train_combinations': {len(train_combs)} combinations")
    if not split_combs:
        raise ValueError(
            f"No combinations found under '{zarr_key}' in {cfg.split_zarr_path}. "
            f"Available keys: {list(split_zarr.attrs.keys())}"
        )

    # Mirror biological.py exactly: train sampled first from shared rng, then test,
    # combined as test + train — this ensures identical combination sets at same seed.
    rng = np.random.default_rng(cfg.seed)
    train_comb_list = sample_list(train_combs, cfg.train_fraction, rng) if cfg.train_fraction > 1e-4 else []
    test_comb_list  = sample_list(split_combs, cfg.test_fraction, rng)
    comb_list = list(test_comb_list) + list(train_comb_list)

    # ── Load embeddings ───────────────────────────────────────────────────────
    print("\n=== Loading embeddings ===")
    embeddings, emb_idx = load_split_embeddings(
        model_name, cfg.split, cfg.h5ad_paths[cfg.split]
    )

    # Keep only combos that are present in the h5ad (embeddings must exist)
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

    # ── Instantiate evaluator (Hydra partial pattern) ─────────────────────────
    run_name  = os.path.basename(os.path.normpath(cfg.run_dir))
    save_dir  = os.path.join(cfg.output.results_root, run_name, cfg.split, cfg.ckpt)
    file_name = cfg.output.file_name
    print(f"\n  Run name : {run_name}")
    print(f"  Save dir : {save_dir}")
    print(f"  File name: {file_name}")

    evaluator_factory = instantiate(cfg.metric.evaluator, _partial_=True)
    evaluator = evaluator_factory(
        model=model,
        embeddings=embeddings,
        emb_idx=emb_idx,
        pert_map=pert_map,
        batch_oh_map=batch_oh_map,
        comb_list=comb_list,
        device=device,
        seed=cfg.seed,  # used only for per-combination cell subsampling RNG
    )
    evaluator.run(save_dir, file_name)
    print(f"\nSuccessfully saved {cfg.metric.meta.name} results under: {save_dir}/")


if __name__ == "__main__":
    main()