from pathlib import Path
"""Custom State Transition training with frozen or fresh decoder.

Imports the ST model directly, sets up the cell_load data module, and
trains with Lightning. Two decoder modes are supported:

  --decoder_mode frozen  (default)
      Replaces State's internal LatentToGeneDecoder with a pre-trained
      frozen adapter (MLP / PCA / AE / nlscVI). The adapter's parameters
      are frozen; when decoder_weight > 0, gene-space loss gradients still
      flow back through the frozen ops into the backbone (detach_decoder=False).

  --decoder_mode fresh
      Keeps State's internally-built LatentToGeneDecoder trained jointly
      end-to-end with the backbone. Matches the paper's ST+SE dual-loss:
        E(output_dim) → 1024 → 1024 → 512 → G(gene_dim)
      where E is the predicted embedding (model output, e.g. 2058 for SE),
      NOT the backbone hidden state h(384). This is f_decode in Section 4.3.5:
      MMD_gene loss = 0.1 × energy_distance(f_decode(ê), f_decode(e_true)).

Usage:
    python train_st.py --model SE
    python train_st.py --model SE --decoder_mode fresh
    python train_st.py --model scGPT --max_steps 40000
    python train_st.py --model AE
    python train_st.py --model PCA --no_wandb
"""

import argparse
import logging
import os
import pickle
import sys
from os.path import join

import yaml

import lightning.pytorch as pl
import torch
from cell_load.utils.modules import get_datamodule
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

from state.tx.models.state_transition import StateTransitionPerturbationModel

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")

_HERE        = Path(__file__).resolve().parent
_CFG_DIR     = _HERE.parent / "configs"
ARCH_CFG_DIR = str(_CFG_DIR / "arch")
DATA_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc_st"
OUT_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/st_paper_latentsize"

EMB_DIMS = {
    "SE": 2058, "scGPT": 512, "scConcept": 512, "scimilarity": 128,
    "PCA_2048": 2048, "AE_2048": 2048, "nlscVI_2048": 2048,
    "AE_512": 512, "nlscVI_512": 512, "PCA_512": 512,
    "AE_128": 128, "nlscVI_128": 128, "PCA_128": 128,
    "AE_32": 32,   "nlscVI_32": 32,   "PCA_32": 32,
    "AE_10": 10,   "nlscVI_10": 10,   "PCA_10": 10,
}

HVG_GENE_DIM = 5412
WEIGHTS_ROOT = "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/weights/pbmc/split03"

DECODER_CONFIGS = {
    "SE": {
        "type": "mlp",
        "ckpt": "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/decoder/pbmc/hvg/SE/split03/MLP/500_3_4096_normal_none_20260319/epoch=117-val/loss_epoch=0.0882.ckpt",
    },
    "scGPT": {
        "type": "mlp",
        "ckpt": "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/decoder/pbmc/hvg/scGPT/split03/MLP/500_2_4096_normal_none_20260318/epoch=68-val/loss_epoch=0.1337.ckpt",
    },
    "scConcept": {
        "type": "mlp",
        "ckpt": "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/decoder/pbmc/hvg/scConcept/split03/MLP/500_2_4096_normal_none_20260320/epoch=105-val/loss_epoch=0.1326.ckpt",
    },
    "scimilarity": {
        "type": "mlp",
        "ckpt": "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/decoder/pbmc/hvg/scimilarity/split03/MLP/500_2_2048_normal_none_20260318/epoch=55-val/loss_epoch=0.1362.ckpt",
    },
    "PCA_2048": {
        "type": "pca",
        "mean_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"),
        "pc_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/pc_2048.zarr"),
    },
    "AE_2048": {
        "type": "ae",
        "ckpt": (
            WEIGHTS_ROOT
            + "/AE/Default/400_[4096]_2048_20251009"
            + "/epoch=116-val/loss_epoch=0.0205.ckpt"
        ),
    },
    "nlscVI_2048": {
        "type": "nlscvi",
        "ckpt": (
            WEIGHTS_ROOT
            + "/nlscVI/Default/400_4096_2048_1_0.1_20251016"
            + "/2025-10-16_23-12-19_reconstruction_loss_validation"
            + "/epoch=65-step=88902-reconstruction_loss_validation=-3203.762939453125.pt"
        ),
    },
    "AE_128": {
        "type": "ae",
        "ckpt": (
            WEIGHTS_ROOT
            + "/AE/Default/400_[4096, 4096, 4096]_128_20251009"
            + "/epoch=253-val/loss_epoch=0.0904.ckpt"
        ),
    },
    "nlscVI_128": {
        "type": "nlscvi",
        "ckpt": (
            WEIGHTS_ROOT
            + "/nlscVI/Default/400_4096_128_4_0.1_20251017"
            + "/2025-10-17_00-38-12_reconstruction_loss_validation"
            + "/epoch=66-step=90249-reconstruction_loss_validation=-653.7234497070312.pt"
        ),
    },
    "PCA_128": {
        "type": "pca",
        "mean_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"),
        "pc_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/pc_128.zarr"),
    },
    "AE_512": {
        "type": "ae",
        "ckpt": (
            WEIGHTS_ROOT
            + "/AE/Default/400_[4096, 4096, 4096, 4096]_512_20251009"
            + "/epoch=146-val/loss_epoch=0.0578.ckpt"
        ),
    },
    "nlscVI_512": {
        "type": "nlscvi",
        "ckpt": (
            WEIGHTS_ROOT
            + "/nlscVI/Default/400_4096_512_3_0.1_20251017"
            + "/2025-10-17_00-54-08_reconstruction_loss_validation"
            + "/epoch=74-step=101025-reconstruction_loss_validation=-1493.1865234375.pt"
        ),
    },
    "AE_32": {
        "type": "ae",
        "ckpt": (
            WEIGHTS_ROOT
            + "/AE/Default/400_[4096, 4096, 4096, 4096]_32_20251009"
            + "/epoch=182-val/loss_epoch=0.1167.ckpt"
        ),
    },
    "nlscVI_32": {
        "type": "nlscvi",
        "ckpt": (
            WEIGHTS_ROOT
            + "/nlscVI/Default/400_4096_32_4_0.1_20251017"
            + "/2025-10-17_05-24-00_reconstruction_loss_validation"
            + "/epoch=50-step=68697-reconstruction_loss_validation=-72.9731216430664.pt"
        ),
    },
    "PCA_32": {
        "type": "pca",
        "mean_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"),
        "pc_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/pc_32.zarr"),
    },
    "AE_10": {
        "type": "ae",
        "ckpt": (
            WEIGHTS_ROOT
            + "/AE/Default/400_[4096, 4096, 4096, 4096]_10_20251009"
            + "/epoch=73-val/loss_epoch=0.1317.ckpt"
        ),
    },
    "nlscVI_10": {
        "type": "nlscvi",
        "ckpt": (
            WEIGHTS_ROOT
            + "/nlscVI/Default/400_4096_10_4_0.1_20251016"
            + "/2025-10-16_18-07-02_reconstruction_loss_validation"
            + "/epoch=52-step=71391-reconstruction_loss_validation=169.5719451904297.pt"
        ),
    },
    "PCA_10": {
        "type": "pca",
        "mean_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"),
        "pc_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/pc_10.zarr"),
    },
    "PCA_512": {
        "type": "pca",
        "mean_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/mean.zarr"),
        "pc_path": os.path.join(WEIGHTS_ROOT, "PCA/Default/pc_512.zarr"),
    },
}


DATA_KWARGS_COMMON = dict(
    # output_space is set dynamically from the arch YAML in build_data_module
    pert_rep="onehot",
    basal_rep="sample",
    num_workers=12,
    pin_memory=True,
    use_consecutive_loading=True,
    n_basal_samples=1,
    basal_mapping_strategy="batch",
    should_yield_control_cells=True,
    batch_col="donor",
    pert_col="target_gene",
    cell_type_key="cell_type",
    control_pert="PBS",
    map_controls=True,
    perturbation_features_file=None,
    store_raw_basal=False,
    int_counts=False,
    barcode=True,
)


def load_arch_cfg(name_or_path: str) -> dict:
    """Load arch YAML by name (e.g. 'hf_se_parse') or by absolute/relative path."""
    if os.path.isfile(name_or_path):
        path = name_or_path
    else:
        path = join(ARCH_CFG_DIR, f"{name_or_path}.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded arch config: {path}")
    return cfg


def build_data_module(model_name, batch_size, cell_set_len, output_space="gene"):
    """Create the PerturbationDataModule from the unified train h5ad."""
    toml_path = str(_CFG_DIR / "st" / "pbmc_train.toml")
    data_kwargs = {
        **DATA_KWARGS_COMMON,
        "output_space": output_space,
        "toml_config_path": toml_path,
        "embed_key": f"X_{model_name}",
    }

    dm = get_datamodule(
        "PerturbationDataModule",
        data_kwargs,
        batch_size=batch_size,
        cell_sentence_len=cell_set_len,
    )
    return dm, data_kwargs


def build_val_data_module(model_name, batch_size, cell_set_len, output_space="gene"):
    """Create a separate PerturbationDataModule for validation data."""
    toml_path = str(_CFG_DIR / "st" / "pbmc_val.toml")
    data_kwargs = {
        **DATA_KWARGS_COMMON,
        "output_space": output_space,
        "toml_config_path": toml_path,
        "embed_key": f"X_{model_name}",
    }

    dm = get_datamodule(
        "PerturbationDataModule",
        data_kwargs,
        batch_size=batch_size,
        cell_sentence_len=cell_set_len,
    )
    return dm


def build_model(var_dims, data_kwargs, training_cfg, model_name, arch_cfg,
                decoder_weight=0.1, decoder_mode="frozen", cell_set_len=None):
    """Create the StateTransitionPerturbationModel from an arch config dict."""
    gene_dim = var_dims.get("hvg_dim", var_dims.get("gene_dim", HVG_GENE_DIM))

    # Resolve cell_set_len: explicit arg (from CLI) > arch YAML > 512
    if cell_set_len is None:
        cell_set_len = arch_cfg.get("cell_set_len", 512)

    bb_kwargs = dict(arch_cfg["transformer_backbone_kwargs"])
    # max_position_embeddings must equal cell_set_len — set dynamically
    bb_kwargs["max_position_embeddings"] = cell_set_len
    if bb_kwargs.get("head_dim") is None:
        bb_kwargs.pop("head_dim", None)  # let backbone derive it from hidden/heads

    def _a(key, default=None):
        """Read from arch YAML, fall back to default."""
        return arch_cfg.get(key, default)

    model_kwargs = dict(
        # ── required arch params ──────────────────────────────────────────────
        hidden_dim=arch_cfg["hidden_dim"],
        n_encoder_layers=arch_cfg["n_encoder_layers"],
        n_decoder_layers=arch_cfg["n_decoder_layers"],
        transformer_backbone_key=arch_cfg["transformer_backbone_key"],
        transformer_backbone_kwargs=bb_kwargs,
        decoder_hidden_dims=_a("decoder_hidden_dims", [1024, 512]),
        # ── optional arch params (YAML overrides defaults) ────────────────────
        cell_set_len=cell_set_len,
        blur=_a("blur", 0.05),
        loss=_a("loss", "energy"),
        distributional_loss=_a("distributional_loss", "energy"),
        predict_residual=_a("predict_residual", True),
        use_basal_projection=_a("use_basal_projection", True),
        batch_encoder=_a("batch_encoder", False),
        use_batch_token=_a("use_batch_token", False),
        confidence_token=_a("confidence_token", False),
        mask_attn=_a("mask_attn", False),
        lora=_a("lora", None),
        freeze_pert_backbone=_a("freeze_pert_backbone", False),
        finetune_vci_decoder=_a("finetune_vci_decoder", False),
        residual_decoder=_a("residual_decoder", False),
        nb_decoder=_a("nb_decoder", False),
        regularization=_a("regularization", 0.0),
        init_from=_a("init_from", None),
        # ── fixed ─────────────────────────────────────────────────────────────
        extra_tokens=0,
        decoder_weight=decoder_weight,
        # ── training / data keys expected by the model constructor ────────────
        embed_key=data_kwargs["embed_key"],
        output_space=_a("output_space", data_kwargs["output_space"]),
        gene_names=var_dims.get("gene_names", []),
        batch_size=training_cfg["batch_size"],
        control_pert=data_kwargs["control_pert"],
        lr=training_cfg["lr"],
        max_steps=training_cfg["max_steps"],
        train_seed=training_cfg["train_seed"],
        weight_decay=training_cfg["weight_decay"],
        wandb_track=training_cfg["wandb_track"],
        gradient_clip_val=training_cfg["gradient_clip_val"],
    )

    # decoder_cfg for State's internal LatentToGeneDecoder.
    # frozen mode: this is a placeholder — replaced by inject_frozen_decoder().
    # fresh mode: use dims from arch YAML (decoder_hidden_dims).
    arch_decoder_dims = arch_cfg.get("decoder_hidden_dims", [1024, 1024, 512])
    decoder_hidden_dims = arch_decoder_dims if decoder_mode == "fresh" else [1024, 512]
    decoder_cfg = dict(
        latent_dim=var_dims["output_dim"],
        gene_dim=gene_dim,
        hidden_dims=decoder_hidden_dims,
        dropout=0.1,
        residual_decoder=False,
    )
    model_kwargs["decoder_cfg"] = decoder_cfg

    model = StateTransitionPerturbationModel(
        input_dim=var_dims["input_dim"],
        gene_dim=gene_dim,
        hvg_dim=var_dims.get("hvg_dim", HVG_GENE_DIM),
        output_dim=var_dims["output_dim"],
        pert_dim=var_dims["pert_dim"],
        batch_dim=var_dims.get("batch_dim", None),
        basal_mapping_strategy=data_kwargs["basal_mapping_strategy"],
        **model_kwargs,
    )
    return model


def inject_frozen_decoder(model, model_name, device="cpu", decoder_ckpt=None, decoder_weight=0.1):
    """Replace model.gene_decoder with the appropriate frozen decoder."""
    cfg = DECODER_CONFIGS[model_name]
    dtype = cfg["type"]

    if dtype == "mlp":
        from sc_reconstruction.adapters.state_decoder_adapter import FrozenMLPDecoderAdapter
        ckpt = decoder_ckpt or cfg["ckpt"]
        print(f"Loading frozen MLP decoder from {ckpt}")
        frozen = FrozenMLPDecoderAdapter(ckpt, map_location=device)
    elif dtype == "pca":
        from sc_reconstruction.adapters.e2e_decoder_adapters import FrozenPCADecoderAdapter
        print(f"Loading frozen PCA decoder")
        frozen = FrozenPCADecoderAdapter(cfg["mean_path"], cfg["pc_path"])
    elif dtype == "ae":
        from sc_reconstruction.adapters.e2e_decoder_adapters import FrozenAEDecoderAdapter
        ckpt = decoder_ckpt or cfg["ckpt"]
        print(f"Loading frozen AE decoder from {ckpt}")
        frozen = FrozenAEDecoderAdapter(ckpt, map_location=device)
    elif dtype == "nlscvi":
        from sc_reconstruction.adapters.e2e_decoder_adapters import FrozenNLscVIDecoderAdapter
        ckpt = decoder_ckpt or cfg["ckpt"]
        print(f"Loading frozen nlscVI decoder from {ckpt}")
        frozen = FrozenNLscVIDecoderAdapter(ckpt, map_location=device)
    else:
        raise ValueError(f"Unknown decoder type: {dtype}")

    print(f"  Frozen decoder: {frozen.n_input} -> {frozen.n_output}")
    assert frozen.n_input == model.output_dim, (
        f"Decoder input dim {frozen.n_input} != model output_dim {model.output_dim}"
    )

    model.gene_decoder = frozen
    # detach_decoder=False when decoder_weight > 0: gene-space loss gradients flow
    # back through the frozen decoder ops into the backbone (params stay frozen, but
    # autograd chain rule propagates through the linear layers). This matches the
    # paper's 0.1-weighted gene-expression loss that keeps latent predictions on the
    # decoded manifold. With decoder_weight=0.0 we detach to skip the computation.
    model.detach_decoder = decoder_weight == 0.0

    n_frozen = sum(1 for p in model.gene_decoder.parameters() if not p.requires_grad)
    n_total = sum(1 for p in model.gene_decoder.parameters())
    print(f"  Decoder params: {n_total} total, {n_frozen} frozen")


def main():
    parser = argparse.ArgumentParser(description="Train ST with frozen or fresh decoder")
    all_models = sorted(EMB_DIMS.keys())
    parser.add_argument("--model", required=True, choices=all_models)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cell_set_len", type=int, default=None,
                        help="Override arch YAML cell_set_len. "
                             "If not set, uses arch YAML value (or 512 if missing).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--decoder_mode", choices=["frozen", "fresh"], default="frozen",
                        help="frozen (default): replace internal decoder with pre-trained frozen "
                             "adapter (MLP/PCA/AE/nlscVI). "
                             "fresh: keep State's internally-built LatentToGeneDecoder "
                             "E(output_dim)→1024→1024→512→G and train jointly — "
                             "matches the paper's ST+SE dual-loss (Section 4.3.5).")
    parser.add_argument("--decoder_ckpt", type=str, default=None,
                        help="Override default HVG decoder checkpoint path (frozen mode only)")
    parser.add_argument("--decoder_weight", type=float, default=0.0,
                        help="Weight for gene-space decoder loss. "
                             "0.0 (default): pure latent energy distance — clean benchmark objective, "
                             "decoder used only at eval time. "
                             "0.1 = paper default (only meaningful with a jointly-trained fresh decoder).")
    parser.add_argument("--scheduler", action="store_true", default=False,
                        help="Enable linear warmup LR schedule (5%% of max_steps, min 500 steps), "
                             "then fixed LR for the rest of training. "
                             "Use for small-latent models (dim<=32) to prevent NaN at startup. "
                             "Default: fixed LR throughout (matches paper ST training).")
    parser.add_argument("--arch_config", type=str, default="hf_se_parse",
                        help="Model architecture: name in runs/configs/st/arch/ (without .yaml) "
                             "or an absolute path to a YAML file. "
                             "Built-in options: 'hf_se_parse' (hidden=384, matches HF checkpoint), "
                             "'paper_parse_pbmc' (hidden=1440, paper Table 3).")
    parser.add_argument("--out_root", type=str, default=OUT_ROOT,
                        help=f"Output root (default: {OUT_ROOT}). "
                             f"Use a parallel root for extra training seeds.")
    args = parser.parse_args()

    model_name = args.model
    arch_cfg = load_arch_cfg(args.arch_config)
    arch_tag = os.path.splitext(os.path.basename(args.arch_config))[0]
    # CLI --cell_set_len overrides arch YAML when explicitly provided (not None)
    if args.cell_set_len is not None:
        cell_set_len = args.cell_set_len
    else:
        cell_set_len = arch_cfg.get("cell_set_len", 512)
    # Keep legacy run name only for (seed=42 AND default root); otherwise append _s{seed}
    legacy = (args.seed == 42 and args.out_root == OUT_ROOT)
    seed_suffix = "" if legacy else f"_s{args.seed}"
    run_name = (f"{model_name}_pbmc_split03_{args.decoder_mode}_dw{args.decoder_weight}"
                f"_{arch_tag}_cs{cell_set_len}{seed_suffix}")
    run_dir = join(args.out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    training_cfg = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        train_seed=args.seed,
        weight_decay=0.0005,
        wandb_track=not args.no_wandb,
        gradient_clip_val=10,
        val_freq=2000,
        limit_val_batches=100,
        ckpt_every_n_steps=4000,
    )

    pl.seed_everything(args.seed)

    # 1. Data module (train + val)
    print(f"\n{'=' * 60}")
    print(f"Setting up data for {model_name}")
    print(f"{'=' * 60}")
    output_space = arch_cfg.get("output_space", "gene")
    print(f"  cell_set_len: {cell_set_len} (arch_cfg: {arch_cfg.get('cell_set_len', 'not set')}, CLI: {args.cell_set_len})")
    dm, data_kwargs = build_data_module(model_name, args.batch_size, cell_set_len, output_space)
    dm.setup(stage="fit")

    dm_val = build_val_data_module(model_name, args.batch_size, cell_set_len, output_space)
    dm_val.setup(stage="fit")
    print("  Validation data module ready")

    var_dims = dm.get_var_dims()
    print(f"  var_dims: input_dim={var_dims['input_dim']}, output_dim={var_dims['output_dim']}, "
          f"pert_dim={var_dims['pert_dim']}, gene_dim={var_dims.get('gene_dim', '?')}")

    # Save mappings
    torch.save(dm.pert_onehot_map, join(run_dir, "pert_onehot_map.pt"))
    with open(join(run_dir, "batch_onehot_map.pkl"), "wb") as f:
        pickle.dump(dm.batch_onehot_map, f)
    with open(join(run_dir, "cell_type_onehot_map.pkl"), "wb") as f:
        pickle.dump(dm.cell_type_onehot_map, f)
    with open(join(run_dir, "var_dims.pkl"), "wb") as f:
        pickle.dump(var_dims, f)

    # 2. Model
    print(f"\n{'=' * 60}")
    print(f"Building ST model (pertsets GPT2)")
    print(f"{'=' * 60}")
    model = build_model(var_dims, data_kwargs, training_cfg, model_name, arch_cfg,
                        args.decoder_weight, args.decoder_mode,
                        cell_set_len=cell_set_len)

    # 3. Decoder setup
    if args.decoder_mode == "frozen":
        inject_frozen_decoder(model, model_name, decoder_ckpt=args.decoder_ckpt,
                              decoder_weight=args.decoder_weight)
        # Prevent on_load_checkpoint from overwriting the frozen decoder with a fresh
        # LatentToGeneDecoder when resuming from a checkpoint (STATE's hook checks this flag).
        model._decoder_externally_configured = True
    else:
        # fresh: State's internally-built LatentToGeneDecoder E→1024→1024→512→G is trained
        # jointly end-to-end with the backbone — paper's ST+SE dual-loss (Section 4.3.5).
        print(f"Using fresh jointly-trained decoder (paper's ST+SE approach)")
        model.detach_decoder = args.decoder_weight == 0.0

    # base.py's configure_optimizers uses plain Adam and ignores weight_decay.
    # Override via types.MethodType so Lightning picks it up correctly.
    import types
    _wd    = training_cfg["weight_decay"]
    _lr    = training_cfg["lr"]
    _use_scheduler = args.scheduler
    _warmup = max(500, training_cfg["max_steps"] // 20)   # 5% warmup, min 500 steps
    def _configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=_lr,
            weight_decay=_wd,
        )
        if not _use_scheduler:
            return optimizer
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=_warmup,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": warmup_sched, "interval": "step", "frequency": 1},
        }
    model.configure_optimizers = types.MethodType(_configure_optimizers, model)
    if _use_scheduler:
        print(f"  LR scheduler: warmup({_warmup} steps), then fixed {_lr}")
    else:
        print("  LR scheduler: none (fixed LR)")

    param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Total params: {param_gb:.2f} GB, trainable: {trainable:,}, frozen: {frozen:,}")

    # 4. Callbacks
    ckpt_dir = join(run_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="step={step}-val_loss={val_loss:.4f}",
            save_last="link",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        ),
    ]

    # 5. Loggers
    loggers = []
    if training_cfg["wandb_track"]:
        try:
            from lightning.pytorch.loggers import WandbLogger
            # detect existing run ID from wandb/run-{datetime}-{id} directory
            wandb_id = None
            wandb_dir = join(run_dir, "wandb")
            if os.path.isdir(wandb_dir):
                import glob
                run_dirs = sorted(glob.glob(join(wandb_dir, "run-*")))
                if run_dirs:
                    wandb_id = run_dirs[-1].split("-")[-1]
                    print(f"  Resuming W&B run: {wandb_id}")
            wl = WandbLogger(
                project="st_perturbation",
                name=run_name,
                save_dir=run_dir,
                id=wandb_id,
                resume="allow",
            )
            loggers.append(wl)
        except Exception as e:
            print(f"  wandb setup failed: {e}, continuing without wandb")

    # 6. Trainer
    print(f"\n{'=' * 60}")
    print(f"Starting training: {args.max_steps} steps")
    print(f"{'=' * 60}")

    val_loader = dm_val.train_dataloader()

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=args.max_steps,
        check_val_every_n_epoch=None,
        val_check_interval=training_cfg["val_freq"],
        limit_val_batches=training_cfg["limit_val_batches"],
        logger=loggers if loggers else None,
        callbacks=callbacks,
        gradient_clip_val=training_cfg["gradient_clip_val"],
        use_distributed_sampler=False,
    )

    # Resume from last checkpoint if exists
    last_ckpt = join(ckpt_dir, "last.ckpt")
    ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
    if ckpt_path:
        print(f"  Resuming from {ckpt_path}")

    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    # Save final checkpoint
    final_ckpt = join(ckpt_dir, "final.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"\n  Saved final checkpoint: {final_ckpt}")

    # Save config for inference
    import yaml
    if args.decoder_mode == "frozen":
        decoder_info = args.decoder_ckpt or DECODER_CONFIGS[model_name]
    else:
        decoder_info = {"type": "fresh", "latent_dim": var_dims["output_dim"],
                        "hidden_dims": [1024, 1024, 512], "dropout": 0.1}
    config = {
        "model_name": model_name,
        "decoder_mode": args.decoder_mode,
        "decoder": decoder_info,
        "training": training_cfg,
        "data": data_kwargs,
        "var_dims": {k: v for k, v in var_dims.items() if k != "gene_names"},
    }
    with open(join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nTraining complete for {model_name}!")


if __name__ == "__main__":
    main()
