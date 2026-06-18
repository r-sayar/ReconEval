import math
import os
import warnings
from collections.abc import Iterable

import lightning as L
import numpy as np
import torch
import torch.nn as nn

import argparse
from pathlib import Path

import anndata as ad
from torch.utils.data import DataLoader, Dataset, random_split
from sc_reconstruction.decoders._base_decoder import BaseReconstructionDecoder
from concept.decoder.decoder_model import TransformerDecoderModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment


class ReconTransformerModule(L.LightningModule):
    """Lightning wrapper that adapts TransformerDecoderModel to the ReconMLPDecoder API.

    - ``forward(z, *cat_list) -> (B, n_output)``
    - ``training_step`` expects batch dict with keys like ReconMLPDecoder:
      ``batch['x']``, ``batch['z']``, optional ``batch['batch_onehot']``.
    - Automatically constructs gene_indices ``(0..n_output-1)`` and expands to batch.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] | None = None,
        dim_model: int = 128,
        num_head: int = 8,
        dim_hid: int = 256,
        nlayers: int = 6,
        dropout: float = 0.1,
        distribution: str = "normal",
        learning_rate: float = 1e-4,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 5,
        lr_threshold: float = 1e-3,
        lr_min: float = 0.0,
        library_size_mode: str = "none",
        # optional optimizer knobs
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_input = n_input
        self.n_output = n_output
        self.n_cat_list = n_cat_list
        self.distribution = distribution
        self.learning_rate = learning_rate
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_threshold = lr_threshold
        self.lr_min = lr_min
        self.library_size_mode = library_size_mode
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.wandb_run_id = None
        self.resume_from_checkpoint = None
        total_input_dim = n_input
        if n_cat_list is not None:
            total_input_dim += sum(n_cat_list)

        self.transformer = TransformerDecoderModel(
            num_genes=n_output,
            cell_emb_dim=total_input_dim,
            dim_model=dim_model,
            num_head=num_head,
            dim_hid=dim_hid,
            nlayers=nlayers,
            dropout=dropout,
            lr=learning_rate,            
            weight_decay=weight_decay,  
        )

        self.register_buffer("gene_index_base", torch.arange(n_output, dtype=torch.long), persistent=True)

        if self.distribution == "normal_mle_gene":
            initial = torch.zeros(n_output).normal_(mean=0.0, std=0.1)
            self.px_r = nn.Parameter(initial)

    def _expand_gene_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        base = self.gene_index_base.to(device=device)
        return base.unsqueeze(0).expand(batch_size, -1)

    def forward(self, z: torch.Tensor, *cat_list: torch.Tensor) -> torch.Tensor:
        # Match ReconMLPDecoder: concatenate conditional one-hots if present
        if cat_list and len(cat_list) > 0:
            z_cat = torch.cat([z] + list(cat_list), dim=1)
        else:
            z_cat = z

        if self.library_size_mode != "none":
            warnings.warn(
                "library_size_mode is set but not implemented for decoding only. "
                "Ignoring library size mode.",
                stacklevel=2,
            )

        gene_indices = self._expand_gene_indices(z_cat.shape[0], z_cat.device)
        preds = self.transformer(z_cat, gene_indices)  # (B, n_output)
        return preds

    def _shared_step(self, batch, stage: str = "train"):
        x = batch["x"]
        z = batch["z"]

        if "batch_onehot" in batch and batch["batch_onehot"] is not None:
            reconstruction = self.forward(z, batch["batch_onehot"])
        else:
            reconstruction = self.forward(z)

        loss = self.compute_loss(x, reconstruction)
        self.log_metrics({"loss": loss}, stage)
        return loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        if opt is not None:
            lr = opt.param_groups[0].get("lr", self.learning_rate)
            self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        if self.distribution == "normal":
            return nn.MSELoss()(reconstruction, x)
        elif self.distribution == "huber":
            return nn.HuberLoss()(reconstruction, x)
        elif self.distribution == "l1":
            return nn.L1Loss()(reconstruction, x)
        elif self.distribution == "normal_mle_fixed":
            sigma = torch.tensor(1e-1, device=x.device)
            dist = torch.distributions.Normal(loc=reconstruction, scale=sigma)
            log_prob = dist.log_prob(x)
            return (-log_prob.sum(dim=1)).mean()
        elif self.distribution == "normal_mle_gene":
            sigma = torch.exp(torch.clamp(self.px_r, min=-10.0, max=10.0))  # (n_output,)
            sigma_batch = sigma.unsqueeze(0)  # (1, n_output)
            dist = torch.distributions.Normal(loc=reconstruction, scale=sigma_batch)
            log_prob = dist.log_prob(x)
            return (-log_prob.sum(dim=1)).mean()
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def configure_optimizers(self):
        if self.use_adamw:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        if self.reduce_lr_on_plateau:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.lr_patience,
                    factor=self.lr_factor,
                    threshold=self.lr_threshold,
                    min_lr=self.lr_min,
                    threshold_mode="abs",
                ),
                "monitor": "val/loss_epoch",
                "interval": "epoch",
                "frequency": 1,
                "name": "lr_scheduler",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    def log_metrics(self, loss_dict, stage: str = "train"):
        for key, value in loss_dict.items():
            self.log(f"{stage}/{key}", value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        """Save W&B run ID when checkpoint is saved."""
        super().on_save_checkpoint(checkpoint)
        
        # Try to get W&B run ID from logger
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                if hasattr(logger, 'experiment') and logger.experiment is not None:
                    self.wandb_run_id = logger.experiment.id
                    break
        
        if self.wandb_run_id:
            checkpoint['wandb_run_id'] = self.wandb_run_id
            checkpoint['wandb_name'] = getattr(logger, 'name', None)
            checkpoint['wandb_project'] = getattr(logger.experiment, 'project', None) if hasattr(logger, 'experiment') and logger.experiment else None

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        
        self.wandb_run_id = checkpoint.get('wandb_run_id')
        self.wandb_name = checkpoint.get('wandb_name')
        self.wandb_project = checkpoint.get('wandb_project')

class ReconTransformerDecoder(BaseReconstructionDecoder):
    """Recon-style wrapper mirroring :class:`ReconMLPDecoder`.

    - ``self.module`` is a :class:`~lightning.pytorch.LightningModule`.
    - ``train(datamodule, **trainer_kwargs)`` — fit on a datamodule.
    - ``decode(z_numpy, *cat_numpy) -> numpy`` — invert latent to expression.
    - ``save / load`` — checkpoint round-trip.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] | None = None,
        # transformer config
        dim_model: int = 128,
        num_head: int = 8,
        dim_hid: int = 256,
        nlayers: int = 6,
        dropout: float = 0.1,
        # training/loss config
        distribution: str = "normal",
        learning_rate: float = 1e-4,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 5,
        lr_threshold: float = 1e-3,
        lr_min: float = 0.0,
        library_size_mode: str = "none",
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        **kwargs,
    ):
        self.model_params = {
            "n_input": n_input,
            "n_output": n_output,
            "n_cat_list": n_cat_list,
            "dim_model": dim_model,
            "num_head": num_head,
            "dim_hid": dim_hid,
            "nlayers": nlayers,
            "dropout": dropout,
            "distribution": distribution,
            "learning_rate": learning_rate,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "lr_threshold": lr_threshold,
            "lr_min": lr_min,
            "library_size_mode": library_size_mode,
            "weight_decay": weight_decay,
            "use_adamw": use_adamw,
        }

        print(
            "ReconTransformerDecoder params: "
            f"n_input={n_input}, n_output={n_output}, "
            f"dim_model={dim_model}, nlayers={nlayers}, num_head={num_head}"
        )

        self.module = ReconTransformerModule(**self.model_params)




    def decode(self, z: np.ndarray, *cat_list: np.ndarray, decode_batch_size: int = 256) -> np.ndarray:
        device = next(self.module.parameters()).device
        self.module.eval()
        # flash_attn requires fp16/bf16 regardless of stored weight dtype
        # (Lightning bf16-mixed trains with autocast but saves weights as fp32)
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        parts = []
        with torch.no_grad(), autocast_ctx:
            for start in range(0, len(z), decode_batch_size):
                z_chunk = torch.from_numpy(z[start:start + decode_batch_size]).float().to(device)
                if cat_list and len(cat_list) > 0:
                    cat_chunks = [
                        torch.from_numpy(cat[start:start + decode_batch_size]).float().to(device)
                        for cat in cat_list
                    ]
                    out = self.module(z_chunk, *cat_chunks)
                else:
                    out = self.module(z_chunk)
                parts.append(out.float().cpu())
        return torch.cat(parts, dim=0).numpy()

    def train(self, datamodule: L.LightningDataModule = None, **train_kwargs) -> None:
        max_epochs = train_kwargs.pop('max_epochs', 400)
        logger = train_kwargs.pop('logger', None)
        callbacks = train_kwargs.pop('callbacks', [])
        precision = train_kwargs.pop('precision', 'bf16-mixed')
        strategy = train_kwargs.pop('strategy', 'auto')
        devices = train_kwargs.pop('devices', 'auto')
        num_nodes = train_kwargs.pop('num_nodes', 1)
        self.resume_from_checkpoint = train_kwargs.pop('resume_from_checkpoint', None)
        
        if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
            print("Warning: bfloat16 not supported on this hardware, using fp16 mixed precision")
            precision = "16-mixed"
        
        plugins = []
        if strategy == "ddp":
            plugins.append(LightningEnvironment())

        default_trainer_kwargs = {
            "precision": precision,
            "max_epochs": max_epochs,
            "logger": logger,
            "callbacks": callbacks,
            "strategy": strategy,
            "devices": devices,
            "num_nodes": num_nodes,
            "plugins": plugins,
        }

        num_devices = devices if isinstance(devices, int) else 1
        if strategy == "ddp" and num_devices > 1 and datamodule is not None:
            datamodule.prepare_data()
            datamodule.setup(stage="fit")
            batch_size = getattr(datamodule, 'minibatch_size', 256)
            chunks_per_worker = getattr(datamodule, 'chunks_per_worker', 5)
            num_workers = getattr(datamodule, 'num_workers', 1)

            for split_name, data_attr, limit_key in [
                ("train", "train_data", "limit_train_batches"),
                ("val", "val_data", "limit_val_batches"),
            ]:
                data = getattr(datamodule, data_attr, None)
                if data is None:
                    continue
                n_samples = data.shape[0]
                n_chunks = data.numblocks[0]
                n_groups = math.ceil(n_chunks / chunks_per_worker)
                groups_per_rank = n_groups // num_devices
                effective_groups = (groups_per_rank // num_workers) * num_workers
                effective_chunks = effective_groups * chunks_per_worker
                avg_chunk_size = n_samples / n_chunks
                effective_samples = effective_chunks * avg_chunk_size
                limit = int(effective_samples / batch_size * 0.98)
                default_trainer_kwargs[limit_key] = limit
                print(f"DDP: {limit_key}={limit} "
                      f"({split_name}: {n_chunks} chunks -> {n_groups} groups -> "
                      f"{groups_per_rank}/rank -> {effective_groups} effective/rank, "
                      f"~{int(effective_samples)} samples/rank, batch={batch_size}, *0.98)")
        
        if self.resume_from_checkpoint:
            print(f"Trainer configured to resume from: {self.resume_from_checkpoint}")
        
        default_trainer_kwargs.update(train_kwargs)
        
        trainer = L.Trainer(**default_trainer_kwargs)
        
        print(f"Starting training with precision: {precision}, strategy: {strategy}, devices: {devices}")
        trainer.fit(self.module, datamodule=datamodule, ckpt_path=self.resume_from_checkpoint)
        
    def save(self, path: str):
        if not path.endswith(".pt") and not path.endswith(".ckpt"):
            path = path + ".pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save state_dict to match ReconMLPDecoder.save behaviour.
        torch.save(self.module.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load(self, path: str, map_location=None) -> None:
        """Load weights into the existing module.

        - ``.pt``: state_dict, or a whole module if that's what was saved
        - ``.ckpt``: Lightning checkpoint
        """
        if path.endswith(".pt"):
            obj = torch.load(path, map_location=map_location, weights_only=False)
            if isinstance(obj, dict):
                missing, unexpected = self.module.load_state_dict(obj, strict=False)
                if missing or unexpected:
                    print(f"Warning: missing={missing}, unexpected={unexpected}")
            elif isinstance(obj, nn.Module):
                self.module = obj
            else:
                raise TypeError(f"Unsupported .pt contents: {type(obj)}")
        elif path.endswith(".ckpt"):
            self.module = ReconTransformerModule.load_from_checkpoint(path, map_location=map_location)
        else:
            raise ValueError(f"Unsupported checkpoint extension: {path}")

        print(f"Transformer decoder loaded from {path}")