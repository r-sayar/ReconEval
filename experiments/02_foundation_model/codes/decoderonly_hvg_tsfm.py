from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

import torch
import lightning as L
import wandb
from sc_reconstruction.dataloaders.datamodules import *

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import zarr
import torch
import os
import json

from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

'''
Main training script for deep learning model
'''

def serialize_config_for_wandb(obj):
    """Convert any object to JSON-serializable format."""
    if isinstance(obj, DictConfig):
        obj = OmegaConf.to_container(obj, resolve=True)
    
    if isinstance(obj, dict):
        return {k: serialize_config_for_wandb(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_config_for_wandb(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Convert any other type to string
        return str(obj)

@hydra.main(config_path='../configs', config_name="base_pretrained_dec_hvg", version_base=None)
def train(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    
    # Set matrix multiplication precision
    torch.set_float32_matmul_precision(cfg.get('matmul_precision', 'high'))
    
    print("cfg.decoder:", cfg.decoder)

    def matching_gene_space(all_genes, target_genes):
        gene_idx = [all_genes.index(gene) for gene in target_genes if gene in all_genes]
        return np.array(gene_idx)
    
    all_genes_zarr = zarr.open(cfg.hvg.all_genes_zarr, mode='r')
    all_genes = all_genes_zarr.attrs['var_names']
    target_genes_zarr = zarr.open(cfg.hvg.target_genes_zarr, mode='r')
    target_genes = target_genes_zarr.attrs['var_names']
    gene_idx = matching_gene_space(all_genes, target_genes)
    print("Number of matched genes:", len(gene_idx))

    decoder_datamodule_factory = instantiate(cfg.data_args, _partial_=True)
    decoder_datamodule = decoder_datamodule_factory(
        target_feature_indices=gene_idx,
    )

    recon_decoder = instantiate(cfg.decoder.model_args)
    
    # Fix: Create W&B logger with serializable config
    if "logging" in cfg.decoder and cfg.decoder.logging.wandb:
        # Get the wandb config as a dict
        wandb_config = OmegaConf.to_container(cfg.decoder.logging.wandb, resolve=True)
        
        # Extract config for wandb logging
        if 'config' in wandb_config:
            # config_to_log is already a dict (not OmegaConf)
            config_to_log = wandb_config['config']
        else:
            # Convert entire cfg to dict
            config_to_log = OmegaConf.to_container(cfg, resolve=True)
        
        # Serialize the config
        serialized_config = serialize_config_for_wandb(config_to_log)
        
        # Remove keys that shouldn't be passed to WandbLogger constructor
        wandb_config.pop('_target_', None)
        wandb_config.pop('config', None)
        
        # Create the logger
        logger = WandbLogger(
            **wandb_config,
            config=serialized_config,
        )
    else:
        logger = None
    
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.decoder.training.callbacks] if "callbacks" in cfg.decoder.training else []

    max_epochs = cfg.decoder.max_epochs
    
    # Determine appropriate precision
    if hasattr(cfg, 'precision') and cfg.precision in ["bf16-mixed", "16-mixed", "32-true"]:
        trainer_precision = cfg.precision
    else:
        trainer_precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    
    ddp_cfg = cfg.decoder.get("ddp", {})
    strategy = ddp_cfg.get("strategy", "auto")
    devices = ddp_cfg.get("devices", "auto")
    num_nodes = ddp_cfg.get("num_nodes", 1)
    
    recon_decoder.train(
        datamodule=decoder_datamodule,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        precision=trainer_precision,
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
    )
    
    if logger:
        wandb.finish()

if __name__ == "__main__":
    train()