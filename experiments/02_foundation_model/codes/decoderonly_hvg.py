from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

import torch
import lightning
import wandb
from sc_reconstruction.dataloaders.datamodules import *

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import os
import scvi

from hydra.utils import instantiate
from lightning.pytorch import seed_everything

'''
Main training script for deep learning model
'''

@hydra.main(config_path='../configs', config_name="base_pretrained_dec_hvg", version_base=None)
def train(cfg: DictConfig) -> None:


    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.precision)
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

    # Debugging: check n_features
    # decoder_datamodule.prepare_data()
    # decoder_datamodule.setup(stage='fit')
    # n_features = decoder_datamodule.n_vars 
    # print(f"Setting model output dimension to {n_features} for HVG subset")

    recon_decoder = instantiate(cfg.decoder.model_args)
    logger = instantiate(cfg.decoder.logging.wandb) if "logging" in cfg.decoder else None
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.decoder.training.callbacks] if "callbacks" in cfg.decoder.training else []

    max_epochs = cfg.decoder.max_epochs
    recon_decoder.train(
        datamodule=decoder_datamodule,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
    )
    wandb.finish()

if __name__ == "__main__":
    train()