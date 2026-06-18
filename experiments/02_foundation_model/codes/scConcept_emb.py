import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm
import zarr
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

from sc_reconstruction.utils.data_tools import *
from hydra.utils import instantiate
from lightning.pytorch import seed_everything


@hydra.main(config_path='../configs/pretrained', config_name="scConcept", version_base=None)
def emb(cfg: DictConfig) -> None:
    all_combinations = get_zarr_attr(cfg.data_args.split_meta_root, target_key=cfg.data_args.target_key)
    print(f"All combinations for key {cfg.data_args.target_key}: len {len(all_combinations)}")

    total_combinations = len(all_combinations)
    part_size = total_combinations // cfg.total_parts
    start_idx = (cfg.parts - 1) * part_size
    end_idx = cfg.parts * part_size if cfg.parts < cfg.total_parts else total_combinations
    selected_combinations = all_combinations[start_idx:end_idx]
    print(f"Processing part {cfg.parts}/{cfg.total_parts}: combinations {start_idx} to {end_idx}.")
     
    test_dataloader = split_zarr_dataloader(
        cfg.data_args.comb_meta_root, 
        selected_combinations, 
        target_key=cfg.data_args.target_key,
        # idx_key='obs_index'
        )
    gene_names = zarr.open(cfg.data_args.var_names_path, mode='r').attrs[cfg.data_args.var_names_key]
    
    pretrained_fm = instantiate(cfg.model_args)
    pretrained_fm.set_genes(gene_names)

    output_dir = cfg.data_args.output_dir
    state_zarr = zarr.open(output_dir, mode='a')
    CHUNK_SIZE = 10_000
    existing_combinations = list(state_zarr.group_keys())
    print(f"{len(existing_combinations)} existing combinations in output Zarr")
    for combination, X in test_dataloader:
        if combination in existing_combinations:
            print(f"Combination {combination} already processed. Skipping.")
            continue
        cell_emb = pretrained_fm.get_latent_representation(X=X)
        state_zarr.create_group(combination, overwrite=True)
        state_zarr[combination].create_dataset(
            'X', 
            data=cell_emb, 
            chunks=(CHUNK_SIZE, cell_emb.shape[1])  
        )
        # state_zarr[combination].create_dataset(
        #     'X',
        #     data=X,
        #     chunks=(CHUNK_SIZE, X.shape[1])
        # )
        # state_zarr[combination].create_dataset(
        #     'obs_index',
        #     data=obs_index,
        #     chunks=(CHUNK_SIZE,)
        # )

    print(f"Embedding extraction for key {cfg.data_args.target_key} completed.")

if __name__ == "__main__":
    emb()