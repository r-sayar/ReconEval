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

if 'MPLBACKEND' in os.environ:
    del os.environ['MPLBACKEND']

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

STATE_SRC = os.environ.get("STATE_SRC")
if STATE_SRC:
    sys.path.insert(0, STATE_SRC)

from sc_reconstruction.utils.data_tools import *
from hydra.utils import instantiate
from lightning.pytorch import seed_everything


@hydra.main(config_path='../configs/pretrained', config_name="SE", version_base=None)
def emb(cfg: DictConfig) -> None:
    adata = sc.read_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/tahoe/split02/test.h5ad')

    gene_names = get_zarr_attr('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/tahoe/comb_w_obs_3000wCCM.zarr', target_key='var_names')
    
    pretrained_se = instantiate(cfg.model_args)
    overlap_genes = pretrained_se.get_overlap_genes(gene_names)

    pretrained_se.set_genes(gene_names)

    output_dir = '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/tahoe/split02/test_se.h5ad'

    cell_emb = pretrained_se.get_latent_representation(X=adata)
    adata.obsm['SE'] = cell_emb
    adata.write_h5ad(output_dir)
    print(f"Embedding extraction for key {cfg.data_args.target_key} completed.")

if __name__ == "__main__":
    emb()