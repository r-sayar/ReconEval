import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import sys
sys.path.append('../')

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from functools import partial
import pandas as pd
import numpy as np
import zarr
from sc_reconstruction.metrics.utils import gene_r_squared, mean_squared_error, mmd_rbf, energy_distance
from sc_reconstruction.utils.run_tools import sample_list, find_model_paths




@hydra.main(config_path="../configs", config_name="base_eval_decode", version_base=None)
def main(cfg: DictConfig) -> None:

    running_device = None if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {running_device}")
    # Define metrics
    metric_functions = {
        "r2_score": gene_r_squared,
        "mmd_rbf": partial(mmd_rbf, gamma=0.5),
        "energy_distance": energy_distance,
    }

    model = instantiate(cfg.decoder.model_args)
    model.load(cfg.decoder.load.path, map_location=running_device)
    print(f"Loaded model from {cfg.decoder.load.path} using direct instantiation")

    save_path = cfg.output.save_path
    print('save path:', save_path)
    zarr_path = cfg.eval_path
    print(f"Using zarr dataset: {zarr_path}")
    emb_path = cfg.emb_zarr_path
    print(f"Using embedding zarr dataset: {emb_path}")
    root_zarr = zarr.open(zarr_path, mode='r')
    print('Split info path:', cfg.data.split_comb)
    split_zarr = zarr.open(cfg.data.split_comb, mode='r')
    test_combs = split_zarr.attrs['test_combinations']
    train_combs = split_zarr.attrs['train_combinations']
    train_length = len(train_combs)


    
    # Sample 10% of combinations 
    seed = cfg.get("seed", 42) 
    train_fraction = cfg.get("train_fraction", 0.10)  
    test_fraction  = cfg.get("test_fraction", 1.00)    

    rng = np.random.default_rng(seed)
    if train_fraction > 1e-4:
        train_comb_list = sample_list(train_combs, train_fraction, rng)
    else:
        train_comb_list = []
    test_comb_list = sample_list(test_combs, test_fraction, rng)
    comb_list = list(test_comb_list) + list(train_comb_list)

    
    print(f"Sampled {test_fraction} test and {train_fraction} train with seed {seed}")
    print(f"Total combinations: {len(comb_list)}, test combinations: {len(test_comb_list)}, train combinations: {len(train_comb_list)}")
    print(f"First 5 sampled combinations: {comb_list[:5]}")
    
    var_names = root_zarr.attrs['var_names']
    if cfg.target_var_names_path is not None:
        target_var_names = zarr.open(cfg.target_var_names_path, mode='r')['var_names'][:]
        print(f"Using custom var names from {cfg.var_names_path} and target var names from {cfg.target_var_names_path}")
        
        cfg.metric.evaluator.update({
            "target_var_names": target_var_names,
        })
    evaluator_factory = instantiate(cfg.metric.evaluator, _partial_=True)
    evaluator = evaluator_factory(
        model=model,
        comb_list=comb_list,
        var_names=var_names,
    )
    evaluator.run(save_path)
    print(f"Successfully {cfg.metric.meta.name} saved results to:", save_path)

if __name__ == "__main__":
    main() 