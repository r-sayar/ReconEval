import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import sys
sys.path.append('../')

import torch
import os
import scvi
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
from hydra.utils import instantiate
import zarr


from sc_reconstruction.utils.model_loader import create_model_from_cfg
from sc_reconstruction.utils.run_tools import sample_list
import numpy as np
'''
Main multi recon DEG script

TODO: 
    1. metric_functions
'''


@hydra.main(config_path="../configs", config_name="base_eval", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f'Measuring {cfg.metric.meta.name} with biological evaluation')

    running_device = None if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {running_device}")
    # Load model - either using dynamic loading or direct instantiation
    if cfg.get('dynamic_model_loading', False):
        model, checkpoint_path = create_model_from_cfg(cfg, running_device)
        print(f"Loaded model from {checkpoint_path} using dynamic loading")
    else:
        model = instantiate(cfg.model.model_args)
        model.load(cfg.model.load.path, map_location=running_device)
        print(f"Loaded model from {cfg.model.load.path} using direct instantiation")

    # patch for old mlae models — only when the model actually uses a modeled
    # library size (has `l_encoder`). Plain AE/olAE/scVI/nlscVI have no l_encoder
    # and would crash with AttributeError if we patched them.
    import types
    if hasattr(model.module, "l_encoder"):
        def patched_forward_modeled(self, x):
            z = self.encoder(x)
            library_size = self.l_encoder(x)
            return self.decoder(z, library_size)
        model.module._forward_modeled = types.MethodType(patched_forward_modeled, model.module)
        model.module.forward_fn = model.module._forward_modeled
        print("Patched model's modeled lib forward function for compatibility.")
    else:
        print("No l_encoder on model — using default forward_fn (no patch).")

    save_path = cfg.output.save_path
    print('save path:', save_path)
    zarr_path = cfg.eval_path
    print(f"Using zarr dataset: {zarr_path}")


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