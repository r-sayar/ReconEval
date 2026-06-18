import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from functools import partial
import numpy as np
import zarr

from sc_reconstruction.metrics.utils import gene_r_squared, mean_squared_error, mmd_rbf, energy_distance
from sc_reconstruction.metrics.evaluation import ProjectionMetricsBatchEvaluator
from sc_reconstruction.models.reconknn import ReconKNN


@hydra.main(config_path="../configs", config_name="base_metrics_sampled", version_base=None)
def main(cfg: DictConfig) -> None:
    
    model = ReconKNN(n_neighbors=cfg.model.model_args.n_neighbors,
                     metric=cfg.model.model_args.metric,
                     data_path=cfg.model.model_args.data_path,
                     batch_size=cfg.model.model_args.batch_size)

    # Check if projector configuration exists
    if not hasattr(cfg, 'projector'):
        raise ValueError("Projector configuration is missing. Please specify projector config using '+projector=projector'")
    
    # Load projection model (e.g., PCA)
    print("Loading projection model...")
    projector = instantiate(cfg.metric.projector.model_args)
    projector.load(cfg.metric.projector.load.path)
    
    # Define metrics
    metric_functions = {
        "r2_score": gene_r_squared,
        "mse": mean_squared_error,
        "mmd_rbf": partial(mmd_rbf, gamma=0.5),
        "energy_distance": energy_distance,
    }
    


    save_path = cfg.output.save_path
    print('save path:', save_path)
    
    # Load zarr dataset and get all combinations
    zarr_path = cfg.eval_path
    # zarr_path = '/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/comb_w_obs/cl_drg_dos.zarr'
    print(f"Using zarr dataset: {zarr_path}")
    
    root_zarr = zarr.open(zarr_path, mode='r')

    split_zarr = zarr.open(cfg.split_comb, mode='r')
    test_combs = split_zarr.attrs['test_combinations']
    train_combs = split_zarr.attrs['train_combinations']
    train_length = len(train_combs)


    # Sample 10% of combinations 
    seed = cfg.get("seed", 42) 
    sample_fraction = cfg.get("sample_fraction", 0.1)

    rng = np.random.default_rng(seed)
    sample_size = int(train_length * sample_fraction)
    sampled_indices = rng.choice(train_length, size=sample_size, replace=False)
    train_comb_list = [train_combs[i] for i in sampled_indices]
    comb_list = list(test_combs) + train_comb_list
    
    print(f"Sampled {sample_size} combinations ({sample_fraction*100:.1f}%) with seed {seed}") 
    print(f"Total combinations: {len(comb_list)}, test combinations: {len(test_combs)}, sampled train combinations: {len(train_comb_list)}")
    print(f"First 5 sampled combinations: {comb_list[:5]}")
    

    
    evaluator_factory = instantiate(cfg.metric.evaluator, _partial_=True)
    evaluator = evaluator_factory(
        model=model,
        projector=projector,
        comb_list=comb_list,
        metric_funcs=metric_functions,
        projection_metrics=projection_metrics,
        batch_size=cfg.get("batch_size", 500),
        max_cells=cfg.get("max_cells", 1e5),
        split_key=cfg.split_key,
        max_workers=cfg.get("max_workers", 20)
    )
    
    # Run evaluation
    evaluator.run(save_path)
    
    print("Successfully saved results to:", save_path)

if __name__ == "__main__":
    main() 