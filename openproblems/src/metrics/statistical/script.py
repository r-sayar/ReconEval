import sys
from pathlib import Path

import anndata as ad

# Allow local development without rebuilding Docker images.
_repo_src = Path(__file__).resolve().parents[4] / "src"
if _repo_src.is_dir() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

_common_dir = Path(__file__).resolve().parents[2] / "common"
if str(_common_dir) not in sys.path:
    sys.path.insert(0, str(_common_dir))

from openproblems_utils import align_genes, read_expression, write_score  # noqa: E402
from sc_reconstruction.metrics import compute_statistical_metrics  # noqa: E402

## VIASH START
par = {
    "input_solution": "resources_test/reconeval/luca/solution.h5ad",
    "input_prediction": "resources_test/reconeval/luca/prediction.h5ad",
    "output": "output.h5ad",
    "solution_layer": "X",
    "prediction_layer": "X",
    "resolve_genes": "intersection",
}
meta = {"name": "statistical"}
## VIASH END

print("Reading input files", flush=True)
solution = read_expression(ad.read_h5ad(par["input_solution"]), par["solution_layer"])
prediction = read_expression(ad.read_h5ad(par["input_prediction"]), par["prediction_layer"])

print("Aligning genes", flush=True)
solution, prediction = align_genes(
    solution, prediction, resolve_genes=par["resolve_genes"]
)

if par["solution_layer"] == par["prediction_layer"] == "X":
    if solution.n_obs == prediction.n_obs:
        if not (solution.obs_names == prediction.obs_names).all():
            raise ValueError(
                "Matched n_obs but obs_names differ; reorder prediction to match "
                "solution before computing MSE."
            )
    else:
        print(
            "Warning: n_obs differs between solution and prediction; "
            "MSE will be skipped.",
            flush=True,
        )

print("Computing ReconEval statistical metrics", flush=True)
scores = compute_statistical_metrics(solution, prediction)

final_ids = []
final_values = []
for metric_id in ["r2", "mse", "energy_distance"]:
    if metric_id == "mse" and solution.n_obs != prediction.n_obs:
        print("Skipping MSE because n_obs differs.", flush=True)
        continue
    final_ids.append(metric_id)
    final_values.append(float(scores[metric_id]))

print("Writing output", flush=True)
write_score(
    dataset_id=prediction.uns["dataset_id"],
    normalization_id=prediction.uns["normalization_id"],
    method_id=prediction.uns["method_id"],
    metric_ids=final_ids,
    metric_values=final_values,
    output_path=par["output"],
)
