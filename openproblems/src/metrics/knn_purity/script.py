import sys
from pathlib import Path

import anndata as ad

_repo_src = Path(__file__).resolve().parents[4] / "src"
if _repo_src.is_dir() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

_common_dir = Path(__file__).resolve().parents[2] / "common"
if str(_common_dir) not in sys.path:
    sys.path.insert(0, str(_common_dir))

from openproblems_utils import read_expression, write_score  # noqa: E402
from sc_reconstruction.metrics import metric_knn_purity  # noqa: E402

## VIASH START
par = {
    "input_prediction": "resources_test/reconeval_demo/prediction_perturbed.h5ad",
    "input_solution_perturbed": "resources_test/reconeval_demo/solution_perturbed.h5ad",
    "input_solution_control": "resources_test/reconeval_demo/solution_control.h5ad",
    "output": "output.h5ad",
    "prediction_layer": "X",
    "solution_layer": "X",
    "use_rep": None,
    "k": 20,
}
meta = {"name": "knn_purity"}
## VIASH END

print("Reading input files", flush=True)
prediction = read_expression(
    ad.read_h5ad(par["input_prediction"]), par["prediction_layer"]
)
perturbed = read_expression(
    ad.read_h5ad(par["input_solution_perturbed"]), par["solution_layer"]
)
control = read_expression(
    ad.read_h5ad(par["input_solution_control"]), par["solution_layer"]
)

use_rep = par.get("use_rep") or None
if use_rep in ("", "null", "None"):
    use_rep = None

print("Computing KNN purity", flush=True)
score = metric_knn_purity(
    adata_pred=prediction,
    adata_pert_true=perturbed,
    adata_ctrl=control,
    k=int(par["k"]),
    use_rep=use_rep,
)

print("Writing output", flush=True)
write_score(
    dataset_id=prediction.uns["dataset_id"],
    normalization_id=prediction.uns["normalization_id"],
    method_id=prediction.uns["method_id"],
    metric_ids=["knn_purity"],
    metric_values=[float(score)],
    output_path=par["output"],
)
