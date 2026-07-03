import math
import sys
from pathlib import Path

import anndata as ad
import numpy as np

_repo_src = Path(__file__).resolve().parents[4] / "src"
if _repo_src.is_dir() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

_common_dir = Path(__file__).resolve().parents[2] / "common"
if str(_common_dir) not in sys.path:
    sys.path.insert(0, str(_common_dir))

from openproblems_utils import align_genes, read_expression, write_score  # noqa: E402
from sc_reconstruction.metrics import (  # noqa: E402
    compute_biological_metrics,
    load_cell_cycle_genes,
    load_cytokine_dict_from_csv,
)

## VIASH START
par = {
    "input_solution": "resources_test/reconeval_demo/solution.h5ad",
    "input_prediction": "resources_test/reconeval_demo/prediction.h5ad",
    "output": "output.h5ad",
    "solution_layer": "X",
    "prediction_layer": "X",
    "resolve_genes": "intersection",
    "cell_cycle_genes": "resources/regev_lab_cell_cycle_genes.txt",
    "cytokine_signatures": None,
    "input_reference_solution": "resources_test/reconeval_demo/reference_solution.h5ad",
    "input_reference_prediction": "resources_test/reconeval_demo/reference_prediction.h5ad",
    "reference_solution_layer": "X",
    "reference_prediction_layer": "X",
    "min_cells": 5,
}
meta = {"name": "biological"}
## VIASH END

print("Reading input files", flush=True)
solution = read_expression(ad.read_h5ad(par["input_solution"]), par["solution_layer"])
prediction = read_expression(ad.read_h5ad(par["input_prediction"]), par["prediction_layer"])
solution, prediction = align_genes(
    solution, prediction, resolve_genes=par["resolve_genes"]
)

s_genes, g2m_genes = load_cell_cycle_genes(par["cell_cycle_genes"])

cytokine_dict = None
if par.get("cytokine_signatures"):
    cytokine_dict = load_cytokine_dict_from_csv(par["cytokine_signatures"])

deg_refs = None
if par.get("input_reference_solution") and par.get("input_reference_prediction"):
    ref_solution = read_expression(
        ad.read_h5ad(par["input_reference_solution"]),
        par["reference_solution_layer"],
    )
    ref_prediction = read_expression(
        ad.read_h5ad(par["input_reference_prediction"]),
        par["reference_prediction_layer"],
    )
    ref_solution, ref_prediction = align_genes(
        ref_solution, ref_prediction, resolve_genes=par["resolve_genes"]
    )
    deg_refs = (ref_solution, ref_prediction)

print("Computing ReconEval biological metrics", flush=True)
scores = compute_biological_metrics(
    solution,
    prediction,
    s_genes=s_genes,
    g2m_genes=g2m_genes,
    cytokine_dict=cytokine_dict,
    deg_refs=deg_refs,
    min_cells=int(par["min_cells"]),
)

metric_order = [
    "cellcycle_proportion_same_phase",
    "coexpression",
    "pathway",
    "deg_dice_at_100",
    "deg_logfc_spearman",
    "cytokine",
]

final_ids = []
final_values = []
for metric_id in metric_order:
    value = float(scores.get(metric_id, float("nan")))
    if math.isnan(value):
        print(f"Skipping {metric_id} (not computed).", flush=True)
        continue
    final_ids.append(metric_id)
    final_values.append(value)

if not final_ids:
    raise RuntimeError("No biological metrics were computed.")

print("Writing output", flush=True)
write_score(
    dataset_id=prediction.uns["dataset_id"],
    normalization_id=prediction.uns["normalization_id"],
    method_id=prediction.uns["method_id"],
    metric_ids=final_ids,
    metric_values=final_values,
    output_path=par["output"],
)
