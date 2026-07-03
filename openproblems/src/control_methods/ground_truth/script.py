import anndata as ad
import numpy as np

## VIASH START
par = {
    "input_train": "resources_test/reconeval/luca/train.h5ad",
    "input_test": "resources_test/reconeval/luca/test.h5ad",
    "input_solution": "resources_test/reconeval/luca/solution.h5ad",
    "output": "output.h5ad",
}
meta = {"name": "ground_truth"}
## VIASH END

print(">> Read inputs", flush=True)
train = ad.read_h5ad(par["input_train"])
solution = ad.read_h5ad(par["input_solution"])

X = solution.X
if hasattr(X, "toarray"):
    X = X.toarray()

output = ad.AnnData(
    X=np.asarray(X, dtype=np.float32),
    obs=solution.obs.copy(),
    var=solution.var.copy(),
    uns={
        "dataset_id": train.uns["dataset_id"],
        "normalization_id": train.uns["normalization_id"],
        "method_id": meta["name"],
    },
)
output.obs_names = solution.obs_names
output.var_names = solution.var_names
output.write_h5ad(par["output"], compression="gzip")
