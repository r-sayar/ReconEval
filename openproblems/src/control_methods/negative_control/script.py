import anndata as ad
import numpy as np

## VIASH START
par = {
    "input_train": "resources_test/reconeval/luca/train.h5ad",
    "input_test": "resources_test/reconeval/luca/test.h5ad",
    "input_solution": "resources_test/reconeval/luca/solution.h5ad",
    "output": "output.h5ad",
}
meta = {"name": "negative_control"}
## VIASH END

print(">> Read inputs", flush=True)
train = ad.read_h5ad(par["input_train"])
test = ad.read_h5ad(par["input_test"])

X_train = train.X
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
X_train = np.asarray(X_train, dtype=np.float32)

print(">> Compute training mean", flush=True)
train_mean = X_train.mean(axis=0)  # shape: (n_genes,)

n_test = test.n_obs
X_pred = np.broadcast_to(train_mean, (n_test, len(train_mean))).copy()

print(">> Write prediction", flush=True)
output = ad.AnnData(
    X=X_pred.astype(np.float32),
    obs=test.obs.copy(),
    var=test.var.copy(),
    uns={
        "dataset_id": train.uns["dataset_id"],
        "normalization_id": train.uns["normalization_id"],
        "method_id": meta["name"],
    },
)
output.obs_names = test.obs_names
output.var_names = test.var_names
output.write_h5ad(par["output"], compression="gzip")
