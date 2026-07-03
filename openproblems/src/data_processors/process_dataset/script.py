import numpy as np
import scanpy as sc
import anndata as ad

## VIASH START
par = {
    "input": "resources_test/common/luca/dataset.h5ad",
    "method": "random",
    "n_hvg": 2000,
    "test_fraction": 0.2,
    "obs_origin": "origin",
    "test_origin": "tumor_primary",
    "seed": 0,
    "output_train": "train.h5ad",
    "output_test": "test.h5ad",
    "output_solution": "solution.h5ad",
}
meta = {"name": "process_dataset"}
## VIASH END


def _as_dense(x):
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


print(">> Load data", flush=True)
adata = ad.read_h5ad(par["input"])
print("input:", adata, flush=True)

dataset_id = adata.uns.get("dataset_id", "unknown")
normalization_id = adata.uns.get("normalization_id", "log1p_cp10k")

if "counts" in adata.layers and normalization_id == "counts":
    print(">> Normalize counts", flush=True)
    adata = adata.copy()
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    normalization_id = "log1p_cp10k"

print(f">> Select top {par['n_hvg']} HVGs", flush=True)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=min(int(par["n_hvg"]), adata.n_vars - 1),
    flavor="seurat_v3" if _as_dense(adata.X).min() >= 0 else "seurat",
    subset=True,
)

print(f">> Split using method={par['method']}", flush=True)
rng = np.random.default_rng(int(par["seed"]))
if par["method"] == "random":
    n_test = max(1, int(round(adata.n_obs * float(par["test_fraction"]))))
    test_ix = rng.choice(adata.n_obs, size=n_test, replace=False)
    is_test = np.zeros(adata.n_obs, dtype=bool)
    is_test[test_ix] = True
elif par["method"] == "origin":
    if par["obs_origin"] not in adata.obs.columns:
        raise KeyError(f"{par['obs_origin']!r} not in adata.obs")
    is_test = adata.obs[par["obs_origin"]].astype(str) == par["test_origin"]
    if is_test.sum() == 0:
        raise ValueError(f"No cells with origin={par['test_origin']!r}")
else:
    raise ValueError(f"Unknown split method: {par['method']!r}")

train = adata[~is_test].copy()
test = adata[is_test].copy()
solution = test.copy()

for obj in (train, test, solution):
    obj.uns["dataset_id"] = dataset_id
    obj.uns["normalization_id"] = normalization_id

print(">> Write outputs", flush=True)
print(f"   train: {train.shape}, test: {test.shape}", flush=True)
train.write_h5ad(par["output_train"], compression="gzip")
test.write_h5ad(par["output_test"], compression="gzip")
solution.write_h5ad(par["output_solution"], compression="gzip")
