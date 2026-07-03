import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA

_repo_src = Path(__file__).resolve().parents[4] / "src"
if _repo_src.is_dir() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

from sc_reconstruction.dataloaders import H5adReconstructionDataModule  # noqa: E402

## VIASH START
par = {
    "input_train": "resources_test/reconeval/luca/train.h5ad",
    "input_test": "resources_test/reconeval/luca/test.h5ad",
    "output": "output.h5ad",
    "n_components": 64,
}
meta = {"name": "pca_reconstruction"}
## VIASH END

print(">> Load h5ad splits", flush=True)
dm = H5adReconstructionDataModule(
    train_path=par["input_train"],
    test_path=par["input_test"],
)
dm.prepare_data()
train = dm.train_adata
test = dm.test_adata

n_components = min(int(par["n_components"]), train.n_vars, train.n_obs - 1)
print(f">> Fit PCA with n_components={n_components}", flush=True)
X_train = dm.get_train_matrix()
X_test = dm.get_test_matrix()

pca = PCA(n_components=n_components, random_state=0)
pca.fit(X_train)
X_pred = pca.inverse_transform(pca.transform(X_test))

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
