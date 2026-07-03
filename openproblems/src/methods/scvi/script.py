import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scvi
import torch

_repo_src = Path(__file__).resolve().parents[4] / "src"
if _repo_src.is_dir() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

from sc_reconstruction.dataloaders import H5adReconstructionDataModule  # noqa: E402

## VIASH START
par = {
    "input_train": "resources_test/reconeval/luca/train.h5ad",
    "input_test": "resources_test/reconeval/luca/test.h5ad",
    "output": "output.h5ad",
    "n_latent": 10,
    "n_hidden": 128,
    "n_layers": 1,
    "max_epochs": 100,
    "seed": 0,
}
meta = {"name": "scvi"}
## VIASH END


def main() -> None:
    scvi.settings.seed = int(par["seed"])
    torch.manual_seed(int(par["seed"]))
    np.random.seed(int(par["seed"]))

    print(">> Load h5ad splits", flush=True)
    dm = H5adReconstructionDataModule(
        train_path=par["input_train"],
        test_path=par["input_test"],
    )
    dm.prepare_data()
    train_adata = dm.train_adata.copy()
    test_adata = dm.test_adata.copy()

    print(f">> Setup SCVI on train {train_adata.shape}", flush=True)
    scvi.model.SCVI.setup_anndata(train_adata)
    model = scvi.model.SCVI(
        train_adata,
        n_latent=int(par["n_latent"]),
        n_hidden=int(par["n_hidden"]),
        n_layers=int(par["n_layers"]),
    )
    print(f">> Train for up to {par['max_epochs']} epochs", flush=True)
    model.train(max_epochs=int(par["max_epochs"]))

    print(">> Predict normalized expression on test", flush=True)
    scvi.model.SCVI.setup_anndata(test_adata)
    X_pred = model.get_normalized_expression(
        adata=test_adata,
        return_numpy=True,
        library_size="latent",
    )

    print(">> Write prediction", flush=True)
    output = ad.AnnData(
        X=np.asarray(X_pred, dtype=np.float32),
        obs=test_adata.obs.copy(),
        var=test_adata.var.copy(),
        uns={
            "dataset_id": train_adata.uns["dataset_id"],
            "normalization_id": train_adata.uns["normalization_id"],
            "method_id": meta["name"],
        },
    )
    output.obs_names = test_adata.obs_names
    output.var_names = test_adata.var_names
    output.write_h5ad(par["output"], compression="gzip")


if __name__ == "__main__":
    main()
