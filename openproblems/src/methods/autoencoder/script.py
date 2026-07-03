import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_repo_src = Path(__file__).resolve().parents[4] / "src"
if _repo_src.is_dir() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

from sc_reconstruction.dataloaders import H5adReconstructionDataModule  # noqa: E402

## VIASH START
par = {
    "input_train": "resources_test/reconeval/luca/train.h5ad",
    "input_test": "resources_test/reconeval/luca/test.h5ad",
    "output": "output.h5ad",
    "n_latent": 32,
    "n_hidden": 256,
    "n_layers": 2,
    "epochs": 40,
    "batch_size": 256,
    "learning_rate": 0.001,
    "seed": 0,
}
meta = {"name": "autoencoder"}
## VIASH END


def _build_mlp(in_dim: int, hidden: int, layers: int, out_dim: int) -> nn.Sequential:
    blocks: list[nn.Module] = []
    prev = in_dim
    for _ in range(layers):
        blocks += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
        prev = hidden
    blocks.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*blocks)


class Autoencoder(nn.Module):
    def __init__(self, n_vars: int, n_hidden: int, n_layers: int, n_latent: int):
        super().__init__()
        self.encoder = _build_mlp(n_vars, n_hidden, n_layers, n_latent)
        self.decoder = _build_mlp(n_latent, n_hidden, n_layers, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def main() -> None:
    torch.manual_seed(int(par["seed"]))
    np.random.seed(int(par["seed"]))

    print(">> Load h5ad splits", flush=True)
    dm = H5adReconstructionDataModule(
        train_path=par["input_train"],
        test_path=par["input_test"],
    )
    dm.prepare_data()
    train_adata = dm.train_adata
    test_adata = dm.test_adata

    X_train = dm.get_train_matrix().astype(np.float32)
    X_test = dm.get_test_matrix().astype(np.float32)

    n_vars = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(
        n_vars=n_vars,
        n_hidden=int(par["n_hidden"]),
        n_layers=int(par["n_layers"]),
        n_latent=int(par["n_latent"]),
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(par["learning_rate"]))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=int(par["batch_size"]),
        shuffle=True,
        drop_last=X_train.shape[0] > int(par["batch_size"]),
    )

    print(
        f">> Train AE on {X_train.shape} for {par['epochs']} epochs (device={device})",
        flush=True,
    )
    model.train()
    for epoch in range(int(par["epochs"])):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optim.zero_grad()
            recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            loss.backward()
            optim.step()
            total += float(loss) * batch.shape[0]
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"   epoch {epoch + 1}: loss={total / X_train.shape[0]:.4f}", flush=True)

    print(">> Reconstruct test", flush=True)
    model.eval()
    with torch.no_grad():
        X_pred = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    print(">> Write prediction", flush=True)
    output = ad.AnnData(
        X=X_pred.astype(np.float32),
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
