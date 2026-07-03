"""h5ad-backed datamodule for gene-expression reconstruction.

Kept in a separate module from :mod:`datamodules` so it can be imported
without pulling in the Dask / Zarr / Lightning / omegaconf stack that the
Zarr-based streaming classes require. This is what OpenProblems Viash
components use to load ``train`` / ``val`` / ``test`` splits.
"""

from __future__ import annotations

import numpy as np


class H5adReconstructionDataModule:
    """h5ad-backed datamodule for gene-expression reconstruction.

    Loads train / val / test splits from ``.h5ad`` files instead of Zarr.
    Intended for OpenProblems pipelines and local prototyping.
    """

    def __init__(
        self,
        train_path: str,
        val_path: str | None = None,
        test_path: str | None = None,
        layer: str = "X",
        batch_size: int = 256,
        num_workers: int = 0,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.layer = layer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_adata = None
        self.val_adata = None
        self.test_adata = None
        self.n_vars: int | None = None

    @staticmethod
    def read_h5ad(path: str, layer: str = "X"):
        """Load an AnnData and optionally move a layer into ``.X``."""
        import anndata as ad

        adata = ad.read_h5ad(path)
        if layer == "X":
            return adata
        if layer in adata.layers:
            out = adata.copy()
            out.X = adata.layers[layer]
            return out
        raise KeyError(f"Layer {layer!r} not found in {path}")

    def prepare_data(self) -> None:
        self.train_adata = self.read_h5ad(self.train_path, self.layer)
        self.n_vars = self.train_adata.n_vars
        if self.val_path:
            self.val_adata = self.read_h5ad(self.val_path, self.layer)
        if self.test_path:
            self.test_adata = self.read_h5ad(self.test_path, self.layer)

    def get_train_matrix(self) -> np.ndarray:
        if self.train_adata is None:
            self.prepare_data()
        return _as_dense_matrix(self.train_adata.X)

    def get_test_matrix(self) -> np.ndarray:
        if self.test_path is None:
            raise ValueError("test_path was not provided.")
        if self.test_adata is None:
            self.prepare_data()
        return _as_dense_matrix(self.test_adata.X)


def _as_dense_matrix(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)
