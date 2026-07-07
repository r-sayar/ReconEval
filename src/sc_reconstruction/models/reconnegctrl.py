from __future__ import annotations

from typing import Any, Dict
import numpy as np
import os

from sc_reconstruction.models._base_model import BaseReconstructionModel


class ReconNegativeControl(BaseReconstructionModel):
    """Negative-control reconstruction: returns the training-set gene mean for every test cell.

    Predicts the same gene-expression vector (training mean) regardless of
    test-cell identity, giving zero information about the query cell.  Any
    method that genuinely exploits cell-state information should score strictly
    above this baseline on all metrics.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean: np.ndarray | None = None

    def prepare(self, data: Any = None, **kwargs) -> None:
        pass

    def train(self, X_train: np.ndarray, **kwargs) -> None:
        """Compute and store the gene-wise mean of the training data.

        Parameters
        ----------
        X_train:
            Dense float array of shape (n_train_cells, n_genes).
        """
        self.mean = np.mean(X_train, axis=0, keepdims=True).astype(np.float32)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Return training mean broadcast to match the number of test cells.

        Parameters
        ----------
        X:
            Dense float array of shape (n_test_cells, n_genes).  Only the row
            count is used — cell identity is intentionally ignored.

        Returns
        -------
        np.ndarray of shape (n_test_cells, n_genes) where every row equals the
        training mean.
        """
        if self.mean is None:
            raise RuntimeError("Call train() before predict().")
        n_cells = X.shape[0]
        return np.repeat(self.mean, n_cells, axis=0)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "mean.npy"), self.mean)

    def load(self, path: str, **kwargs) -> None:
        """Load from the directory written by save()."""
        if os.path.isdir(path):
            path = os.path.join(path, "mean.npy")
        self.mean = np.load(path)
