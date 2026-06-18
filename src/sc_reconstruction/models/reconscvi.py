from __future__ import annotations

from typing import Any, Dict
import torch
import numpy as np
from anndata import AnnData

from scvi.model import SCVI
from scvi.model.base import UnsupervisedTrainingMixin
from scvi import REGISTRY_KEYS

import os
from sc_reconstruction.models._base_model import BaseReconstructionModel
from scvi.module._constants import MODULE_KEYS


class ReconSCVI(SCVI, BaseReconstructionModel):
    """
    A structured SCVI model for reconstruction that adheres to the base class interface.
    """
    def __init__(
        self,
        adata: AnnData | None = None,
        registry: Dict | None = None,
        **kwargs 
    ):
        self.__class__.__name__ = "SCVI"
        super().__init__(adata=adata, registry=registry, **kwargs)

    def prepare(self, adata: Any, **kwargs) -> None:
        """Register adata with the SCVI data manager and cache the reference."""
        self.setup_anndata(adata, **kwargs)
        self.adata = adata

    def train(self, **train_kwargs) -> None:
        """Drop the unused ``save_path`` kwarg and delegate to ``SCVI.train``."""
        train_kwargs.pop('save_path', None)
        super().train(**train_kwargs)

    def _prepare_batch(self, X: np.ndarray) -> dict:
        """
        Convert a raw NumPy array into the mini_batch dictionary
        required by self.module.forward.
        """
        return {
            REGISTRY_KEYS.X_KEY: torch.from_numpy(X),
            REGISTRY_KEYS.BATCH_KEY: torch.zeros((X.shape[0], 1), dtype=torch.int64),
            REGISTRY_KEYS.LABELS_KEY: torch.zeros((X.shape[0], 1), dtype=torch.int64),
        }

    def _predict_batch(self, mini_batch: dict, inference_kwargs: dict) -> torch.Tensor:
        """
        Runs the model forward pass on a mini_batch and returns the reconstructed output.
        """
        self.module.eval()
        with torch.no_grad():
            _, generative_outputs = self.module.forward(
                tensors=mini_batch,
                inference_kwargs=inference_kwargs,
                compute_loss=False
            )
        return generative_outputs["px"].loc

    def _predict_batch_deterministic(self, mini_batch: dict, inference_kwargs: dict) -> torch.Tensor:
        """
        Runs the model forward pass on a mini_batch and returns the reconstructed output.
        """
        self.module.eval()
        with torch.no_grad():
            inference_outputs, generative_outputs = self.module.forward(
                tensors=mini_batch,
                inference_kwargs=inference_kwargs,
                compute_loss=False
            )
            qz = inference_outputs["qz"]  
            z_mean = qz.loc
            ql = inference_outputs["ql"]  
            l_mean = ql.loc
            
            generative_inputs = self.module._get_generative_input(
                mini_batch, inference_outputs

            )
            generative_inputs[MODULE_KEYS.Z_KEY] = z_mean
            if not self.use_observed_lib_size:
                generative_inputs[MODULE_KEYS.LIBRARY_KEY] = l_mean
            generative_outputs_deterministic = self.module.generative(**generative_inputs)
        
        return generative_outputs_deterministic["px"].loc

    def _predict_deterministic(self, 
                X: np.ndarray,
                inference_kwargs: dict = None) -> np.ndarray:
        """
        deterministic prediction for scvi style models
        """ 
        mini_batch = self._prepare_batch(X)
        pred_tensor = self._predict_batch_deterministic(mini_batch, inference_kwargs)
        return pred_tensor.cpu().numpy()

    def _predict_random(self, 
                X: np.ndarray,
                inference_kwargs: dict = None) -> np.ndarray:
        """
        random prediction for scvi style models
        
        Parameters:
        X: np.ndarray - Raw input data.
        inference_kwargs: Optional dictionary of inference parameters (default: {"n_samples": 1, "return_mean": True}).
        Returns:
        np.ndarray: The predicted/reconstructed output.
        """


        mini_batch = self._prepare_batch(X)
        pred_tensor = self._predict_batch(mini_batch, inference_kwargs)
        return pred_tensor.cpu().numpy()
        
    def predict(self, 
                X: np.ndarray,
                inference_kwargs: dict = None,
                pred_type = "random") -> np.ndarray:
        """
        Parameters:
        X: np.ndarray - Raw input data.
        inference_kwargs: Optional dictionary of inference parameters (default: {"n_samples": 1, "return_mean": True}).
        pred_type: Optional string indicating the type of prediction ("random" or "deterministic").
        """ 

        if pred_type == "random":
            return self._predict_random(X, inference_kwargs)
        elif pred_type == "deterministic":
            return self._predict_deterministic(X, inference_kwargs)
        else:
            raise ValueError(f"Invalid prediction type: {pred_type}")

    def get_latent_representation(self, X: np.ndarray, **inference_kwargs) -> np.ndarray:
        
        mini_batch = self._prepare_batch(X)
        self.module.eval()
        with torch.no_grad():
            inference_outputs, generative_outputs = self.module.forward(
                tensors=mini_batch,
                inference_kwargs=inference_kwargs,
                compute_loss=False
            )
            qz = inference_outputs["qz"]  
            z_mean = qz.loc
        return z_mean.cpu().numpy()
        

    def save(self, path: str, **kwargs) -> None:
        """Save the model module to ``path``.

        Args:
            path: Destination path; ``.pt`` is appended if no extension is set.
            kwargs: Forwarded to the parent class (e.g. ``save_anndata``,
                ``overwrite``).
        """
        # Add .pt extension if no .pt or .ckpt extension is present
        if not path.endswith('.pt') and not path.endswith('.ckpt'):
            path = path + '.pt'
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.module, path)
        print(f"Model saved to {path}")

    def load(self, path: str,  map_location=None) -> None:
        """Load the model"""
        self.module = torch.load(path, map_location=map_location, weights_only=False)
        print(f"Model loaded from {path}")