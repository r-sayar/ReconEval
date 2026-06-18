from __future__ import annotations
from typing import Any, Dict
import torch
import numpy as np
from anndata import AnnData

from scvi.model import SCVI
from scvi.model.base import UnsupervisedTrainingMixin, BaseMinifiedModeModelClass
from scvi import REGISTRY_KEYS


import logging
import warnings
from typing import TYPE_CHECKING

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import EmbeddingMixin, UnsupervisedTrainingMixin
from scvi.module import VAE
from scvi.utils import setup_anndata_dsp

import os
from sc_reconstruction.models._base_model import BaseReconstructionModel
from scvi.module._constants import MODULE_KEYS





class ReconMLSCVI(SCVI, BaseReconstructionModel):
    """
    A structured SCVI model for reconstruction that adheres to the base class interface.
    """
    def __init__(
        self,
        adata: AnnData | None = None,
        registry: dict | None = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        use_observed_lib_size: bool = True,
        latent_distribution: Literal["normal", "ln"] = "normal",
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        **kwargs,
    ):
        self.__class__.__name__ = "SCVI"
        BaseMinifiedModeModelClass.__init__(self, adata, registry)
        

        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            **kwargs,
        }
        self._model_summary_string = (
            "SCVI model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}."
        )
        
        self._custom_library_log_means = library_log_means
        self._custom_library_log_vars = library_log_vars

        if self._custom_library_log_means is None or self._custom_library_log_vars is None:
            warnings.warn(
                "Library size parameters not provided. Will take default values [0,1]. Unreliabe if not for inference only.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            self._custom_library_log_means = np.array(0.0).reshape(1, 1)
            self._custom_library_log_vars = np.array(1.0).reshape(1, 1)
        
        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model was initialized without `adata`. The module will be initialized when "
                "calling `train`. This behavior is experimental and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            if adata is not None:
                n_cats_per_cov = (
                    self.adata_manager.get_state_registry(
                        REGISTRY_KEYS.CAT_COVS_KEY
                    ).n_cats_per_key
                    if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
                    else None
                )
            else:
                # custom datamodule
                if (
                    len(
                        self.registry["field_registries"][f"{REGISTRY_KEYS.CAT_COVS_KEY}"][
                            "state_registry"
                        ]
                    )
                    > 0
                ):
                    n_cats_per_cov = tuple(
                        self.registry["field_registries"][f"{REGISTRY_KEYS.CAT_COVS_KEY}"][
                            "state_registry"
                        ]["n_cats_per_key"]
                    )
                else:
                    n_cats_per_cov = None

            n_batch = self.summary_stats.n_batch
            use_size_factor_key = self.registry_["setup_args"][
                f"{REGISTRY_KEYS.SIZE_FACTOR_KEY}_key"
            ]

            # Determine library size parameters
            if self._custom_library_log_means is not None and self._custom_library_log_vars is not None:
                library_log_means = self._custom_library_log_means
                library_log_vars = self._custom_library_log_vars
            else:
                library_log_means, library_log_vars = None, None
                if (
                    not use_size_factor_key
                    and self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR
                    and not use_observed_lib_size
                ):
                    library_log_means, library_log_vars = _init_library_size(
                        self.adata_manager, n_batch
                    )
            self.module = self._module_cls(
                n_input=self.summary_stats.n_vars,
                n_batch=n_batch,
                n_labels=self.summary_stats.n_labels,
                n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
                n_cats_per_cov=n_cats_per_cov,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                dispersion=dispersion,
                gene_likelihood=gene_likelihood,
                use_observed_lib_size=use_observed_lib_size,
                latent_distribution=latent_distribution,
                use_size_factor_key=use_size_factor_key,
                library_log_means=library_log_means,
                library_log_vars=library_log_vars,
                **kwargs,
            )
            self.module.minified_data_type = self.minified_data_type

        self.init_params_ = self._get_init_params(locals())

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