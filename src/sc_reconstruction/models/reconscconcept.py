from __future__ import annotations

from concept import scConcept
import scanpy as sc
import numpy as np
import warnings
from anndata import AnnData
import anndata as ad
import lightning as L
# Can't inherit from baserecon class due to version conflict
# Largely simplified version
class ReconPretrainedscConcept(): 
    def __init__(
        self,
        checkpoint_path: str, 
        emb_key: str = 'X_scConcept',
        **kwargs
    ):
        """Pre-trained scConcept foundation model wrapper.

        Wraps the scConcept encoder for ReconEval's foundation-model
        reconstruction task. Requires the ``concept`` package
        (FlashAttention, Ampere+ GPU) and runs in the
        ``scconcept_env`` conda env.

        Args:
            checkpoint_path: Path to the pre-trained model checkpoint
                (config, model weights, gene mapping).
        """
        self.model_params = {
            'checkpoint_path': checkpoint_path,
            'emb_key': emb_key,
        }

        self.inferer = self._load_pretrained_model(checkpoint_path)
        self.emb_key = emb_key
        self.genes = None
        self.overlap_genes = None

    def _load_pretrained_model(self, checkpoint_path):
        """Load the pre-trained scConcept model."""
        concept = scConcept(cache_dir=f'{checkpoint_path}/cache/')
        config_path = f"{checkpoint_path}/config.yaml"
        model_path = f"{checkpoint_path}/model.ckpt"
        gene_mapping_path = f"{checkpoint_path}/pc_gene_token_mapping.pkl"
        concept.load_config_and_model(
                config=config_path,
                model_path=model_path,
                gene_mapping_path=gene_mapping_path,
            )
    
        return concept

    def prepare(self, adata: AnnData | None = None, **kwargs):
        """Cache the adata reference and its gene list on the wrapper."""
        if adata is not None:
            self.adata = adata
            self.genes = adata.var_names.tolist()

    def get_latent_representation(
        self, 
        X: np.ndarray|ad.AnnData
        ) -> np.ndarray:

        device = next(self.inferer.model.parameters()).device
        self.inferer.model.eval()
        
        if isinstance(X, ad.AnnData):
            temp_adata = X
        else:
            temp_adata = AnnData(X=X)
            temp_adata.var['gene_id'] = self.genes
            
        cell_embs = self.inferer.extract_embeddings(
            adata=temp_adata,
            gene_id_column='gene_id'
        )
        return cell_embs['cls_cell_emb']


    def train(self, datamodule: L.LightningDataModule = None, **train_kwargs):
        """No-op: scConcept is used frozen. Raises if a datamodule is passed."""
        if datamodule is not None:
            raise NotImplementedError(
                "Fine-tuning is not supported for the pretrained scConcept wrapper."
            )

    def set_genes(self, genes: list[str]):
        """Set the genes for reconstruction"""
        self.genes = genes

    def get_overlap_genes(self, genes) -> list[str]:
        """Not implemented for scConcept; reserved for parity with other FM wrappers."""
        raise NotImplementedError(
            "get_overlap_genes is not implemented for the scConcept wrapper."
        )


    def load(self, path: str, map_location=None) -> None:
        """Load model configuration"""
        self.inferer = self._load_pretrained_model(path)