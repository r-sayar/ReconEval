from __future__ import annotations

import scanpy as sc
import numpy as np
import scipy.sparse as sp
import warnings
from pathlib import Path
import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from anndata import AnnData
import anndata as ad


class ReconPretrainedscGPT():
    def __init__(
        self,
        checkpoint_path: str,
        gene_col: str = 'var_names',
        **kwargs
    ):
        """Pre-trained scGPT foundation model :cite:`cui:24scgpt`.

        Wraps the scGPT encoder for ReconEval's foundation-model
        reconstruction task. Requires the ``scgpt`` package and runs
        in the ``scgpt`` conda env.

        Args:
            checkpoint_path: Path to the pre-trained model checkpoint.
        """
        self.checkpoint_path = checkpoint_path
        self.gene_col = gene_col
        self.genes = None
        self.overlap_genes = None

        self._vocab = GeneVocab.from_file(Path(checkpoint_path) / "vocab.json")

    def prepare(self, adata: AnnData | None = None, **kwargs):
        """Cache the adata reference and its gene list on the wrapper."""
        if adata is not None:
            self.adata = adata
            self.genes = adata.var_names.tolist()

    def get_latent_representation(
        self,
        X: np.ndarray | ad.AnnData
    ) -> np.ndarray:
        if isinstance(X, ad.AnnData):
            temp_adata = X
        else:
            temp_adata = AnnData(X=X)
            temp_adata.var[self.gene_col] = self.genes

        if self.gene_col in ("var_names", "index"):
            gene_names = temp_adata.var_names.tolist()
        else:
            gene_names = temp_adata.var[self.gene_col].tolist()
        vocab_mask = np.array([g in self._vocab for g in gene_names])

        mat = temp_adata.X
        if sp.issparse(mat):
            vocab_nnz = np.diff(sp.csc_matrix(mat[:, vocab_mask]).tocsr().indptr)
        else:
            vocab_nnz = np.count_nonzero(mat[:, vocab_mask], axis=1)
        has_expr = vocab_nnz > 0

        if not has_expr.all():
            n_skip = int((~has_expr).sum())
            warnings.warn(
                f"scGPT: {n_skip}/{temp_adata.n_obs} cells have zero expression "
                f"in all vocab genes — they will get zero embeddings."
            )

        if not has_expr.any():
            raise ValueError("All cells have zero expression in scGPT vocab genes")

        if has_expr.all():
            return scg.tasks.embed_data(
                temp_adata,
                self.checkpoint_path,
                gene_col=self.gene_col,
                batch_size=64,
                return_new_adata=True,
            ).X

        result = scg.tasks.embed_data(
            temp_adata[has_expr].copy(),
            self.checkpoint_path,
            gene_col=self.gene_col,
            batch_size=64,
            return_new_adata=True,
        ).X

        full = np.zeros((temp_adata.n_obs, result.shape[1]), dtype=np.float32)
        full[has_expr] = result
        return full

    def set_genes(self, genes: list[str]):
        """Set the genes for reconstruction"""
        self.genes = genes