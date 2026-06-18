from __future__ import annotations

import os
import warnings
from functools import partial
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from anndata import AnnData
from scipy import stats
from tqdm import tqdm

from sc_reconstruction.metrics.base_eval import MetricsBatchEvaluator


class CoexpressionBatchEvaluator(MetricsBatchEvaluator):
    """
    Batch evaluator for coexpression metrics.

    Modes:
        - reconstruction: x -> model.predict(x)
        - decode:        z -> model.decode(z)
        - conditional:   x -> model.decode(x, ref_batch_onehot), compare to reference batch
    """

    def __init__(
        self,
        var_names: List[str] = None,
        overlap_threshold: int = 5,
        min_cells: int = 20,
        threshold: float = 0.0,
        correlation_measure: list[str] | str = ("pearson", "spearman"),
        pipeline_output: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        var_names : List[str]
            Gene names of the *input* features in the same order as X.
        overlap_threshold : int
            Minimum overlap of genes between a gene set and common features to test a gene set.
        min_cells : int
            Minimum number of cells a gene must be expressed in.
        threshold : float
            Threshold on correlation magnitude in true data to consider a gene pair.
        correlation_measure : list[str] | str
            One or more of ['pearson', 'spearman'].
        pipeline_output : bool
            If True, only the average coexpression per gene set is used; otherwise per-gene-set
            values + average are returned from `CoexpressionCalculator`.
        **kwargs :
            Passed to `MetricsBatchEvaluator` (model, zarr_path, comb_list, mode, reference_batch, ...).
        """
        # Forward var_names into base class as well
        super().__init__(var_names=var_names, **kwargs)

        self.overlap_threshold = overlap_threshold
        self.min_cells = min_cells
        self.threshold = threshold
        if isinstance(correlation_measure, str):
            correlation_measure = [correlation_measure]
        self.correlation_measure = correlation_measure
        self.pipeline_output = pipeline_output

        self.metric_var_names: List[str] | None = (
            self.target_var_names if self.target_var_names is not None else self.var_names
        )
        if self.metric_var_names is None:
            raise ValueError(
                "CoexpressionBatchEvaluator requires `var_names` (or `target_var_names`) "
                "to map to gene-set names."
            )

        # Pre-load hallmark gene sets from MSigDB via omnipath
        self.geneset_dict = self._get_hallmark_geneset_dict()


    def _process_batch_reconstruction(self, batch: List[str]) -> List[Dict]:
        """
        Self reconstruction: x -> model.predict(x)
        """
        x_combined, comb_indices = self._load_batch_data(batch)
        recon_combined = self._run_predict(x_combined)

        x_for_metrics, recon_for_metrics = self._preprocess_for_metrics(
            x_combined, recon_combined
        )

        comb_data = []
        for comb, start, end in comb_indices:
            x_slice = x_for_metrics[start:end]
            recon_slice = recon_for_metrics[start:end]
            comb_data.append((comb, x_slice, recon_slice))

        if not comb_data:
            return []

        worker = partial(
            compute_coexpression_metrics,
            self.metric_var_names,
            self.split_key,
            self.overlap_threshold,
            self.min_cells,
            self.threshold,
            self.correlation_measure,
            self.pipeline_output,
            self.geneset_dict,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results

    def _process_batch_decode(self, batch: List[str]) -> List[Dict]:
        """
        Decode mode: z -> model.decode(z) (requires emb_zarr_path).
        """
        z_combined, x_combined, comb_indices = self._load_decode_data(batch)
        recon_combined = self._run_decode(z_combined)

        x_for_metrics, recon_for_metrics = self._preprocess_for_metrics(
            x_combined, recon_combined
        )

        comb_data = []
        for comb, start, end in comb_indices:
            x_slice = x_for_metrics[start:end]
            recon_slice = recon_for_metrics[start:end]
            comb_data.append((comb, x_slice, recon_slice))

        if not comb_data:
            return []

        worker = partial(
            compute_coexpression_metrics,
            self.metric_var_names,
            self.split_key,
            self.overlap_threshold,
            self.min_cells,
            self.threshold,
            self.correlation_measure,
            self.pipeline_output,
            self.geneset_dict,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results

    def _process_batch_conditional(self, batch: List[str]) -> List[Dict]:
        """
        Conditional / cross-batch:
            - x_combined: original data for all combinations in batch.
            - all_ref: reference batch expressions via `_load_ref_data`.
            - recon_combined: decode(x_combined, reference_batch_onehot).
        We compute coexpression metrics between reference expression and reconstructed expression.
        """
        x_combined, comb_indices = self._load_batch_data(batch)
        all_ref, ref_indices, if_ref_labels = self._load_ref_data(batch)

        ref_batch_onehot = np.tile(
            self.reference_batch_onehot, (x_combined.shape[0], 1)
        )
        recon_combined = self._run_conditional_decode(x_combined, ref_batch_onehot)

        comb_data = []
        for (
            (comb, start, end),
            (ref_comb, ref_start, ref_end),
            is_true_ref,
        ) in zip(comb_indices, ref_indices, if_ref_labels):
            if not is_true_ref:
                warnings.warn(
                    f"Reference batch {self.reference_batch} not found for combination "
                    f"{comb}. Skipping this combination."
                )
                continue

            x_ref = all_ref[ref_start:ref_end]
            recon = recon_combined[start:end]

            x_ref_for_metrics, recon_for_metrics = self._preprocess_for_metrics(
                x_ref, recon
            )

            if self.proj_out_store is not None:
                self._write_pred_to_zarr(self.reference_batch, comb, recon_for_metrics)

            comb_data.append((comb, ref_comb, x_ref_for_metrics, recon_for_metrics))

        if not comb_data:
            return []

        worker = partial(
            compute_coexpression_metrics_ref,
            self.metric_var_names,
            self.split_key,
            self.overlap_threshold,
            self.min_cells,
            self.threshold,
            self.correlation_measure,
            self.pipeline_output,
            self.geneset_dict,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results


    def _get_hallmark_geneset_dict(self) -> Dict[str, List[str]]:
        """Get Hallmark pathway gene sets from MSigDB via omnipath."""
        import omnipath as op

        anns = op.requests.Annotations.get(resources="MSigDB")
        return CoexpressionCalculator._get_msigdb_collection(anns, verbose=True)


    def run(self, output_path: str):
        # Fresh start: downstream writers append in chunks.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Existing output file {output_path} removed.")

        if self.mode == "conditional":
            base_columns = [
                "combination",
                "combination_ref",
                *self.split_key,
                "n_cells",
                "n_cells_recon",
                "method",
                "average",
            ]
        else:
            base_columns = [
                "combination",
                *self.split_key,
                "n_cells",
                "method",
                "average",
            ]

        all_columns = base_columns + sorted(self.geneset_dict.keys())

        pd.DataFrame(columns=all_columns).to_csv(output_path, index=False)

        # Process batches and append results
        batch_gen = self._dynamic_batch_generator()
        with tqdm(total=len(self.all_combs), desc="Total progress") as pbar:
            for batch in batch_gen:
                batch_results = self._process_batch(batch)
                if not batch_results:
                    pbar.update(len(batch))
                    continue
                df = pd.DataFrame(batch_results).reindex(columns=all_columns)
                df.to_csv(output_path, mode="a", header=False, index=False)
                pbar.update(len(batch))
                del batch_results

        self.pool.close()
        self.pool.join()
        print(f"Processing completed. Results saved to {output_path}")


def compute_coexpression_metrics(
        var_names: List[str],
        split_key: list[str],
        overlap_threshold: int,
        min_cells: int,
        threshold: float,
        correlation_measure: list[str],
        pipeline_output: bool,
        geneset_dict: Dict[str, List[str]],
        comb_data: tuple,
    ) -> Dict:
    """
    Wrapper function for computing coexpression metrics in parallel.
    """
    # Option A: at top of the worker function
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    comb, x, x_hat = comb_data
    records = []
    
    adata_true = AnnData(x)
    adata_recon = AnnData(x_hat)
    # Set gene names. Maybe also obs later
    adata_true.var_names = var_names
    adata_recon.var_names = var_names
    for measure in correlation_measure:
        metrics = {
            "combination": comb,
            split_key[0]: comb.split('-')[0],
            split_key[1]: comb.split('-')[1],
            split_key[2]: comb.split('-')[2],
            "n_cells": x.shape[0],
            "method": measure,
        }
        metrics.update(CoexpressionCalculator.gene_set_coexpression(
            adata_true = adata_true,
            adata_recon = adata_recon,
            overlap_threshold= overlap_threshold,
            min_cells = min_cells,
            threshold = threshold,
            correlation_measure = measure,
            pipeline_output = pipeline_output,
            geneset_dict=geneset_dict,
            )) 
        records.append(metrics)
    return records
        
def compute_coexpression_metrics_ref(
        var_names: List[str],
        split_key: list[str],
        overlap_threshold: int,
        min_cells: int,
        threshold: float,
        correlation_measure: list[str],
        pipeline_output: bool,
        geneset_dict: Dict[str, List[str]],
        comb_data: tuple,
    ) -> Dict:
    """
    Wrapper function for computing coexpression metrics in parallel for cross batch reconstruction.
    """
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    comb, ref_comb, x, x_hat = comb_data
    records = []
    
    adata_true = AnnData(x)
    adata_recon = AnnData(x_hat)
    # Set gene names. Maybe also obs later
    adata_true.var_names = var_names
    adata_recon.var_names = var_names
    
    comb_keys = comb.split('-')
    for measure in correlation_measure:
        metrics = {
            "combination": comb,
            "combination_ref": ref_comb,
            split_key[0]: comb_keys[0],
            split_key[1]: comb_keys[1] if len(comb_keys) > 1 else 'None',
            split_key[2]: comb_keys[2] if len(comb_keys) > 2 else 'None',
            "n_cells": x.shape[0],
            "n_cells_recon": x_hat.shape[0],
            "method": measure,
        }
        metrics.update(CoexpressionCalculator.gene_set_coexpression(
            adata_true = adata_true,
            adata_recon = adata_recon,
            overlap_threshold= overlap_threshold,
            min_cells = min_cells,
            threshold = threshold,
            correlation_measure = measure,
            pipeline_output = pipeline_output,
            geneset_dict=geneset_dict,
            )) 
        records.append(metrics)
    return records




# Code from txsim, slightly modified 
class CoexpressionCalculator:
    @staticmethod
    def gene_set_coexpression(
        adata_true: AnnData,
        adata_recon: AnnData,
        overlap_threshold: int = 5,
        min_cells: int = 20,
        threshold: float = 0,
        correlation_measure: str = "pearson",
        pipeline_output: bool = False,
        geneset_dict: Dict[str, List[str]] = None,
        **kwargs
    ) -> float:
        """Calculate coexpression score similarity on gene sets
        Parameters
        ----------
        spatial_data : AnnData
            annotated ``AnnData`` object with counts from spatial data
        seq_data : AnnData
            annotated ``AnnData`` object with counts scRNAseq data
        overlap_threshold : int, optional (Default: 5)
            minimal overlap of genes between a gene set and the common features bewteen
            the spatial and sequencing datasets for the gene set to be tested.
        min_cells : int, optional (Default: 20)
            The number of cells a gene must be expressed in at minimum to contribute to
            coexpression similarity score.
        correlation_measure: str, optional (Default: pearson)
            The metric used for assess the correlation of gene expression, pearson, spearman or mutual
        pipeline_output : bool, optional (Default: True)
            flag whether to output a single value (True) or results broken down into
            gene sets

        Returns
        -------
        mean : float
            mean of coexpression score  across all tested gene sets. For further details
            on the coexpression score, see the `coexpression_score` function documentation

        Future extension
        ----------------
        Calculate the score per cell type and average over that
        """
        if geneset_dict is None:
            import omnipath as op
            anns = op.requests.Annotations.get(resources='MSigDB')
            geneset_dict = CoexpressionCalculator._get_msigdb_collection(anns)

        # Filter gene sets that can be tested (gene expressed in at least X cells)
        vars_in_x_cells_st = sc.pp.filter_genes(adata_true, min_cells = min_cells, inplace=False)[0]
        true_features = adata_true.var_names[vars_in_x_cells_st].tolist()

        vars_in_x_cells_sc = sc.pp.filter_genes(adata_recon, min_cells = min_cells, inplace=False)[0]
        recon_features = adata_recon.var_names[vars_in_x_cells_sc].tolist()

        # # Common features in both modalities
        # common_features = list(set(true_features).intersection(recon_features))
        ####### For reconstruction: choose true features as common features
        common_features = true_features

        geneset_dict_filtered = {}
        for g, genes in geneset_dict.items():
            overlap = len(set(genes).intersection(common_features))
            if overlap >= overlap_threshold:
                geneset_dict_filtered[g] = genes
        
        

        gene_ids = common_features
        
        # Calculate score per gene set
        results = dict()

        for g in geneset_dict_filtered.keys():
            # print(f'Testing gene set: {g}')
            ids_in_set = [gene_ids.index(x) for x in geneset_dict_filtered[g] if x in gene_ids]
            # Remove duplicates in the gene set
            ids_in_set = list(set(ids_in_set))
            val, *_ = CoexpressionCalculator.coexpression_similarity(
                adata_true[:,ids_in_set],
                adata_recon[:,ids_in_set],
                min_cells = min_cells,
                threshold = threshold,
                correlation_measure=correlation_measure,
                pipeline_output = False
            )

            results[g] = val
 
        results['average'] = np.nanmean(list(results.values()))
        
        if not pipeline_output:
            return results
        # Aggregate co-expression outputs
        return results['average']

    @staticmethod
    def _get_msigdb_collection(anns, collection='hallmark', verbose=False):
        """
        Parser for MSigDB gene sets from omnipath
        
        Returns:
            A str: list[str] dictionary of gene set names to HGNC gene names
        
        """
        from collections import defaultdict

        """
        The 33196 gene sets in the Human Molecular Signatures Database (MSigDB) 
        are divided into 9 major collections, and several sub-collections, e.g. 
        Hallmark gene sets summarize and represent specific well-defined biological 
        states or processes and display coherent expression. 
        """
        # load `record_ids`, which is unique within the records of each database
        ids = anns[(anns.entity_type.isin(['protein'])) &
                (anns.label.isin(['collection'])) &
                (anns.value.isin([collection]))].record_id
        # load gene sets based on record ids
        collection_anns = anns[(anns.entity_type.isin(['protein'])) & 
                            (anns.label.isin(['geneset'])) &
                            (anns.record_id.isin(ids))]
        
        if verbose:
            print(f'Number of genes: {len(set(collection_anns.genesymbol))}')
            print(f'Number of annotations: {len(collection_anns.genesymbol)}')
            print(f'Number of gene sets: {collection_anns}')

        # convert the dataframe of annotation to a dictionary of gene sets
        geneset_dict = defaultdict(list)

        [geneset_dict[collection_anns['value'][i]].append(collection_anns['genesymbol'][i])
        for i in collection_anns.index];
        
        return geneset_dict

    @staticmethod
    def coexpression_similarity(
            adata_true: AnnData,
            adata_recon: AnnData,
            min_cells: int = 30,
            threshold: float = 0,
            key: str = 'celltype',
            by_celltype: bool = False,
            correlation_measure: str = "pearson",
            pipeline_output: bool = True,
    ) -> float | tuple[float, pd.Series, pd.Series]:
        """Calculate the mean difference between correlation matrices of spatial and scRNAseq data.
        
        Parameters
        ----------
        spatial_data : AnnData
            annotated ``AnnData`` object with counts from spatial data
        seq_data : AnnData
            annotated ``AnnData`` object with counts scRNAseq data
        min_cells : int, optional
            Minimum number of cells in which a gene should be detected to be considered
            expressed. By default 20. If `by_celltype` is True, the filter is applied for each cell type individually.
        thresh : float, optional
            threshold for significant pairs from scRNAseq data. Pairs with correlations
            below the threshold (by magnitude) will be ignored when calculating mean, by
            default 0
        key : str
            name of the column containing the cell type information
        by_celltype: bool
            run analysis by cell type? If False, computation will be performed using the
            whole gene expression matrix
        correlation_measure: str
            metric used to evaluate the correlation of gene expression, mutual, spearman or pearson.
            default pearson
        pipeline_output: bool, optional (default: True)
            flag whether the metric is run as part of the pipeline or not. If not, then
            coexpression similarity matrices are returned for each modality. Otherwise,
            only the mean absolute difference of the upper triangle of the coexpression
            matrices is returned as a single score.
            
        Returns
        -------
        float
            coexpression summary metric (mean of absolute correlation difference between spatial and scRNAseq data). If 
            `by_celltype` is True, the summary metric is the mean of the mean coexpression similarity per cell type.
        if pipeline_output is False also returns:
        pd.Series
            coexpression similarity per gene (mean over cell types if by_celltype is True)
        pd.Series
            coexpression similarity per cell type. (empty if by_celltype is False)
        """

        SUPPORTED_CORR = ["mutual", "pearson", "spearman"]
        assert correlation_measure in SUPPORTED_CORR, f"Invalid correlation measure {correlation_measure}"
        if correlation_measure == "mutual":
            raise NotImplementedError("Mutual information is not yet supported")

        # Reduce to shared genes
        genes_unfiltered = adata_true.var_names.intersection(adata_recon.var_names)
        
        adata_true = adata_true[:, genes_unfiltered].copy()
        adata_recon = adata_recon[:, genes_unfiltered].copy()

        # Filter genes based on number of cells that express them
        genes_true = adata_true.var_names[sc.pp.filter_genes(adata_true, min_cells=min_cells, inplace=False)[0]]
        genes_recon = adata_recon.var_names[sc.pp.filter_genes(adata_recon, min_cells=min_cells, inplace=False)[0]]

        genes = genes_true
        
        if (not len(genes)):
            # print("No expressed genes present")
            if pipeline_output:
                return np.nan 
            else: 
                return [np.nan, pd.Series(index=genes_unfiltered, data=np.nan), pd.Series(dtype=float)]

        if not by_celltype:
            # Get matrices for commonly expressed genes
            mat_true = adata_true[:, genes].X
            mat_recon = adata_recon[:, genes].X

            # Calculate coexpression similarity matrix
            coexp_sim_mat = CoexpressionCalculator.coexpression_similarity_matrix(mat_true, mat_recon, threshold, correlation_measure)
            
            # Calculate summary metric
            coexp_sim = np.nanmean(coexp_sim_mat[np.triu_indices(len(coexp_sim_mat), k=1)])
            
            if pipeline_output:
                return coexp_sim
            else:
                return [coexp_sim, pd.Series(index=genes, data=np.nanmean(coexp_sim_mat, axis=1)), pd.Series(dtype=float)]
        else:        
            # Get shared cell types
            shared_cts = list(set(adata_true.obs[key].unique()).intersection(set(adata_recon.obs[key].unique())))
            
            # Init coexpression similarity matrices
            coexp_sim_matrices = np.zeros((len(shared_cts), len(genes), len(genes)), dtype=float)
            coexp_sim_matrices[:,:,:] = np.nan

            for ct_idx, ct in enumerate(shared_cts):
                # Adatas for the given cell type
                adata_true_ct = adata_true[adata_true.obs[key] == ct, :]
                adata_recon_ct = adata_recon[adata_recon.obs[key] == ct, :]

                # Filter genes based on number of cells that express them
                genes_true_ct = adata_true_ct.var_names[sc.pp.filter_genes(adata_true_ct, min_cells=min_cells, inplace=False)[0]]
                genes_recon_ct = adata_recon_ct.var_names[sc.pp.filter_genes(adata_recon_ct, min_cells=min_cells, inplace=False)[0]]
                
                # Get common genes
                genes_ct = genes_true_ct.intersection(genes_recon_ct)
                genes_mask = np.isin(genes, genes_ct)
                
                # Skip cell type if no expressed genes are shared
                if len(genes_ct) == 0:
                    print(f"No expressed genes are shared in both modalities for cell type {ct}")
                    continue
                    
                # Get matrices for commonly expressed genes
                mat_true = genes_true_ct[:, genes_ct].X
                mat_recon = adata_recon_ct[:, genes_ct].X
                
                # Calculate coexpression similarity matrix
                coexp_sim_mat = CoexpressionCalculator.coexpression_similarity_matrix(mat_true, mat_recon, threshold, correlation_measure)
                coexp_sim_matrices[ct_idx][np.ix_(genes_mask, genes_mask)] = coexp_sim_mat
                
            coexp_sim_per_ct = pd.Series(index=shared_cts, data=np.nan)
            for ct_idx, ct in enumerate(shared_cts):
                coexp_sim_mat = coexp_sim_matrices[ct_idx, :, :]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    coexp_sim_per_ct.loc[ct] = np.nanmean(coexp_sim_mat[np.triu_indices(len(coexp_sim_mat), k=1)])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                coexp_sim_mat_ct_avg = np.nanmean(coexp_sim_matrices, axis=0)
                coexp_sim_per_gene = pd.Series(index=genes, data=np.nanmean(coexp_sim_mat_ct_avg, axis=0))
            
            # Calculate summary metric
            coexp_sim = coexp_sim_per_ct.mean()
            
            if pipeline_output:
                return coexp_sim
            else:
                return [coexp_sim, coexp_sim_per_gene, coexp_sim_per_ct]

    @staticmethod
    def coexpression_similarity_matrix(
            mat_1: np.ndarray | scipy.sparse.csr_matrix, 
            mat_2: np.ndarray | scipy.sparse.csr_matrix, 
            threshold: float, 
            correlation_measure: str
    ) -> np.ndarray:
        """ Calculate the coexpression similarity matrix between spatial and scRNAseq data.
        
        The treshold to filter out pairs of low correlations is only applied to the scRNAseq data. The reasoning is that 
        scRNAseq should have less correlation artifacts and is understood as the "ground truth".
        
        Parameters
        ----------
        mat_1 : np.ndarray | scipy.sparse.csr_matrix
        mat_2 : np.ndarray | scipy.sparse.csr_matrix
        thresh : float
            threshold for significant pairs from scRNAseq data. Pairs with correlations below the threshold (by magnitude) 
            will be set to NaN.
        correlation_measure : str
            Metric for the correlation measure refering to coexpression. Supported are "pearson" and "spearman".
            
        Returns
        -------
        np.ndarray
            Coexpression similarity matrix (1 - absolute difference of the correlation matrices/2 of spatial and scRNAseq 
            data). Diagonal is set to NaN.
        """
        
        # Calculate correlation matrices
        if correlation_measure == 'pearson':
            coexp_1 = CoexpressionCalculator.get_pearson_correlation_matrix(mat_1)
            coexp_2 = CoexpressionCalculator.get_pearson_correlation_matrix(mat_2)
        elif correlation_measure == 'spearman':
            coexp_1 = CoexpressionCalculator.get_spearman_correlation_matrix(mat_1)
            coexp_2 = CoexpressionCalculator.get_spearman_correlation_matrix(mat_2)
            
        # Gene pair filter based on true data
        mask = np.abs(coexp_1) < threshold
            
        # Calculate difference between modalities
        coexp_diff_matrix = np.abs(coexp_1 - coexp_2)
        coexp_diff_matrix[mask] = np.nan
        
        # Transform to similarity matrix
        coexp_sim_matrix = 1 - coexp_diff_matrix / 2
        
        # Set diagonal to NaN
        np.fill_diagonal(coexp_sim_matrix, np.nan)
        
        return coexp_sim_matrix
        
    @staticmethod
    def get_pearson_correlation_matrix(mat: np.ndarray | scipy.sparse.csr_matrix) -> np.ndarray:
        """ Calculate the Pearson correlation matrix of the input matrix.
        
        Parameters
        ----------
        mat : np.ndarray | scipy.sparse.csr_matrix
            input matrix (e.g. adata.X)
            
        Returns
        -------
        np.ndarray
            Pearson correlation matrix of genes
        """
        mat_ = mat.toarray().astype(float) if scipy.sparse.issparse(mat) else mat.astype(float)
        if mat_.shape[1] > 1:
            return np.corrcoef(mat_, rowvar=False)
        else:
            return np.ones((1,1), dtype=np.float32)

    @staticmethod
    def get_spearman_correlation_matrix(mat: np.ndarray | scipy.sparse.csr_matrix) -> np.ndarray:
        """ Calculate the Spearman correlation
        
        Parameters
        ----------
        mat : np.ndarray | scipy.sparse.csr_matrix
            input matrix (e.g. adata.X)
            
        Returns
        -------
        np.ndarray
            Spearman correlation matrix of genes
        """
        mat_ = mat.toarray().astype(float) if scipy.sparse.issparse(mat) else mat.astype(float)
        
        if mat.shape[1] > 2: # spearman automatically returns a matrix for 3 and more genes
            try:
                return stats.spearmanr(mat_).correlation.astype(np.float32)
            except:
                noise = np.random.normal(0, 1e-6, mat_.shape)
                mat_noisy = mat_ + noise
                corr = stats.spearmanr(mat_noisy).correlation.astype(np.float32)
                return corr
        elif mat.shape[1] == 2: # still get matrix output for 2 or 1 genes
            spearmanr = np.ones((mat.shape[1], mat.shape[1]), dtype=np.float32)
            spearmanr[0,1] = stats.spearmanr(mat_[:,0], mat_[:,1]).statistic
            spearmanr[1,0] = spearmanr[0,1]
            return spearmanr
        else:
            return np.ones((1,1), dtype=np.float32)