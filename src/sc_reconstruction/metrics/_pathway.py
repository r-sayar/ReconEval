import os
import warnings
from functools import partial
from itertools import chain
from typing import Dict, List, Tuple

import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import stats
from tqdm import tqdm

from sc_reconstruction.metrics.base_eval import MetricsBatchEvaluator


class PathwayBatchEvaluator(MetricsBatchEvaluator):
    """
    Batch evaluator for pathway activity metrics (PROGENy-based).

        P = {P_1, ..., P_M} (PROGENy pathways)
        S^{P_i} = ULM(X, G_i)       (pathway score vector over cells)
        r_i     = corr(S^{P_i}, Ŝ^{P_i})
        S       = mean_{i in significant pathways}(r_i)
    """

    def __init__(
        self,
        var_names: List[str] = None,
        overlap_threshold: int = 5,
        min_cells: int = 30,
        correlation_measure: list[str] | str = ("pearson", "spearman"),
        pipeline_output: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        var_names : List[str]
            Gene names of the input features in the same order as X.
        overlap_threshold : int
            Minimum gene overlap between pathway gene set and common genes to test pathway.
        min_cells : int
            Minimum number of cells a gene must be expressed in.
        correlation_measure : list[str] | str
            One or more of ["pearson", "spearman"].
        pipeline_output : bool
            If True, return only a scalar average; otherwise per-pathway dict + "average".
        **kwargs :
            Passed to `MetricsBatchEvaluator` (model, zarr_path, comb_list, mode, etc.).
        """
        # Forward var_names into base class so filtering stays consistent
        super().__init__(var_names=var_names, **kwargs)

        self.overlap_threshold = overlap_threshold
        self.min_cells = min_cells
        if isinstance(correlation_measure, str):
            correlation_measure = [correlation_measure]
        self.correlation_measure = correlation_measure
        self.pipeline_output = pipeline_output

        # For metrics, use target_var_names if set (post-filter), otherwise var_names
        self.metric_var_names: List[str] | None = (
            self.target_var_names if self.target_var_names is not None else self.var_names
        )
        if self.metric_var_names is None:
            raise ValueError(
                "PathwayBatchEvaluator requires `var_names` (or `target_var_names`) "
                "to map pathway genes."
            )

        self.progeny_model, self.pathway_dict = self._get_progeny_dict()

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
            compute_pathway_metrics,
            self.metric_var_names,
            self.split_key,
            self.overlap_threshold,
            self.min_cells,
            self.correlation_measure,
            self.pipeline_output,
            self.progeny_model,
            self.pathway_dict,
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
            compute_pathway_metrics,
            self.metric_var_names,
            self.split_key,
            self.overlap_threshold,
            self.min_cells,
            self.correlation_measure,
            self.pipeline_output,
            self.progeny_model,
            self.pathway_dict,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results

    def _process_batch_conditional(self, batch: List[str]) -> List[Dict]:
        """
        Conditional / cross-batch:
            - x_combined: original data for all combinations in batch.
            - all_ref: reference batch expression (via `_load_ref_data`).
            - recon_combined: decode(x_combined, reference_batch_onehot).
        Pathway similarity is computed between reference expression and reconstructed expression.
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
            compute_pathway_metrics_ref,
            self.metric_var_names,
            self.split_key,
            self.overlap_threshold,
            self.min_cells,
            self.correlation_measure,
            self.pipeline_output,
            self.progeny_model,
            self.pathway_dict,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results


    def _get_progeny_dict(self) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Returns (progeny_model, pathway_dict) for human PROGENy.
        """
        return PathwayCalculator._get_progeny_dict()

    def run(self, output_path: str):
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

        all_columns = base_columns + sorted(self.pathway_dict.keys())
        pd.DataFrame(columns=all_columns).to_csv(output_path, index=False)

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



def compute_pathway_metrics(
        var_names: List[str],
        split_key: list[str],
        overlap_threshold: int,
        min_cells: int,
        correlation_measure: list[str]|str,
        pipeline_output: bool,
        progeny_model: pd.DataFrame,  
        pathway_dict: Dict[str, List[str]],
        comb_data: tuple,
    ) -> Dict:
    """
    Wrapper function for computing coexpression metrics in parallel.
    """
    comb, x, x_hat = comb_data
    records = []
    
    adata_true = AnnData(x)
    adata_recon = AnnData(x_hat)
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
        try:
            metrics.update(PathwayCalculator.pathway_score_similarity(
                adata_true = adata_true,
                adata_recon = adata_recon,
                overlap_threshold= overlap_threshold,
                min_cells = min_cells,
                correlation_measure = measure,
                pipeline_output = pipeline_output,
                progeny_model=progeny_model,
                pathway_dict=pathway_dict, 
                )) 
        except Exception as e:
            warnings.warn(f"Failed to compute pathway metrics for combination {comb}: {e}")
            metrics.update({pathway: np.nan for pathway in pathway_dict.keys()})
            metrics['average'] = np.nan
        records.append(metrics)
    return records
        


class PathwayCalculator:
    @staticmethod
    def pathway_score_similarity(
        adata_true: AnnData,
        adata_recon: AnnData,
        min_cells: int = 30,
        overlap_threshold: int = 5,
        correlation_measure: str = "pearson",
        pipeline_output: bool = False,
        progeny_model: pd.DataFrame = None,  
        pathway_dict: Dict[str, List[str]] = None, 
        **kwargs
    ) -> float:
        """
        Calculate pathway activity score similarity between true and reconstructed data
        
        Parameters
        ----------
        adata_true : AnnData
            Ground truth AnnData object
        adata_recon : AnnData
            Reconstructed AnnData object
        pval_threshold : float, optional (Default: 1, not filetering)
            P-value threshold for significant pathway activities
        correlation_measure : str, optional (Default: "pearson")
            Correlation metric: "pearson", "spearman"
        
        Returns
        -------
        similarity_score : float
            Mean correlation of pathway activity vectors (only significant pathways)
        """
        if progeny_model is None or pathway_dict is None:
            progeny_model, pathway_dict = PathwayCalculator._get_progeny_dict()


        genes_true = adata_true.var_names[sc.pp.filter_genes(adata_true, min_cells=min_cells, inplace=False)[0]]
        genes_recon = adata_recon.var_names[sc.pp.filter_genes(adata_recon, min_cells=min_cells, inplace=False)[0]]
        # common_genes = genes_true.intersection(genes_recon)
        #### For reconstruction: select genes from true data 
        common_genes = genes_true

        adata_true = adata_true[:, common_genes]
        adata_recon = adata_recon[:, common_genes]

        valid_pathways = []
        for pathway, genes in pathway_dict.items():
            overlap = len(set(genes) & set(common_genes))
            if overlap >= overlap_threshold:
                valid_pathways.append(pathway)

        if not valid_pathways:
            warnings.warn(f"No pathways passed overlap threshold ({overlap_threshold})")
            return np.nan if pipeline_output else {}

        filtered_model = progeny_model[progeny_model['source'].isin(valid_pathways)]


        adata_true = PathwayCalculator._run_progeny(adata_true, filtered_model)
        adata_recon = PathwayCalculator._run_progeny(adata_recon, filtered_model)
        
        true_scores = adata_true.obsm['score_ulm']
        recon_scores = adata_recon.obsm['score_ulm']

        # Per-pathway correlation across cells
        if correlation_measure == "pearson":
            corr = np.array([
                stats.pearsonr(true_scores[c], recon_scores[c]).statistic
                for c in true_scores.columns
            ])
        elif correlation_measure == "spearman":
            corr = np.array([
                stats.spearmanr(true_scores[c], recon_scores[c]).statistic
                for c in true_scores.columns
            ])
        else:
            raise ValueError(f"Unsupported correlation: {correlation_measure}")

        pathway_corrs = dict(zip(true_scores.columns, corr))
        pathway_corrs['average'] = float(np.nanmean(corr))

        if not pipeline_output:
            return pathway_corrs
        # Aggregate
        return pathway_corrs['average']

    @staticmethod
    def _get_progeny_dict(organism="human") -> tuple:
        """Get PROGENy model and pathway-gene dictionary"""
        progeny_model = dc.op.progeny(organism=organism)
        pathway_dict = {}
        for pathway in progeny_model['source'].unique():
            genes = progeny_model[progeny_model['source'] == pathway]['target'].tolist()
            pathway_dict[pathway] = genes
        return progeny_model, pathway_dict
    

    @staticmethod
    def _run_progeny(adata: AnnData, progeny_model: pd.DataFrame) -> AnnData:
        """Run PROGENy and store results in obsm"""
        adata = adata.copy()
        dc.mt.ulm(
            adata,
            net=progeny_model,
        )
        return adata
