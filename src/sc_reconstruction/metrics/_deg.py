from __future__ import annotations

import os
import warnings
from functools import partial
from itertools import chain
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import zarr
from tqdm import tqdm

from .utils import compute_pearson, compute_spearman
from sc_reconstruction.metrics.base_eval import MetricsBatchEvaluator


def get_reference_cluster(root_zarr: zarr.Group, comb: str, dataset: str) -> str | None:
    """Resolve a combination string to its reference (control) cluster key.

    Used by the paper's batch pipelines for the three benchmark datasets
    (tahoe, luca, pbmc). Each dataset encodes the control condition in the
    combination string differently:

      - tahoe : ``{cell_line}-DMSO_TF-0``
      - luca  : ``{cell_line}-{tissue}-normal_adjacent`` (fallback ``-normal``)
      - pbmc  : ``{cell_type}-{donor}-PBS``

    Returns ``None`` if the combination is already the control (e.g. DMSO in
    tahoe) or if no reference is found in ``root_zarr``.
    """
    split_level_1 = comb.split("-")[0]
    split_level_2 = comb.split("-")[1]

    if "tahoe" in dataset:
        ref_key = split_level_1 + "-DMSO_TF-0"
        if split_level_2 == "DMSO_TF":
            return None
    elif "luca" in dataset:
        ref_key = split_level_1 + "-" + split_level_2 + "-normal_adjacent"
    elif "pbmc" in dataset:
        ref_key = split_level_1 + "-" + split_level_2 + "-PBS"
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    if ref_key in root_zarr:
        return ref_key

    fallback = split_level_1 + "-" + split_level_2 + "-normal"
    if fallback in root_zarr:
        return fallback

    warnings.warn(f"Reference key {ref_key} for {comb} not found in zarr")
    return None


class DegBatchEvaluator(MetricsBatchEvaluator):
    """
    Evaluate DEG-based metrics on reconstructed vs true treatment / control clusters.
    """

    def __init__(self,
                 var_names: List[str] = None,
                 dice_k: List[int] = [10, 50, 100, 200, 400],
                 methods: List[str] = ['t-test', 'wilcoxon'],
                 min_cells: int = 30,
                 set_neg_to_zero: bool = True,
                 compute_mean_diff: bool = True,
                 compute_topk_corr: bool = False,
                 sampling_down: bool = False,
                 fdr_threshold: float | None = 0.05,
                 **kwargs):

        super().__init__(**kwargs)

        self.var_names = var_names
        self.dice_k = dice_k
        self.methods = methods
        self.min_cells = min_cells
        self.set_neg_to_zero = set_neg_to_zero
        self.compute_mean_diff = compute_mean_diff
        self.compute_topk_corr = compute_topk_corr
        self.sampling_down = sampling_down
        self.fdr_threshold = fdr_threshold


    def _process_batch_decode(self, batch: List[str]) -> List[Dict]:
        comb_data = []

        for comb in batch:
            refer_comb = get_reference_cluster(self.root_zarr, comb, self.dataset)
            if refer_comb is None:
                warnings.warn(f"Reference cluster for {comb} not found, skipping.")
                continue

            z = self.emb_root_zarr[comb][self.input_key][:]
            x = self.root_zarr[comb][self.output_key][:]

            refer_z = self.emb_root_zarr[refer_comb][self.input_key][:]
            refer_x = self.root_zarr[refer_comb][self.output_key][:]

            x_hat = self._run_decode(z)
            refer_hat = self._run_decode(refer_z)

            x_for_metrics, x_hat_for_metrics = self._preprocess_for_metrics(x, x_hat)
            refer_for_metrics, refer_hat_for_metrics = self._preprocess_for_metrics(refer_x, refer_hat)

            comb_data.append((comb, x_for_metrics, x_hat_for_metrics, refer_for_metrics, refer_hat_for_metrics))

        if not comb_data:
            return []

        worker = partial(
            compute_deg_metrics,
            self.min_cells,
            self.methods,
            self.split_key,
            self.set_neg_to_zero,
            self.dice_k,
            self.compute_mean_diff,
            self.compute_topk_corr,
            self.fdr_threshold,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results


    def _process_batch_reconstruction(self, batch: List[str]) -> List[Dict]:
        comb_data = []

        for comb in batch:
            refer_comb = get_reference_cluster(self.root_zarr, comb, self.dataset)
            if refer_comb is None:
                warnings.warn(f"Reference cluster for {comb} not found, skipping.")
                continue

            x = self.root_zarr[comb][self.output_key][:]
            refer_x = self.root_zarr[refer_comb][self.output_key][:]

            x_hat = self._run_predict(x)
            refer_hat = self._run_predict(refer_x)

            x_for_metrics, x_hat_for_metrics = self._preprocess_for_metrics(x, x_hat)
            refer_for_metrics, refer_hat_for_metrics = self._preprocess_for_metrics(refer_x, refer_hat)

            comb_data.append((comb, x_for_metrics, x_hat_for_metrics, refer_for_metrics, refer_hat_for_metrics))

        if not comb_data:
            return []

        worker = partial(
            compute_deg_metrics,
            self.min_cells,
            self.methods,
            self.split_key,
            self.set_neg_to_zero,
            self.dice_k,
            self.compute_mean_diff,
            self.compute_topk_corr,
            self.fdr_threshold,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results

    def run(self, output_path: str):
        # header
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Existing output file {output_path} removed.")

        base_columns = ["combination", *self.split_key, "n_cells", "n_cells_ctrl", "method"]
        metric_columns: List[str] = []

        for prefix in ['true', 'pred']:
            metric_columns.extend([f"{prefix}_deg_dice_{k}" for k in self.dice_k])
            metric_columns.extend([f"{prefix}_deg_pearson", f"{prefix}_deg_spearman"])
            if self.compute_topk_corr:
                metric_columns.extend([f"{prefix}_deg_pearson_orig_{k}" for k in self.dice_k])
                metric_columns.extend([f"{prefix}_deg_spearman_orig_{k}" for k in self.dice_k])
            if self.compute_mean_diff:
                metric_columns.extend(
                    [f"{prefix}_mean_genediff_pearson", f"{prefix}_mean_genediff_spearman"]
                )

        all_columns = base_columns + metric_columns
        pd.DataFrame(columns=all_columns).to_csv(output_path, index=False)

        # Process batches and append results
        batch_gen = self._dynamic_batch_generator()
        with tqdm(total=len(self.all_combs), desc="Total progress") as pbar:
            for batch in batch_gen:
                if self.mode == "reconstruction":
                    batch_results = self._process_batch_reconstruction(batch)
                elif self.mode == "decode":
                    batch_results = self._process_batch_decode(batch)
                else:
                    raise ValueError(
                        f"DegBatchEvaluator only supports 'reconstruction' / 'decode' modes, "
                        f"got '{self.mode}'."
                    )

                if len(batch_results) > 0:
                    pd.DataFrame(batch_results).to_csv(
                        output_path, mode='a', header=False, index=False
                    )
                pbar.update(len(batch))
                del batch_results

        self.pool.close()
        self.pool.join()
        print(f"Processing completed. Results saved to {output_path}")



def compute_deg_metrics(
        min_cells: int,
        methods: list[str],
        split_key: list[str],
        set_neg_to_zero: bool,
        dice_k: list[int],
        compute_mean_diff: bool,
        compute_topk_corr: bool,
        fdr_threshold: float | None,
        comb_data: tuple,
    ) -> Dict:
    '''
    Wrapper function to compute DEG metrics for a given combination of cell line, drug, and dosage.

    Args:
        comb: str, combination name
        x: np.ndarray, treatment data
        x_hat: np.ndarray, predicted treatment data
        refer: np.ndarray, reference data
        refer_hat: np.ndarray, predicted reference data
        methods: List of methods to compute DEG metrics for.
        set_neg_to_zero: Whether to set negative values to zero.
        dice_k: List of k values to compute DEG metrics for.
        compute_mean_diff: Whether to compute mean difference metrics.
        compute_topk_corr: Whether to compute topk correlation metrics.

    Returns:
        Dict containing the DEG metrics for the given combination.
    '''
    comb, x, x_hat, refer, refer_hat = comb_data
    records = []
    for method in methods:
        metrics = {
            "combination": comb,
            split_key[0]: comb.split('-')[0],
            split_key[1]: comb.split('-')[1],
            split_key[2]: comb.split('-')[2],
            "n_cells": x.shape[0],
            "n_cells_ctrl": refer.shape[0],
            "method": method,
        }
        metrics.update(DegCalculator.compute_deg(x = x,
                                                 x_hat = x_hat,
                                                 refer = refer,
                                                 refer_hat = refer_hat,
                                                 method = method,
                                                 min_cells = min_cells,
                                                 set_neg_to_zero = set_neg_to_zero,
                                                 dice_k = dice_k,
                                                 compute_mean_diff = compute_mean_diff,
                                                 compute_topk_corr = compute_topk_corr,
                                                 fdr_threshold = fdr_threshold,
                                                 ))
        records.append(metrics)
    return records

class DegCalculator:
    """
    Helper class to compute differential expression (DEG) metrics.
    Provides static methods to compute DEG metrics such as Dice coefficient,
    Pearson and Spearman correlations for given treatment and reference datasets.
    """

    @staticmethod
    def compute_deg(x: np.ndarray,
                    x_hat: np.ndarray,
                    refer: np.ndarray,
                    refer_hat: np.ndarray,
                    method: str,
                    min_cells: int,
                    set_neg_to_zero: bool,
                    dice_k: list[int],
                    compute_mean_diff: bool,
                    compute_topk_corr: bool,
                    fdr_threshold: float | None = 0.05,
                    ) -> Dict:

        if set_neg_to_zero:
            x_hat = np.clip(x_hat, 0, None)
            refer_hat = np.clip(refer_hat, 0, None)

        n_treat = x.shape[0]
        n_ctrl = refer.shape[0]

        if n_treat < min_cells or n_ctrl < min_cells:
            return DegCalculator._null_metrics(dice_k, compute_mean_diff, compute_topk_corr)
        
        orig_deg = DegCalculator.perform_deg(x, refer, method)
        recon_deg = DegCalculator.perform_deg(x_hat, refer, method)
        recon_deg_hat = DegCalculator.perform_deg(x_hat, refer_hat, method)

        metrics_true = DegCalculator._deg_metrics(
            orig_deg, recon_deg, dice_k,
            compute_mean_diff, compute_topk_corr,
            x, refer, x_hat, refer, fdr_threshold,
        )
        metrics_hat = DegCalculator._deg_metrics(
            orig_deg, recon_deg_hat, dice_k,
            compute_mean_diff, compute_topk_corr,
            x, refer, x_hat, refer_hat, fdr_threshold,
        )
        
        return {
            **{f"true_{k}": v for k, v in metrics_true.items()},
            **{f"pred_{k}": v for k, v in metrics_hat.items()}
        }


    @staticmethod
    def perform_deg(x: np.ndarray, refer: np.ndarray, method: str) -> pd.DataFrame:
        adata = ad.AnnData(np.vstack([x, refer]))
        adata.obs['group'] = ['treatment'] * x.shape[0] + ['control'] * refer.shape[0]
        
        sc.tl.rank_genes_groups(
            adata, 
            groupby='group', 
            groups=['treatment'],
            reference='control',
            method=method,
            use_raw=False  
        )
        
        return pd.DataFrame({
            'gene': adata.uns['rank_genes_groups']['names']['treatment'],
            'scores': adata.uns['rank_genes_groups']['scores']['treatment'],
            'logfc': adata.uns['rank_genes_groups']['logfoldchanges']['treatment'],
            'pval_adj': adata.uns['rank_genes_groups']['pvals_adj']['treatment']
        })
    
    @staticmethod
    def _deg_metrics(
        orig_deg: pd.DataFrame,
        recon_deg: pd.DataFrame,
        dice_k: list[int],
        compute_mean_diff: bool,
        compute_topk_corr: bool,
        x: np.ndarray,
        refer: np.ndarray,
        x_hat: np.ndarray,
        recon_refer: np.ndarray,
        fdr_threshold: float | None = 0.05,
    ) -> Dict[str, float]:
        metrics = {}
        metrics.update(DegCalculator._deg_compute_dice(orig_deg, recon_deg, dice_k, fdr_threshold))

        merged = DegCalculator._merge_deg_results(orig_deg, recon_deg)

        metrics.update(DegCalculator._compute_logfc_corr(merged))

        if compute_topk_corr:
            metrics.update(DegCalculator._compute_topk_corr(merged, orig_deg, dice_k, fdr_threshold))

        if compute_mean_diff:
            metrics.update(DegCalculator._compute_mean_diff_corr(x.mean(axis=0)-refer.mean(axis=0), x_hat.mean(axis=0)-recon_refer.mean(axis=0)))

        return metrics

    
    @staticmethod
    def _merge_deg_results(
        orig_deg: pd.DataFrame, 
        recon_deg: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.merge(
            orig_deg[['gene', 'logfc', 'pval_adj']], 
            recon_deg[['gene', 'logfc', 'pval_adj']], 
            on='gene', 
            suffixes=('_orig', '_recon'),
            how='inner'
        )
                       
    @staticmethod
    def _top_genes_by_logfc(deg: pd.DataFrame, k: int, fdr_threshold: float | None) -> set:
        """Return top-k genes ranked by |logfc|, optionally pre-filtered by FDR."""
        candidates = deg if fdr_threshold is None else deg[deg['pval_adj'] <= fdr_threshold]
        if candidates.empty:
            candidates = deg  # fall back to all genes if filter removes everything
        candidates = candidates.copy()
        candidates['abs_logfc'] = candidates['logfc'].abs()
        return set(candidates.nlargest(k, 'abs_logfc')['gene'])

    @staticmethod
    def _deg_compute_dice(
        orig_deg: pd.DataFrame,
        recon_deg: pd.DataFrame,
        dice_k: list[int],
        fdr_threshold: float | None = 0.05,
    ) -> Dict[str, float]:
        dice_dict = {}
        for k in dice_k:
            orig_genes  = DegCalculator._top_genes_by_logfc(orig_deg,  k, fdr_threshold)
            recon_genes = DegCalculator._top_genes_by_logfc(recon_deg, k, fdr_threshold)
            overlap = orig_genes & recon_genes
            dice = 2 * len(overlap) / (len(orig_genes) + len(recon_genes))
            dice_dict[f'deg_dice_{k}'] = dice
        return dice_dict

    @staticmethod
    def _compute_logfc_corr(merged: pd.DataFrame) -> Dict[str, float]:
        if len(merged) < 2:
            return {
                'deg_pearson': float('nan'),
                'deg_spearman': float('nan')
            }
        return {
            'deg_pearson': compute_pearson(merged['logfc_orig'], merged['logfc_recon']),
            'deg_spearman': compute_spearman(merged['logfc_orig'], merged['logfc_recon'])
        }

    @staticmethod
    def _compute_topk_corr(
        merged: pd.DataFrame,
        orig_deg: pd.DataFrame,
        dice_k: List[int],
        fdr_threshold: float | None = 0.05,
    ) -> Dict[str, float]:
        results = {}
        for k in dice_k:
            top_genes = DegCalculator._top_genes_by_logfc(orig_deg, k, fdr_threshold)
            subset = merged[merged['gene'].isin(top_genes)]
            if len(subset) < 2:
                results.update({
                    f'deg_pearson_orig_{k}': float('nan'),
                    f'deg_spearman_orig_{k}': float('nan')
                })
            else:
                results.update({
                    f'deg_pearson_orig_{k}': compute_pearson(subset['logfc_orig'], subset['logfc_recon']),
                    f'deg_spearman_orig_{k}': compute_spearman(subset['logfc_orig'], subset['logfc_recon'])
                })
        return results

    @staticmethod
    def _compute_mean_diff_corr(
        true_diff: np.ndarray, 
        recon_diff: np.ndarray
    ) -> Dict[str, float]:
        if len(true_diff) < 2 or len(recon_diff) < 2:
            return {
                'mean_genediff_pearson': float('nan'),
                'mean_genediff_spearman': float('nan')
            }
        return {
            'mean_genediff_pearson': compute_pearson(true_diff, recon_diff),
            'mean_genediff_spearman': compute_spearman(true_diff, recon_diff)
        }

    @staticmethod
    def _null_metrics(
        dice_k: List[int],
        compute_mean_diff: bool,
        compute_topk_corr: bool,
    ) -> Dict[str, float]:
        """All-NaN result, used when fewer than `min_cells` cells are available."""
        base_metrics = [f"deg_dice_{k}" for k in dice_k]
        base_metrics += ["deg_pearson", "deg_spearman"]
        if compute_topk_corr:
            for k in dice_k:
                base_metrics.extend([f"deg_pearson_orig_{k}", f"deg_spearman_orig_{k}"])
        if compute_mean_diff:
            base_metrics.extend(["mean_genediff_pearson", "mean_genediff_spearman"])

        return {
            f"{prefix}_{name}": float("nan")
            for prefix in ("true", "pred")
            for name in base_metrics
        }
