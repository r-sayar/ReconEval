import os
import warnings
from functools import partial
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm

from sc_reconstruction.metrics.base_eval import MetricsBatchEvaluator


class CellCycleBatchEvaluator(MetricsBatchEvaluator):
    """
    Evaluate cell-cycle based metrics on reconstructed vs true clusters.

    Modes:
        - reconstruction: x -> model.predict(x)
        - decode:        z -> model.decode(z)
        - conditional:   x -> model.decode(x, ref_batch_onehot), compare to reference batch
    """

    def __init__(
        self,
        cell_cycle_genes_path: str,
        min_cells: int = 20,
        **kwargs,
    ):
        """
        Parameters
        ----------
        cell_cycle_genes_path : str
            Path to a text file containing one cell-cycle gene per line.
            First 43 are S-phase genes, rest are G2M.
        min_cells : int
            Minimum number of cells per gene for inclusion.
        **kwargs :
            Passed to MetricsBatchEvaluator (model, zarr_path, comb_list, var_names, target_var_names, ...).
        """
        super().__init__(**kwargs)
        self.cell_cycle_genes_path = cell_cycle_genes_path
        self.min_cells = min_cells

        # determine var_names to use for the metrics: the actually used feature axis
        self.metric_var_names: List[str] | None = (
            self.target_var_names if self.target_var_names is not None else self.var_names
        )

        # Load cell cycle genes
        (
            self.cell_cycle_genes,
            self.s_genes,
            self.g2m_genes,
        ) = self._load_cell_cycle_genes()


    def _load_cell_cycle_genes(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Load cell cycle genes from file and split into S and G2M phases.
        Assumes first 43 entries are S-phase, rest G2M.
        """
        with open(self.cell_cycle_genes_path, "r") as f:
            cell_cycle_genes = [x.strip() for x in f.readlines()]

        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]

        return cell_cycle_genes, s_genes, g2m_genes


    def _process_batch_reconstruction(self, batch: List[str]) -> List[Dict]:
        """
        Self reconstruction: x -> model.predict(x)
        """
        x_combined, comb_indices = self._load_batch_data(batch)
        recon_combined = self._run_predict(x_combined)

        # apply any gene filtering defined in the base class
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
            compute_cell_cycle_metrics,
            self.metric_var_names,
            self.split_key,
            self.min_cells,
            self.cell_cycle_genes,
            self.s_genes,
            self.g2m_genes,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results

    def _process_batch_decode(self, batch: List[str]) -> List[Dict]:
        """
        Decode mode: z from emb_zarr -> model.decode(z), compare to true x.
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
            compute_cell_cycle_metrics,
            self.metric_var_names,
            self.split_key,
            self.min_cells,
            self.cell_cycle_genes,
            self.s_genes,
            self.g2m_genes,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results

    def _process_batch_conditional(self, batch: List[str]) -> List[Dict]:
        """
        Conditional / cross-batch reconstruction:
            - x_combined: data from batch combinations
            - all_ref: reference batch (per comb) via `_load_ref_data` from base
            - recon_combined: decode(x_combined, reference_batch_onehot)
        Metrics are computed between reference batch expression and recon.
        """
        x_combined, comb_indices = self._load_batch_data(batch)
        all_ref, ref_indices, if_ref_labels = self._load_ref_data(batch)

        # tile one-hot ref batch
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
            compute_cell_cycle_metrics_ref,
            self.metric_var_names,
            self.split_key,
            self.min_cells,
            self.cell_cycle_genes,
            self.s_genes,
            self.g2m_genes,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results


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
                "proportion_same_phase",
                "proportion_S",
                "proportion_G2M",
                "proportion_G1",
                "global_proportion_S_true",
                "global_proportion_S_recon",
                "global_proportion_G2M_true",
                "global_proportion_G2M_recon",
                "global_proportion_G1_true",
                "global_proportion_G1_recon",
                "proportion_mean_diff",
            ]
        else:
            base_columns = [
                "combination",
                *self.split_key,
                "n_cells",
                "proportion_same_phase",
                "proportion_S",
                "proportion_G2M",
                "proportion_G1",
                "global_proportion_S_true",
                "global_proportion_S_recon",
                "global_proportion_G2M_true",
                "global_proportion_G2M_recon",
                "global_proportion_G1_true",
                "global_proportion_G1_recon",
                "proportion_mean_diff",
            ]

        pd.DataFrame(columns=base_columns).to_csv(output_path, index=False)

        batch_gen = self._dynamic_batch_generator()
        with tqdm(total=len(self.all_combs), desc="Total progress") as pbar:
            for batch in batch_gen:
                batch_results = self._process_batch(batch)
                if not batch_results:
                    pbar.update(len(batch))
                    continue
                df = pd.DataFrame(batch_results).reindex(columns=base_columns)
                df.to_csv(output_path, mode="a", header=False, index=False)
                pbar.update(len(batch))
                del batch_results

        self.pool.close()
        self.pool.join()
        print(f"Processing completed. Results saved to {output_path}")


def compute_cell_cycle_metrics(
        var_names: List[str],
        split_key: List[str],
        min_cells: int,
        cell_cycle_genes: List[str],
        s_genes: List[str],
        g2m_genes: List[str],
        comb_data: tuple,
    ) -> Dict:

    comb, x, x_hat = comb_data
    records = []
    
    adata_true = AnnData(x)
    adata_recon = AnnData(x_hat)
    adata_true.var_names = var_names
    adata_recon.var_names = var_names
    
    metrics = {
        "combination": comb,
        split_key[0]: comb.split('-')[0],
        split_key[1]: comb.split('-')[1],
        split_key[2]: comb.split('-')[2],
        "n_cells": x.shape[0],
    }
    
    try:
        metrics.update(CellCycleCalculator.cell_cycle_labeling_similarity(
            adata_true=adata_true,
            adata_recon=adata_recon,
            min_cells=min_cells,
            cell_cycle_genes=cell_cycle_genes,
            s_genes=s_genes,
            g2m_genes=g2m_genes,
        ))
    except Exception as e:
        warnings.warn(f"Failed to compute cell cycle metrics for combination {comb}: {e}")
        metrics.update({
            "proportion_same_phase": np.nan,
            "proportion_S": np.nan,
            "proportion_G2M": np.nan,
            "proportion_G1": np.nan,
            "global_proportion_S_true": np.nan,
            "global_proportion_S_recon": np.nan,
            "global_proportion_G2M_true": np.nan,
            "global_proportion_G2M_recon": np.nan,
            "global_proportion_G1_true": np.nan,
            "global_proportion_G1_recon": np.nan,
            "proportion_mean_diff": np.nan
        })
    
    records.append(metrics)
    return records

class CellCycleCalculator:
    @staticmethod
    def cell_cycle_labeling_similarity(
        adata_true: AnnData,
        adata_recon: AnnData,
        min_cells: int = 30,
        cell_cycle_genes: List[str] = None,
        s_genes: List[str] = None,
        g2m_genes: List[str] = None,
        **kwargs
    ) -> Dict:
        """
        Calculate cell cycle labeling similarity between true and reconstructed data.
        
        Parameters
        ----------
        adata_true : AnnData
            Ground truth AnnData object
        adata_recon : AnnData
            Reconstructed AnnData object
        min_cells : int, optional (Default: 30)
            Minimum number of cells required for analysis
        cell_cycle_genes : List[str]
            List of all cell cycle genes
        s_genes : List[str]
            List of S-phase genes
        g2m_genes : List[str]
            List of G2M-phase genes
        
        Returns
        -------
        metrics : Dict
            Dictionary containing various cell cycle labeling metrics
        """
        genes_true = adata_true.var_names[sc.pp.filter_genes(adata_true, min_cells=min_cells, inplace=False)[0]]
        genes_recon = adata_recon.var_names[sc.pp.filter_genes(adata_recon, min_cells=min_cells, inplace=False)[0]]
        common_genes = genes_true
        
        cell_cycle_genes = [x for x in cell_cycle_genes if x in common_genes]
        s_genes = [x for x in s_genes if x in common_genes]
        g2m_genes = [x for x in g2m_genes if x in common_genes]
        
        if len(s_genes) < 5 or len(g2m_genes) < 5:
            warnings.warn("Insufficient cell cycle genes for analysis. Threshold: 5")
            return {
                "proportion_same_phase": np.nan,
                "proportion_S": np.nan,
                "proportion_G2M": np.nan,
                "proportion_G1": np.nan,
                "global_proportion_S_true": np.nan,
                "global_proportion_S_recon": np.nan,
                "global_proportion_G2M_true": np.nan,
                "global_proportion_G2M_recon": np.nan,
                "global_proportion_G1_true": np.nan,
                "global_proportion_G1_recon": np.nan,
                "proportion_mean_diff": np.nan
            }
        
        # Calculate cell cycle scores
        sc.tl.score_genes_cell_cycle(adata_true, s_genes=s_genes, g2m_genes=g2m_genes)
        sc.tl.score_genes_cell_cycle(adata_recon, s_genes=s_genes, g2m_genes=g2m_genes)
        s_mask_true = (adata_true.obs['phase'] == 'S')
        g2m_mask_true = (adata_true.obs['phase'] == 'G2M')
        g1_mask_true = (adata_true.obs['phase'] == 'G1')
            
        n_cells_true = adata_true.n_obs 
        n_cells_recon = adata_recon.n_obs 
        if n_cells_true != n_cells_recon:
            # For cross-batch reconstruction, only compare distribution-level metrics
            proportion_same_phase = np.nan
            proportion_S = np.nan
            proportion_G2M = np.nan
            proportion_G1 = np.nan
        else:
            # For end-to-end reconstruction with same number of cells, compute cell-level consistency
            proportion_same_phase = (adata_true.obs['phase'] == adata_recon.obs['phase']).sum() / n_cells_true
            proportion_S = (s_mask_true & (adata_recon.obs['phase'] == 'S')).sum() / s_mask_true.sum() if s_mask_true.sum() > 0 else np.nan
            proportion_G2M = (g2m_mask_true & (adata_recon.obs['phase'] == 'G2M')).sum() / g2m_mask_true.sum() if g2m_mask_true.sum() > 0 else np.nan
            proportion_G1 = (g1_mask_true & (adata_recon.obs['phase'] == 'G1')).sum() / g1_mask_true.sum() if g1_mask_true.sum() > 0 else np.nan
        
        # Global proportions
        global_proportion_S_true = s_mask_true.sum() / adata_true.n_obs
        global_proportion_S_recon = (adata_recon.obs['phase'] == 'S').sum() / adata_recon.n_obs
        global_proportion_G2M_true = g2m_mask_true.sum() / adata_true.n_obs
        global_proportion_G2M_recon = (adata_recon.obs['phase'] == 'G2M').sum() / adata_recon.n_obs
        global_proportion_G1_true = g1_mask_true.sum() / adata_true.n_obs
        global_proportion_G1_recon = (adata_recon.obs['phase'] == 'G1').sum() / adata_recon.n_obs
        
        # Calculate mean difference in proportions
        proportion_mean_diff = (
            abs(global_proportion_S_true - global_proportion_S_recon) + 
            abs(global_proportion_G2M_true - global_proportion_G2M_recon) + 
            abs(global_proportion_G1_true - global_proportion_G1_recon)
        ) / 3
        

        return {
            "proportion_same_phase": proportion_same_phase,
            "proportion_S": proportion_S,
            "proportion_G2M": proportion_G2M,
            "proportion_G1": proportion_G1,
            "global_proportion_S_true": global_proportion_S_true,
            "global_proportion_S_recon": global_proportion_S_recon,
            "global_proportion_G2M_true": global_proportion_G2M_true,
            "global_proportion_G2M_recon": global_proportion_G2M_recon,
            "global_proportion_G1_true": global_proportion_G1_true,
            "global_proportion_G1_recon": global_proportion_G1_recon,
            "proportion_mean_diff": proportion_mean_diff
        }