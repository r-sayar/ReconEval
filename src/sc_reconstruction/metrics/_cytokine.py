import os
import warnings
from functools import partial
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import stats
from tqdm import tqdm

from sc_reconstruction.metrics.base_eval import MetricsBatchEvaluator

class CytokineBatchEvaluator(MetricsBatchEvaluator):
    """
    Batch evaluator for cytokine activity metrics.

    Modes supported:
        - reconstruction: x -> model.predict(x)
        - decode:        z -> model.decode(z)
    (conditional mode is not implemented here)
    """

    def __init__(
        self,
        cytokine_csv_path: str,
        var_names: List[str] = None,
        min_genes: int = 5,
        ctrl_size: int = 50,
        n_bins: int = 25,
        reducer: str = "mean",
        **kwargs,
    ):
        """
        Parameters
        ----------
        cytokine_csv_path : str
            Path to CSV containing cytokine–gene associations with columns
            ['Celltype_Str', 'Cytokine_Str', 'Gene'].
        var_names : List[str]
            Gene names in the same order as the input features X.
        min_genes : int
            Minimum number of genes per cytokine to keep that cytokine.
        ctrl_size : int
            `ctrl_size` for `scanpy.tl.score_genes`.
        n_bins : int
            `n_bins` for `scanpy.tl.score_genes`.
        reducer : str
            How to aggregate cell-level scores ('mean' or 'median').
        **kwargs :
            Passed to `MetricsBatchEvaluator` (model, zarr_path, comb_list, mode, etc.).
        """
        super().__init__(var_names=var_names, **kwargs)

        self.cytokine_csv_path = cytokine_csv_path
        self.min_genes = min_genes
        self.ctrl_size = ctrl_size
        self.n_bins = n_bins
        self.reducer = reducer


        self.metric_var_names: List[str] | None = (
            self.target_var_names if self.target_var_names is not None else self.var_names
        )
        if self.metric_var_names is None:
            raise ValueError(
                "CytokineBatchEvaluator requires `var_names` (or `target_var_names`) "
                "to map genes for cytokine scoring."
            )
        self.cytokine_dict = self._load_cytokine_data()
        self.celltype_mapping = self._load_celltype_mapping()


    def _load_celltype_mapping(self) -> Dict[str, str]:
        return {
            'B_cell': 'B_cell',
            'CD1c_positive_myeloid_dendritic_cell': 'cDC2',
            'CD4_positive__alpha_beta_T_cell': 'T_cell_CD4',
            'CD8_positive__alpha_beta_T_cell': 'T_cell_CD8',
            'alveolar_macrophage': 'Macrophage',
            'bronchus_fibroblast_of_lung': 'NA',
            'capillary_endothelial_cell': 'NA',
            'classical_monocyte': 'Monocyte',
            'club_cell': 'NA',
            'conventional_dendritic_cell': 'cDC2',
            'dendritic_cell': 'MigDC',
            'endothelial_cell_of_lymphatic_vessel': 'NA',
            'epithelial_cell_of_lung': 'NA',
            'fibroblast_of_lung': 'NA',
            'macrophage': 'Macrophage',
            'malignant_cell': 'NA',
            'mast_cell': 'Mast_cell',
            'mesothelial_cell': 'NA',
            'multiciliated_epithelial_cell': 'NA',
            'myeloid_cell': 'Monocyte',
            'natural_killer_cell': 'NK_cell',
            'neutrophil': 'Neutrophil',
            'non_classical_monocyte': 'Monocyte',
            'pericyte': 'NA',
            'plasma_cell': 'B_cell',
            'plasmacytoid_dendritic_cell': 'pDC',
            'pulmonary_alveolar_type_1_cell': 'NA',
            'pulmonary_alveolar_type_2_cell': 'NA',
            'pulmonary_artery_endothelial_cell': 'NA',
            'regulatory_T_cell': 'Treg',
            'smooth_muscle_cell': 'NA',
            'stromal_cell': 'NA',
            'vein_endothelial_cell': 'NA',
            # ── short names (PBMC ST comb_w_obs.zarr) ────────────────────
            "B": "B_cell",
            "CD4": "T_cell_CD4",
            "CD8": "T_cell_CD8",
            "CD14_Mono": "Monocyte",
            "CD16_Mono": "Monocyte",
            "CD56_bright_NK": "NK_cell",
            "CD56_dim_NK": "NK_cell",
            "NKT": "NK_cell",
            "Treg": "Treg",
            "cDC": "cDC2",
            "pDC": "pDC",
            "Plasmablast": "B_cell",
            "MAIT": "T_cell_CD8",
            "HSPC": "NA",
            "ILC": "ILC",
            "Platelet": "NA",
            "other": "NA",
        }

    def _load_cytokine_data(self) -> Dict[Tuple[str, str], List[str]]:
        """
        Load cytokine data from CSV file and process into dict:
            (Celltype_Str, Cytokine_Str) -> [genes]
        """
        cytokine_act_csv = pd.read_csv(self.cytokine_csv_path)
        cytokine_act_csv["Gene"] = cytokine_act_csv["Gene"].str.upper()
        sig_genes_by_pair = (
            cytokine_act_csv.loc[:, ["Celltype_Str", "Cytokine_Str", "Gene"]]
            .groupby(["Celltype_Str", "Cytokine_Str"])["Gene"]
            .apply(list)
            .to_dict()
        )

        filtered_sig = {
            k: sorted(set(v))
            for k, v in sig_genes_by_pair.items()
            if len(set(v)) >= self.min_genes
        }
        return filtered_sig


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
            compute_cytokine_metrics,
            self.metric_var_names,
            self.split_key,
            self.min_genes,
            self.ctrl_size,
            self.n_bins,
            self.reducer,
            self.cytokine_dict,
            self.celltype_mapping,
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
            compute_cytokine_metrics,
            self.metric_var_names,
            self.split_key,
            self.min_genes,
            self.ctrl_size,
            self.n_bins,
            self.reducer,
            self.cytokine_dict,
            self.celltype_mapping,
        )
        batch_records_nested = list(self.pool.imap(worker, comb_data, chunksize=10))
        flat_results = list(chain.from_iterable(batch_records_nested))
        return flat_results

    def _process_batch_conditional(self, batch: List[str]) -> List[Dict]:
        """
        Optional: if you don't want cytokine metrics for conditional mode,
        make that explicit.
        """
        raise NotImplementedError(
            "CytokineBatchEvaluator does not support 'conditional' mode."
        )


    def run(self, output_path: str):
        # Fresh start: downstream writers append in chunks.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Existing output file {output_path} removed.")

        cytokine_names: set[str] = set()
        for (celltype, cytokine) in self.cytokine_dict.keys():
            cytokine_names.add(cytokine)

        base_columns = [
            "combination",
            *self.split_key,
            "n_cells",
            "method",
            "average",
            "n_common_cytokines",
        ]

        for cytokine in sorted(cytokine_names):
            base_columns.extend([f"true-{cytokine}", f"recon-{cytokine}"])

        pd.DataFrame(columns=base_columns).to_csv(output_path, index=False)

        # Process batches and append results
        batch_gen = self._dynamic_batch_generator()
        with tqdm(total=len(self.all_combs), desc="Total progress") as pbar:
            for batch in batch_gen:
                batch_results = self._process_batch(batch)  # already List[Dict]
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


def compute_cytokine_metrics(
        var_names: List[str],
        split_key: List[str],
        min_genes: int,
        ctrl_size: int,
        n_bins: int,
        reducer: str,
        cytokine_dict: Dict[Tuple[str, str], List[str]],
        celltype_mapping: Dict[str, str],
        comb_data: tuple,
    ) -> Dict:
    """
    Wrapper function for computing cytokine metrics in parallel.
    """
    comb, x, x_hat = comb_data
    records = []
    
    adata_true = AnnData(x)
    adata_recon = AnnData(x_hat)
    adata_true.var_names = var_names
    adata_recon.var_names = var_names
    
    # Extract cell type from combination string
    cell_type = comb.split('-')[0]
    mapped_cell_type = celltype_mapping.get(cell_type, "NA")
    
    # Filter cytokine dictionary for this cell type
    celltype_cytokines = {
        cytokine: genes 
        for (ct, cytokine), genes in cytokine_dict.items() 
        if ct == mapped_cell_type
    }
    
    # If no cytokines for this cell type, return empty results
    if not celltype_cytokines:
        for method in ["pearson", "spearman"]:
            metrics = {
                "combination": comb,
                split_key[0]: comb.split('-')[0],
                split_key[1]: comb.split('-')[1],
                split_key[2]: comb.split('-')[2],
                "n_cells": x.shape[0],
                "method": method,
                "average": np.nan,
                "n_common_cytokines": 0,
            }
            
            # Add empty columns for each cytokine
            for cytokine in set(cytokine for (_, cytokine) in cytokine_dict.keys()):
                metrics[f"true-{cytokine}"] = np.nan
                metrics[f"recon-{cytokine}"] = np.nan
                
            records.append(metrics)
        return records
    
    try:
        cytokine_results = CytokineCalculator.cytokine_activity_similarity(
            adata_true=adata_true,
            adata_recon=adata_recon,
            cytokine2genes=celltype_cytokines,
            min_genes=min_genes,
            ctrl_size=ctrl_size,
            n_bins=n_bins,
            reducer=reducer,
        )
        
        # Convert to list of records (one for pearson, one for spearman)
        for result in cytokine_results:
            result.update({
                "combination": comb,
                split_key[0]: comb.split('-')[0],
                split_key[1]: comb.split('-')[1],
                split_key[2]: comb.split('-')[2],
                "n_cells": x.shape[0],
            })
            records.append(result)
            
    except Exception as e:
        warnings.warn(f"Failed to compute cytokine metrics for combination {comb}: {e}")
        for method in ["pearson", "spearman"]:
            metrics = {
                "combination": comb,
                split_key[0]: comb.split('-')[0],
                split_key[1]: comb.split('-')[1],
                split_key[2]: comb.split('-')[2],
                "n_cells": x.shape[0],
                "method": method,
                "average": np.nan,
                "n_common_cytokines": 0,
            }
            
            # Add empty columns for each cytokine
            for cytokine in set(cytokine for (_, cytokine) in cytokine_dict.keys()):
                metrics[f"true-{cytokine}"] = np.nan
                metrics[f"recon-{cytokine}"] = np.nan
                
            records.append(metrics)
    
    return records




class CytokineCalculator:
    @staticmethod
    def cytokine_activity_similarity(
        adata_true: AnnData,
        adata_recon: AnnData,
        cytokine2genes: Dict[str, List[str]],
        min_genes: int = 5,
        ctrl_size: int = 50,
        n_bins: int = 25,
        reducer: str = "mean",
        **kwargs
    ) -> List[Dict]:
        """
        Calculate cytokine activity similarity between true and reconstructed data.
        """
        
        def _score_one(ad, cytokine2genes, min_genes, ctrl_size, n_bins, reducer):
            """Score cytokine activities for a single AnnData object"""
            vals = {}
            for cyto, genes in cytokine2genes.items():
                use = [g for g in genes if g in ad.var_names]
                if len(use) < min_genes:
                    vals[cyto] = np.nan
                    continue
                tmp = "__tmp_score__"
                sc.tl.score_genes(
                    ad, gene_list=use, score_name=tmp,
                    ctrl_size=ctrl_size, n_bins=n_bins, random_state=0
                )
                s = ad.obs[tmp]
                vals[cyto] = float(s.median() if reducer == "median" else s.mean())
                ad.obs.drop(columns=[tmp], inplace=True)
            return vals

        true_vals = _score_one(adata_true, cytokine2genes, min_genes, ctrl_size, n_bins, reducer)
        recon_vals = _score_one(adata_recon, cytokine2genes, min_genes, ctrl_size, n_bins, reducer)
        cyto_cols = list(cytokine2genes.keys())
        vtrue = np.array([true_vals.get(c, np.nan) for c in cyto_cols], dtype=float)
        vrecon = np.array([recon_vals.get(c, np.nan) for c in cyto_cols], dtype=float)
        
        # mask = ~np.isnan(vtrue) & ~np.isnan(vrecon)
        mask = ~np.isnan(vtrue)
        n_common = int(mask.sum())
        
        if n_common >= 2:
            try:
                pr = stats.pearsonr(vtrue[mask], vrecon[mask]).statistic
                sr = stats.spearmanr(vtrue[mask], vrecon[mask]).statistic
            except:
                vrecon_noised = vrecon[mask] + np.random.normal(0, 1e-6, vrecon[mask].shape)
                pr = stats.pearsonr(vtrue[mask], vrecon_noised).statistic
                sr = stats.spearmanr(vtrue[mask], vrecon_noised).statistic
        else:
            pr = sr = np.nan
        
        def _expand(vals, tag):
            return {f"{tag}-{cyto}": vals.get(cyto, np.nan) for cyto in cyto_cols}
        
        rec_pearson = {
            "method": "pearson",
            "average": pr,  
            "n_common_cytokines": n_common,
            **_expand(true_vals, "true"),
            **_expand(recon_vals, "recon"),
        }
        
        rec_spearman = {
            "method": "spearman",
            "average": sr,  
            "n_common_cytokines": n_common,
            **_expand(true_vals, "true"),
            **_expand(recon_vals, "recon"),
        }
        
        return [rec_pearson, rec_spearman]