from typing import Dict, Any, List, Tuple
import numpy as np
import os
import warnings
from functools import partial

from tqdm import tqdm

from sc_reconstruction.metrics.base_eval import MetricsBatchEvaluator


class ProjectionMetricsBatchEvaluator(MetricsBatchEvaluator):

    def __init__(
        self,
        projector: Any,
        projection_metrics: List[str] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        projector : Any
            An object with a `.predict(ndarray)` method that performs dimension reduction.
        projection_metrics : List[str] | None
            List of metric names in `metric_funcs` that should use projected space.
            If None, defaults to ['mmd_rbf', 'energy_distance'].
        **kwargs :
            Passed to `MetricsBatchEvaluator`.
        """
        super().__init__(**kwargs)
        self.projector = projector
        if projection_metrics is None:
            projection_metrics = ["mmd_rbf", "energy_distance"]
        self.projection_metrics = set(projection_metrics)


    def _process_batch_reconstruction(self, batch: List[str]) -> List[Dict]:

        x_combined, comb_indices = self._load_batch_data(batch)
        recon_combined = self._run_predict(x_combined)

        x_for_metrics, recon_for_metrics = self._preprocess_for_metrics(
            x_combined, recon_combined
        )

        x_proj_combined = self.projector.predict(x_combined)
        recon_proj_combined = self.projector.predict(recon_combined)

        results = []
        for comb, start, end in comb_indices:
            x_slice = x_for_metrics[start:end]
            recon_slice = recon_for_metrics[start:end]
            x_proj_slice = x_proj_combined[start:end]
            recon_proj_slice = recon_proj_combined[start:end]
            results.append(
                (
                    comb,
                    None,  # comb_ref
                    x_slice,
                    recon_slice,
                    x_proj_slice,
                    recon_proj_slice,
                )
            )

        worker = partial(
            compute_metrics_with_projection,
            self.metric_funcs,
            self.split_key,
            self.projection_metrics,
        )
        batch_results = list(self.pool.imap(worker, results, chunksize=20))
        del x_combined, recon_combined, x_proj_combined, recon_proj_combined, results
        return batch_results

    def _process_batch_decode(self, batch: List[str]) -> List[Dict]:
        z_combined, x_combined, comb_indices = self._load_decode_data(batch)
        recon_combined = self._run_decode(z_combined)

        x_for_metrics, recon_for_metrics = self._preprocess_for_metrics(
            x_combined, recon_combined
        )

        x_proj_combined = self.projector.predict(x_combined)
        recon_proj_combined = self.projector.predict(recon_combined)

        results = []
        for comb, start, end in comb_indices:
            x_slice = x_for_metrics[start:end]
            recon_slice = recon_for_metrics[start:end]
            x_proj_slice = x_proj_combined[start:end]
            recon_proj_slice = recon_proj_combined[start:end]
            results.append(
                (
                    comb,
                    None,
                    x_slice,
                    recon_slice,
                    x_proj_slice,
                    recon_proj_slice,
                )
            )

        worker = partial(
            compute_metrics_with_projection,
            self.metric_funcs,
            self.split_key,
            self.projection_metrics,
        )
        batch_results = list(self.pool.imap(worker, results, chunksize=20))
        del z_combined, x_combined, recon_combined, x_proj_combined, recon_proj_combined, results
        return batch_results

    def _process_batch_conditional(self, batch: List[str]) -> List[Dict]:
        x_combined, comb_indices = self._load_batch_data(batch)
        all_ref, ref_indices, if_ref_labels = self._load_ref_data(batch)

        ref_batch_onehot = np.tile(
            self.reference_batch_onehot, (x_combined.shape[0], 1)
        )
        recon_combined = self._run_conditional_decode(x_combined, ref_batch_onehot)

        ref_proj_combined = self.projector.predict(all_ref)
        recon_proj_combined = self.projector.predict(recon_combined)

        proj_results = []
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

            x_ref_proj = ref_proj_combined[ref_start:ref_end]
            recon_proj = recon_proj_combined[start:end]

            if self.proj_out_store is not None:
                self._write_pred_to_zarr(self.reference_batch, comb, recon_for_metrics)

            proj_results.append(
                (
                    comb,
                    ref_comb,
                    x_ref_for_metrics,
                    recon_for_metrics,
                    x_ref_proj,
                    recon_proj,
                )
            )

        worker = partial(
            compute_metrics_with_projection_ref,
            self.metric_funcs,
            self.split_key,
            self.projection_metrics,
        )
        batch_results = list(self.pool.imap(worker, proj_results, chunksize=20))
        del x_combined, all_ref, recon_combined, ref_proj_combined, recon_proj_combined, proj_results
        return batch_results




def compute_metrics_with_projection(
    metric_funcs: Dict[str, Any],
    split_key: List[str],
    projection_metrics: set[str],
    data: Tuple[str, str | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Dict:
    """
    Compute both standard and projection metrics for a combination.

    data: (comb, comb_ref, x, recon, x_proj, recon_proj)
    """
    comb, comb_ref, x, recon, x_proj, recon_proj = data
    comb_keys = comb.split("-")

    metrics: Dict[str, Any] = {
        "combination": comb,
        "combination_ref": comb_ref,
        split_key[0]: comb_keys[0],
        split_key[1]: comb_keys[1] if len(comb_keys) > 1 else "None",
        split_key[2]: comb_keys[2] if len(comb_keys) > 2 else "None",
        "n_cells": x.shape[0],
        "n_cells_recon": recon.shape[0],

    }

    # full-space metrics
    for name, func in metric_funcs.items():
        if name in projection_metrics:
            continue
        metrics[name] = func(x, recon)

    # projected-space metrics
    for name in projection_metrics:
        if name not in metric_funcs:
            continue
        metrics[name] = metric_funcs[name](x_proj, recon_proj)
    
    return metrics


def compute_metrics_with_projection_ref(
    metric_funcs: Dict[str, Any],
    split_key: List[str],
    projection_metrics: set[str],
    data: Tuple[str, str | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Dict:
    """
    Conditional setting: metrics between reference batch and reconstructed batch.

    data: (comb, comb_ref, x_ref, recon, x_ref_proj, recon_proj)
    """
    comb, comb_ref, x_ref, recon, x_ref_proj, recon_proj = data
    comb_keys = comb.split("-")

    metrics: Dict[str, Any] = {
        "combination": comb,
        "combination_ref": comb_ref,  # None if self reference
        split_key[0]: comb_keys[0],
        split_key[1]: comb_keys[1] if len(comb_keys) > 1 else "None",
        split_key[2]: comb_keys[2] if len(comb_keys) > 2 else "None",
        "n_cells": x_ref.shape[0],
        "n_cells_recon": recon.shape[0],
    }

    for name, func in metric_funcs.items():
        if name in projection_metrics:
            continue
        metrics[name] = func(x_ref, recon)

    for name in projection_metrics:
        if name not in metric_funcs:
            continue
        metrics[name] = metric_funcs[name](x_ref_proj, recon_proj)

    return metrics


