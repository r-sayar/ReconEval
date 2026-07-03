"""Lightning DataModules and supporting callbacks for ReconEval.

``H5adReconstructionDataModule`` is the h5ad-backed loader used by the
OpenProblems Viash components; it has minimal dependencies (anndata +
numpy) so it can be imported without dask/zarr/lightning/omegaconf.

The Zarr-based classes (``IterDaskDataModule``, ``DaskPCADataModule``,
``DatasetEpochCallback``) are imported lazily so a partial install
(e.g. an OpenProblems metric container that only needs the h5ad path)
still gets a working ``sc_reconstruction.dataloaders``.
"""

from __future__ import annotations

import sys as _sys

from .h5ad_datamodule import H5adReconstructionDataModule

__all__ = ["H5adReconstructionDataModule"]

try:
    from .datamodules import (
        DaskPCADataModule,
        DatasetEpochCallback,
        IterDaskDataModule,
    )
except Exception as _exc:  # noqa: BLE001 — heavy deps optional
    print(
        f"[sc_reconstruction.dataloaders] Zarr classes unavailable: "
        f"{type(_exc).__name__}: {_exc}",
        file=_sys.stderr,
    )
else:
    __all__ += ["DaskPCADataModule", "DatasetEpochCallback", "IterDaskDataModule"]
