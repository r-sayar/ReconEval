"""Lightning DataModules and supporting callbacks for ReconEval.

Only the names actually used by the benchmark configs are exported:
``IterDaskDataModule`` for streaming Dask-backed training,
``DaskPCADataModule`` for ReconPCA's chunked SVD path, and
``DatasetEpochCallback`` for epoch-aware dataset state.
"""

from .datamodules import (
    IterDaskDataModule,
    DaskPCADataModule,
    DatasetEpochCallback,
)

__all__ = [
    "IterDaskDataModule",
    "DaskPCADataModule",
    "DatasetEpochCallback",
]
