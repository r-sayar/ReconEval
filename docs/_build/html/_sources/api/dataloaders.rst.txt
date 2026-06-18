dataloaders
===========

.. currentmodule:: sc_reconstruction.dataloaders

LightningDataModules used by the training drivers under
``experiments/01_end_to_end/``. Most users instantiate them via the
Hydra configs rather than calling them directly.

.. toctree::
    :maxdepth: 1

    _autosummary/sc_reconstruction.dataloaders.IterDaskDataModule
    _autosummary/sc_reconstruction.dataloaders.DaskPCADataModule
    _autosummary/sc_reconstruction.dataloaders.DatasetEpochCallback
