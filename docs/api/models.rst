models
======

.. currentmodule:: sc_reconstruction.models

End-to-end (``cstm_scvi_env``)
------------------------------

.. toctree::
    :maxdepth: 1

    _autosummary/sc_reconstruction.models.ReconPCA
    _autosummary/sc_reconstruction.models.ReconAE
    _autosummary/sc_reconstruction.models.ReconSCVI
    _autosummary/sc_reconstruction.models.ReconNLSCVI
    _autosummary/sc_reconstruction.models.ReconMLSCVI
    _autosummary/sc_reconstruction.models.ReconKNN
    _autosummary/sc_reconstruction.models.BaseReconstructionModel

Foundation-model embedders
--------------------------

Each foundation model has its own conda env; the envs are mutually
incompatible. All four wrapper classes share the same shape
(``.set_genes(...)`` then ``.get_latent_representation(adata)``).

.. list-table::
    :header-rows: 1
    :widths: 35 25 40

    * - Class
      - Conda env
      - Backend package
    * - :class:`ReconPretrainedStateModel`
      - ``cstm_scvi_env``
      - ``state`` (STATE foundation model)
    * - :class:`ReconPretrainedscGPT`
      - ``scgpt``
      - ``scgpt``
    * - :class:`ReconPretrainedscConcept`
      - ``scconcept_env``
      - ``concept``
    * - :class:`ReconPretrainedscimilarity`
      - ``scimilarity_env``
      - ``scimilarity``

.. toctree::
    :maxdepth: 1

    _autosummary/sc_reconstruction.models.ReconPretrainedStateModel
    _autosummary/sc_reconstruction.models.ReconPretrainedscGPT
    _autosummary/sc_reconstruction.models.ReconPretrainedscConcept
    _autosummary/sc_reconstruction.models.ReconPretrainedscimilarity
