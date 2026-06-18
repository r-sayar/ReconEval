decoders
========

.. currentmodule:: sc_reconstruction.decoders

Decoders that map a latent embedding back to gene expression.
``ReconMLPDecoder`` works in ``cstm_scvi_env``. ``ReconTransformerDecoder``
depends on the ``concept`` package and runs under ``scconcept_env``.

.. toctree::
    :maxdepth: 1

    _autosummary/sc_reconstruction.decoders.ReconMLPDecoder
    _autosummary/sc_reconstruction.decoders.ReconTransformerDecoder
    _autosummary/sc_reconstruction.decoders.BaseReconstructionDecoder
