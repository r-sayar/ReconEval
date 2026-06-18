"""ReconEval: benchmark for gene-expression reconstruction from single-cell
latent representations.

See :mod:`sc_reconstruction.metrics` for the scoring API,
:mod:`sc_reconstruction.models` for the end-to-end and foundation-model
wrappers, and :mod:`sc_reconstruction.decoders` for the MLP / transformer
decoders that map a latent embedding back to gene space.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("reconeval")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
