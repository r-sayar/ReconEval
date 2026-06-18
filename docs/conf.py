"""Sphinx configuration for the ReconEval docs."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

# -- Project information -----------------------------------------------------

project = "ReconEval"
author = "Theis Lab"
copyright = "2026, Theis Lab"

try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version("reconeval")
    version = ".".join(release.split(".")[:2])
except Exception:
    release = "0.0.0"
    version = "0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- Autodoc / autosummary ---------------------------------------------------

# IMPORTANT: stubs in docs/api/_autosummary/ are pre-generated locally where
# real classes exist (not MagicMock); RTD reads them as-is. Do NOT flip this
# back to True on RTD — it'd regenerate stubs against mocks and you'd get the
# MagicMock-dunder explosion all over again.
autosummary_generate = False  # stubs are pre-generated and committed under docs/api/_autosummary/

autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
autodoc_default_options = {
    "show-inheritance": True,
    "inherited-members": False,
    "undoc-members": False,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

# Mocks for RTD's docs container — third-party heavy deps the package
# imports at module load time. Use the IMPORT name, not the PyPI name
# (e.g. POT installs as "pot" but imports as "ot"; scikit-learn installs
# as "scikit-learn" but imports as "sklearn"). Getting this wrong cascades
# to silent autodoc failures: the import raises, the name vanishes from
# the module namespace, and `.. autofunction::` / `.. autoclass::`
# render nothing (with suppress_warnings on, you don't even get a hint).
autodoc_mock_imports = [
    # Torch ecosystem
    "torch", "lightning", "torchmetrics", "flash_attn", "triton", "torchtext",
    # scvi-tools + scientific Python stack
    "scvi", "sklearn", "scipy", "ot", "tqdm",
    # Foundation-model wrappers
    "state", "scgpt", "concept", "scimilarity",
    # GPU/Dask stack used by ReconPCA + dataloaders
    "dask", "dask_cuda", "rapids_singlecell", "rmm",
    "cupy", "cuml", "cupyx", "cupy_backends",
    # AnnData / single-cell + storage
    "anndata", "scanpy", "decoupler", "omnipath", "zarr", "numcodecs",
    # Misc heavy deps
    "h5py", "hydra", "omegaconf",
]

suppress_warnings = [
    "autodoc",
    "autodoc.import_object",
    "autosummary",
    "autosummary.import_cycle",
    "ref.python",
    "ref.class",
]

# -- Myst / Notebook rendering ----------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
]
myst_heading_anchors = 3
nb_execution_mode = "off"

# -- Bibliography ------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_title = "ReconEval"
html_show_sphinx = False
html_show_sourcelink = False

html_theme_options = {
    "github_url": "https://github.com/theislab/ReconEval",
    "navbar_align": "left",
    "navigation_depth": 3,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "logo": {"text": "ReconEval"},
    "use_edit_page_button": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

nitpicky = False
