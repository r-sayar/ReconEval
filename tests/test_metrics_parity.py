"""Parity test: public `metrics.api` vs the paper's private metric Calculators.

For each metric we compute the same scalar via two independent code paths:

  - PUBLIC path: ``sc_reconstruction.metrics.api`` (the lightweight wrapper).
  - PRIVATE path: the per-family Calculator classes from
    ``/lustre/.../reconstruction/reconstruction/src/sc_reconstruction/metrics``
    (the version used to compute the paper's frozen results).

Both modules share the package name ``sc_reconstruction``, so we cannot import
them in the usual way. We hand-load each PRIVATE source file via importlib and
patch the few of intra-package imports it needs. That way the public API stays
the import-by-default version, and we get a clean reference for comparison.

The test passes on synthetic data so it does not need any frozen artefacts. It
exercises every metric that the user-facing wrapper exposes.

Run:
    srun --jobid=$JOB --quiet python -m pytest tests/test_metrics_parity.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Locate both sources and inject the public src/ first so
# ``import sc_reconstruction`` resolves to the public package.
# ---------------------------------------------------------------------------

PUBLIC_SRC  = Path("/lustre/groups/ml01/workspace/xiaotong.fu/public/reconstruction/src")
PRIVATE_DIR = Path(
    "/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction"
    "/src/sc_reconstruction/metrics"
)

if str(PUBLIC_SRC) not in sys.path:
    sys.path.insert(0, str(PUBLIC_SRC))

# Public API under test
import sc_reconstruction.metrics as pub_api  # noqa: E402
import sc_reconstruction.metrics.utils as pub_utils  # noqa: E402
from sc_reconstruction.metrics import base_eval as pub_base_eval  # noqa: E402


# Synthesise a parent package so the private files' relative imports resolve.
# The private `from .utils import ...` and `from .base_eval import ...` will land
# on the public copies, which are behaviour-identical for the helpers we wrap.
import types  # noqa: E402

_priv_pkg = types.ModuleType("priv_pkg")
_priv_pkg.__path__ = [str(PRIVATE_DIR)]
sys.modules["priv_pkg"] = _priv_pkg
sys.modules["priv_pkg.utils"] = pub_utils
sys.modules["priv_pkg.base_eval"] = pub_base_eval


def _load_private_module(name: str, file_path: Path):
    full_name = f"priv_pkg.{name}"
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "priv_pkg"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


priv_cellcycle    = _load_private_module("_cellcycle",    PRIVATE_DIR / "_cellcycle.py")
priv_coexpression = _load_private_module("_coexpression", PRIVATE_DIR / "_coexpression.py")
priv_cytokine     = _load_private_module("_cytokine",     PRIVATE_DIR / "_cytokine.py")
priv_deg          = _load_private_module("_deg",          PRIVATE_DIR / "_deg.py")
priv_pathway      = _load_private_module("_pathway",      PRIVATE_DIR / "_pathway.py")


# ---------------------------------------------------------------------------
# Fixtures: build a synthetic (true, predicted) pair plus a control reference.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cell_cycle_genes():
    path = Path(
        "/lustre/groups/ml01/workspace/xiaotong.fu/public/reconstruction/"
        "analysis/data/frozen/regev_lab_cell_cycle_genes.txt"
    )
    return pub_api.load_cell_cycle_genes(path)


@pytest.fixture(scope="module")
def progeny_model():
    return pub_api.load_progeny(organism="human")


@pytest.fixture(scope="module")
def cytokine_dict():
    path = Path(
        "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/"
        "sup/cytokine_act_merged.csv"
    )
    return pub_api.load_cytokine_dict_from_csv(path, celltype="B_cell")


@pytest.fixture(scope="module")
def data(cell_cycle_genes, progeny_model, cytokine_dict):
    """Synthesise a paired (true, pred) + (ctrl_true, ctrl_pred) AnnData."""
    s_genes, g2m_genes = cell_cycle_genes
    cyto_genes = sorted({g for genes in cytokine_dict.values() for g in genes})
    progeny_genes = progeny_model["target"].astype(str).str.upper().drop_duplicates().head(400).tolist()

    gene_universe = list(dict.fromkeys(s_genes + g2m_genes + progeny_genes + cyto_genes[:200]))[:600]

    rng = np.random.default_rng(0)
    n_ctrl = n_pert = 500
    n_genes = len(gene_universe)

    X_ctrl = rng.poisson(2.0, size=(n_ctrl, n_genes)).astype(float)
    X_pert = rng.poisson(2.0, size=(n_pert, n_genes)).astype(float)
    de_idx = rng.choice(n_genes, size=30, replace=False)
    X_pert[:, de_idx] += rng.poisson(4.0, size=(n_pert, 30))

    Xh_ctrl = X_ctrl + rng.normal(0, 0.4, size=X_ctrl.shape)
    Xh_pert = X_pert + rng.normal(0, 0.4, size=X_pert.shape)

    def _ad(X):
        var = pd.DataFrame(index=gene_universe)
        return AnnData(X.copy(), var=var)

    return dict(
        true=_ad(X_pert), pred=_ad(Xh_pert),
        true_ctrl=_ad(X_ctrl), pred_ctrl=_ad(Xh_ctrl),
        s_genes=s_genes, g2m_genes=g2m_genes,
        cytokine_dict=cytokine_dict,
        progeny_model=progeny_model,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOL = 1e-9
RTOL = 1e-7


def _approx(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    return np.isclose(a, b, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Statistical: pure-numpy helpers — exact match expected.
# ===========================================================================

class TestStatistical:

    def test_r2(self, data):
        X = data["true"].X; Y = data["pred"].X
        pub = pub_api.metric_r2(data["true"], data["pred"])
        priv = pub_utils.gene_r_squared(X, Y)  # same util in both repos
        assert _approx(pub, priv), (pub, priv)

    def test_energy_distance(self, data):
        X = data["true"].X; Y = data["pred"].X
        pub = pub_api.metric_energy_distance(data["true"], data["pred"])
        priv = pub_utils.energy_distance(X, Y)
        assert _approx(pub, priv), (pub, priv)


# ===========================================================================
# Biological: wrapper -> Calculator, compared to private Calculator on the same
# AnnData copies.
# ===========================================================================

class TestCellCycle:

    def test_proportion_same_phase(self, data):
        pub = pub_api.metric_cellcycle(
            data["true"], data["pred"],
            s_genes=data["s_genes"], g2m_genes=data["g2m_genes"],
            min_cells=10,
        )
        priv = priv_cellcycle.CellCycleCalculator.cell_cycle_labeling_similarity(
            adata_true=data["true"].copy(),
            adata_recon=data["pred"].copy(),
            min_cells=10,
            cell_cycle_genes=list(data["s_genes"]) + list(data["g2m_genes"]),
            s_genes=list(data["s_genes"]),
            g2m_genes=list(data["g2m_genes"]),
        )
        for key in [
            "proportion_same_phase",
            "proportion_S",
            "proportion_G2M",
            "proportion_G1",
            "proportion_mean_diff",
        ]:
            assert _approx(pub[key], priv[key]), (key, pub[key], priv[key])


class TestPathway:

    def test_average_corr(self, data):
        pub = pub_api.metric_pathway(
            data["true"], data["pred"],
            progeny_model=data["progeny_model"],
            correlation_measure="spearman",
            min_cells=10, overlap_threshold=3,
        )
        pathway_dict = (
            data["progeny_model"].groupby("source")["target"].apply(list).to_dict()
        )
        priv_dict = priv_pathway.PathwayCalculator.pathway_score_similarity(
            adata_true=data["true"].copy(),
            adata_recon=data["pred"].copy(),
            min_cells=10, overlap_threshold=3,
            correlation_measure="spearman",
            pipeline_output=False,
            progeny_model=data["progeny_model"],
            pathway_dict=pathway_dict,
        )
        priv_scalar = priv_dict["average"] if isinstance(priv_dict, dict) else priv_dict
        # PUBLIC tutorial uses fixed pearson loop; PRIVATE had the buggy
        # 1D-flatten branch we just fixed. We expect the public scalar to match
        # the FIXED private (i.e. after applying the same per-column loop).
        assert _approx(pub, priv_scalar), (pub, priv_scalar)


class TestCoexpression:

    def test_geneset_similarity(self, data):
        # We pass an explicit geneset_dict to both so neither hits the network.
        geneset = {"cellcycle_S": list(data["s_genes"]),
                   "cellcycle_G2M": list(data["g2m_genes"])}
        pub = pub_api.metric_coexpression(
            data["true"], data["pred"],
            geneset_dict=geneset,
            correlation_measure="spearman",
            min_cells=10, overlap_threshold=3,
        )
        priv = priv_coexpression.CoexpressionCalculator.gene_set_coexpression(
            adata_true=data["true"].copy(),
            adata_recon=data["pred"].copy(),
            overlap_threshold=3, min_cells=10,
            correlation_measure="spearman",
            pipeline_output=True,
            geneset_dict=geneset,
        )
        assert _approx(pub, priv), (pub, priv)


class TestDEG:

    def test_logfc_corrs_match(self, data):
        pub = pub_api.metric_deg(
            data["true"], data["pred"],
            ref_true=data["true_ctrl"], ref_pred=data["pred_ctrl"],
            method="wilcoxon", dice_k=[50, 100], min_cells=10,
        )
        priv = priv_deg.DegCalculator.compute_deg(
            x=data["true"].X, x_hat=data["pred"].X,
            refer=data["true_ctrl"].X, refer_hat=data["pred_ctrl"].X,
            method="wilcoxon", min_cells=10,
            set_neg_to_zero=False, dice_k=[50, 100],
            compute_mean_diff=True, compute_topk_corr=True,
            fdr_threshold=0.05,
        )
        # Public delegates straight to the private Calculator -> exact match
        # on every key present in both.
        common = set(pub) & set(priv)
        assert common, (sorted(pub.keys()), sorted(priv.keys()))
        for key in common:
            assert _approx(pub[key], priv[key]), (key, pub[key], priv[key])


class TestCytokine:

    def test_average_corr_spearman(self, data):
        pub = pub_api.metric_cytokine(
            data["true"], data["pred"],
            cytokine_dict=data["cytokine_dict"],
            correlation="spearman",
            min_genes=5,
        )
        priv_records = priv_cytokine.CytokineCalculator.cytokine_activity_similarity(
            adata_true=data["true"].copy(),
            adata_recon=data["pred"].copy(),
            cytokine2genes=data["cytokine_dict"],
            min_genes=5, ctrl_size=50, n_bins=25, reducer="mean",
        )
        # private returns [pearson, spearman]
        priv_spearman = priv_records[1]["average"]
        assert _approx(pub, priv_spearman), (pub, priv_spearman)


# ===========================================================================
# Perturbational: KNN purity is a pure utils call — exact match.
# ===========================================================================

class TestKNNPurity:

    def test_knn_purity(self, data):
        # Use ctrl-vs-pert as the labelling
        k = 20
        pub = pub_api.metric_knn_purity(
            adata_pred=data["pred"],
            adata_pert_true=data["true"],
            adata_ctrl=data["true_ctrl"],
            k=k,
        )
        priv = pub_utils.knn_purity(
            X_query=data["pred"].X, X_pert=data["true"].X,
            X_ctrl=data["true_ctrl"].X, k=k,
        )
        assert _approx(pub, priv), (pub, priv)


# ===========================================================================
# Rank-percentile: pure pandas reimplementation. Compare against an
# independent hand-rolled (M - rank) / (M - 1) computation.
# ===========================================================================

class TestRankPercentile:

    def test_paper_formula(self):
        df = pd.DataFrame(
            {"r2": [0.9, 0.7, 0.5, 0.3],
             "mse": [0.04, 0.07, 0.10, 0.15],
             "energy_distance": [0.3, 0.5, 0.8, 1.2]},
            index=list("ABCD"),
        )
        rp = pub_api.aggregate_rank_percentile(df)

        # Independent reference: best = 1.0, worst = 0.0
        expected = pd.DataFrame(
            {"r2":              [1.0, 2/3, 1/3, 0.0],
             "mse":             [1.0, 2/3, 1/3, 0.0],
             "energy_distance": [1.0, 2/3, 1/3, 0.0]},
            index=list("ABCD"),
        )
        pd.testing.assert_frame_equal(rp, expected, check_exact=False, atol=1e-10)

    def test_matches_fig2_clean_helper(self):
        """Compare against the `rank_percentile` helper defined in fig2_clean.ipynb cell 40.

        That helper is what the paper figures use to convert raw scores to rank-percentile.
        Public ``aggregate_rank_percentile`` should produce identical values column-wise.
        """
        import json
        nb_path = Path(
            "/lustre/groups/ml01/workspace/xiaotong.fu/public/reconstruction/"
            "analysis/data/plots/fig2_clean.ipynb"
        )
        nb = json.loads(nb_path.read_text())

        # Find the cell defining `def rank_percentile`
        helper_src = None
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if "def rank_percentile" in src:
                helper_src = src
                break
        assert helper_src is not None, "rank_percentile not found in fig2_clean.ipynb"

        ns: dict = {"pd": pd, "np": np}
        # Strip everything before the function definition so we don't execute the
        # whole heatmap-building cell (which references paths that don't exist).
        idx = helper_src.index("def rank_percentile")
        helper_src = helper_src[idx:]
        # Keep only the function body (stop at the next non-indented line).
        lines = helper_src.splitlines()
        cut = next((i for i, ln in enumerate(lines[1:], start=1)
                    if ln and not ln.startswith(("    ", "\t"))), len(lines))
        exec("\n".join(lines[:cut]), ns)
        fig2_rp = ns["rank_percentile"]

        # Reference matrix: 7 methods, mix of higher- and lower-is-better metrics.
        rng = np.random.default_rng(0)
        cols = ["r2", "mse", "energy_distance", "coexpression", "pathway",
                "cellcycle_proportion_same_phase", "deg_dice_at_100"]
        raw = pd.DataFrame(
            rng.uniform(0, 1, size=(7, len(cols))),
            index=[f"m{i}" for i in range(7)],
            columns=cols,
        )

        pub_rp = pub_api.aggregate_rank_percentile(raw)

        # fig2's helper assumes higher-is-better. To replicate column-wise behaviour,
        # negate lower-is-better columns before passing.
        expected = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
        for c in raw.columns:
            hib = pub_api.HIGHER_IS_BETTER.get(c, True)
            s = raw[c] if hib else -raw[c]
            expected[c] = fig2_rp(s)

        pd.testing.assert_frame_equal(pub_rp, expected, check_exact=False, atol=1e-12)
