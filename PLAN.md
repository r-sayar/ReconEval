# ReconEval — Restructure & Packaging Plan

## Context

The public repo at `/lustre/groups/ml01/workspace/xiaotong.fu/public/reconstruction` currently ships cleaned figures and the `src/sc_reconstruction/` source tree. The private repo at `/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction` holds the raw experiment configs and submission scripts for three task families: end-to-end reconstruction, foundation-model (FM) reconstruction, and latent-shift (CellFlow/STATE) perturbation prediction.

Goals for this restructure:

1. Make the **metrics** tutorial usable in a single light env so external users can compute the paper's metrics on their own (true, predicted) pair without installing the FM-specific deps.
2. Lift the experiment + eval scripts from the private repo into a clean `experiments/` and `evaluation/` tree organised by (task × model).
3. Decide the packaging boundary: what is `pip install`-able vs what stays as scripts.
4. Decouple FM embedding (heavy, env-per-FM) from decoder + metrics (light, FM-agnostic) so anyone can plug in their own FM.

## Decisions (locked in)

| # | Question | Decision |
|---|---|---|
| 1 | Example tutorial data | Tiny pair shipped in `data/example/` (offline, self-contained, ~5 MB per task) |
| 2 | FM adapter API | Two-step: `adapter.embed(adata) -> AnnData` writes `.obsm["X_fm"]`; decoder + metrics layer is FM-agnostic |
| 3 | Metrics public surface | `sc_reconstruction.metrics.api` light wrapper only; heavy `MetricsBatchEvaluator` family lives at `sc_reconstruction.metrics._batch` (advanced/internal, not in `__init__`) |
| 4 | Release channel | Git tag `v0.1.0-paper`; install via `pip install "screconstruction-tools[metrics] @ git+https://github.com/theislab/ReconEval@v0.1.0-paper"`; defer PyPI until after paper acceptance |

## Target layout

```
public/reconstruction/                    (theislab/ReconEval)
├── src/sc_reconstruction/
│   ├── metrics/
│   │   ├── __init__.py                   ← re-exports api.py only
│   │   ├── api.py                        ★ NEW — light public API
│   │   ├── _batch.py                     (renamed from base_eval.py — internal)
│   │   ├── distributional.py, _cellcycle.py, _coexpression.py,
│   │   ├── _cytokine.py, _deg.py, _pathway.py, utils.py
│   │   └── loss.py
│   ├── adapters/
│   │   ├── fm_protocol.py                ★ NEW — FoundationModelAdapter Protocol
│   │   ├── fm_scgpt.py, fm_scconcept.py,
│   │   ├── fm_scimilarity.py, fm_se.py   ★ NEW — thin facades on models/recon*
│   │   ├── e2e_encoder_adapters.py, e2e_decoder_adapters.py, state_decoder_adapter.py
│   ├── models/                           (unchanged — recon{ae,pca,scgpt,scconcept,…}.py)
│   ├── decoders/, dataloaders/, train/, utils/
├── tutorials/
│   ├── README.md                         (index — already exists)
│   ├── 01a_end_to_end_setup.ipynb        (placeholder, defer)
│   ├── 01b_end_to_end_metrics.ipynb      ★ PRIORITY
│   ├── 02a_fm_setup.ipynb                (placeholder, defer)
│   ├── 02b_fm_metrics.ipynb              ★ PRIORITY
│   ├── 03a_latent_shift_setup.ipynb      (placeholder, defer)
│   └── 03b_latent_shift_metrics.ipynb    ★ PRIORITY
├── data/example/                         ★ NEW
│   ├── extract.py                        (subsets analysis/frozen/ → tiny pairs)
│   ├── e2e/{true,pred}.h5ad
│   ├── fm/{true,pred}.h5ad
│   └── latent_shift/{true,pred}.h5ad
├── experiments/                          ★ NEW
│   ├── README.md
│   ├── 01_end_to_end/configs/, submit/
│   ├── 02_foundation_model/configs/, submit/
│   └── 03_latent_shift/configs/, submit/
├── evaluation/                           ★ NEW
│   ├── README.md
│   ├── 01_end_to_end/configs/, submit/
│   ├── 02_foundation_model/configs/, submit/
│   └── 03_latent_shift/configs/, submit/
├── envs/                                 ★ NEW
│   ├── metrics.yaml                      (light: numpy/pandas/scanpy/anndata/decoupler/zarr)
│   ├── e2e.yaml                          (torch + scvi-tools)
│   ├── fm-scgpt.yaml, fm-scconcept.yaml,
│   ├── fm-scimilarity.yaml, fm-se.yaml
│   └── latent-shift.yaml                 (cellflow / state)
├── analysis/                             (unchanged)
├── pyproject.toml                        ← extend with optional-dependencies extras
├── README.rst                            ← point at new tutorials/experiments/evaluation
├── Makefile                              ← add `make example-data` target
└── (existing: AUTHORS.rst, CONTRIBUTING.rst, LICENSE, …)
```

## 1. Light metrics API (`src/sc_reconstruction/metrics/api.py`)

```python
# Public functions (only these are exported from sc_reconstruction.metrics)
def compute_statistical_metrics(adata_true, adata_pred, *, projector=None) -> dict[str, float]
def compute_biological_metrics(adata_true, adata_pred, *, groupby="condition") -> dict[str, float]
def compute_perturbational_metrics(adata_pred, adata_ctrl, *, groupby="condition") -> dict[str, float]
def compute_all_metrics(adata_true, adata_pred, adata_ctrl=None, **kwargs) -> dict[str, float]
def aggregate_rank_percentile(scores_df: pd.DataFrame,
                              higher_is_better: dict[str, bool]) -> pd.Series
```

Implementation:
- Each function thin-wraps the existing per-family helpers in `metrics/utils.py` (pure numpy: `compute_pearson`, `compute_spearman`, `mmd_rbf`, `energy_distance`, `gene_r_squared`, `knn_purity`) and the `*Calculator` classes in `_cellcycle.py`, `_coexpression.py`, `_cytokine.py`, `_deg.py`, `_pathway.py`.
- `aggregate_rank_percentile` reuses the paper formula `(M - rank) / (M - 1)` — copy from `analysis/data/plots/fig{2,3}_clean.ipynb`.

`metrics/__init__.py` re-exports only the API functions:
```python
from .api import (compute_statistical_metrics, compute_biological_metrics,
                  compute_perturbational_metrics, compute_all_metrics,
                  aggregate_rank_percentile)
```

The heavy `MetricsBatchEvaluator` family is reachable as `sc_reconstruction.metrics._batch` for the reproduction pipeline but not in `__init__`.

## 2. Tutorials (priority: metrics)

All three metrics notebooks share the same 5-cell skeleton:

| Cell | Content |
|---|---|
| 1 | `import sc_reconstruction.metrics as m`; load `data/example/<task>/{true,pred}.h5ad` |
| 2 | Per-metric walkthrough: call each pure helper from `metrics.utils` and each `*Calculator` individually; print the returned scalar |
| 3 | Biological metrics walkthrough: `compute_pathway_metrics`, `compute_deg_metrics`, etc. on `comb_data = (cond, X_true, X_recon)` |
| 4 | One-call API: `scores = m.compute_all_metrics(adata_true, adata_pred)` → display per-metric `{metric_name: float}` dict |
| 5 | Aggregation: turn `{method: {metric: score}}` into `{method: rank_percentile}` via `m.aggregate_rank_percentile(...)`; show the paper formula `(M - rank) / (M - 1)` produces best=1.0, worst=0.0 |

Per-notebook differences:

- **`01b_end_to_end_metrics.ipynb`** — example: AE-128 decoded vs ground truth on PBMC.
- **`02b_fm_metrics.ipynb`** — example: scConcept embedding loaded from `.obsm["X_fm"]`, decoded via `decoders.MLPDecoder`, evaluated.
- **`03b_latent_shift_metrics.ipynb`** — example: CellFlow predicting held-out perturbation, includes `compute_perturbational_metrics` with `knn_purity`.

Setup notebooks (01a/02a/03a) stay as placeholders for this round.

## 3. FM adapter Protocol + thin facades

`src/sc_reconstruction/adapters/fm_protocol.py`:

```python
from typing import Protocol
from anndata import AnnData

class FoundationModelAdapter(Protocol):
    def load(self, weights_path: str, **kwargs) -> None: ...
    def embed(self, adata: AnnData) -> AnnData:
        """Adds .obsm['X_fm']; returns adata (in-place or new)."""
        ...
```

For each FM, write a thin facade in `adapters/fm_<name>.py` that delegates to the existing model class in `models/recon<name>.py`. Refactor only the I/O — the FM math stays untouched.

Result: a user with a brand-new FM only needs to write `MyFMAdapter(FoundationModelAdapter)` with two methods; the rest of the benchmark (decoder + metrics) is reused as-is. The two-step flow:

```python
# Step 1 — heavy env, FM-specific
from sc_reconstruction.adapters.fm_scgpt import ScGPTAdapter
fm = ScGPTAdapter(); fm.load("scgpt_weights/")
adata = fm.embed(adata)                       # writes .obsm["X_fm"]
adata.write_h5ad("embedding.h5ad")

# Step 2 — light env, FM-agnostic
from sc_reconstruction.decoders import MLPDecoder
from sc_reconstruction.metrics import compute_all_metrics
decoder = MLPDecoder.from_pretrained(ckpt)
adata_pred = decoder.decode(adata)
scores = compute_all_metrics(adata_true, adata_pred)
```

## 4. `experiments/` folder

Pattern per task (lifted from private `runs/configs/` + `runs/scripts/train/`):

```
experiments/<task>/
├── configs/<model>[_<latent>][_<decoder>].yaml    # self-contained (Hydra-flattened)
└── submit/<model>[_<latent>][_<decoder>].sh       # sbatch script
```

Representative sources to lift:

| Public task path | Private source |
|---|---|
| `experiments/01_end_to_end/` | configs: `runs/configs/base_pretrained_emb.yaml` + `runs/configs/model/train/*`<br>submit: `runs/scripts/train/decoderonly_grid_hvg.sbatch` |
| `experiments/02_foundation_model/` | configs: `runs/configs/base_{scgpt,scconcept,scimilarity,se}_ft_hvg.yaml` + `runs/configs/decoder/{MLP,Transformer,KNN}.yaml`<br>submit: variants of `decoderonly_grid_hvg.sbatch` per FM |
| `experiments/03_latent_shift/` | configs: `runs/configs/cf/paper.yaml`, `runs/configs/st/pbmc_train.toml`<br>submit: `runs/scripts/train/launch_cf_sweep.py`, `launch_st_sweep.sh` |

Path rewriting: replace hardcoded `/lustre/...` paths with `$RECONEVAL_ROOT` env var; document this in `experiments/README.md`.

## 5. `evaluation/` folder

Same shape as `experiments/`; lifted from `runs/scripts/eval/`:

| Public task path | Private source |
|---|---|
| `evaluation/01_end_to_end/` | `runs/scripts/eval/{decoder_statistical,decoder_biological,recon_knn}.sbatch` |
| `evaluation/02_foundation_model/` | same eval pipeline, parameterised by FM checkpoint |
| `evaluation/03_latent_shift/` | `runs/scripts/eval/{cf_eval_pipeline.sh, st_eval_pipeline.sh, cf_quality_check.sbatch}` |

Each `submit/<model>.sh` reads the checkpoint produced by the matching `experiments/<task>/submit/<model>.sh`, runs the metrics pipeline on the held-out split, writes a CSV under `results/<task>/<model>/`.

## 6. Packaging mechanics (`pyproject.toml`)

```toml
[project]
name = "screconstruction-tools"
requires-python = ">=3.10"
dependencies = [
  "anndata", "numpy", "pandas", "scanpy", "decoupler", "zarr", "tqdm",
]

[project.optional-dependencies]
metrics       = ["scikit-learn", "torch"]           # metrics.api + heavy _batch
fm-scgpt      = ["scgpt", "torch>=2.0"]
fm-scconcept  = [...]
fm-scimilarity= ["scimilarity"]
fm-se         = ["state-embeddings"]
latent-shift  = ["cellflow", "jax", "torch>=2.0"]
dev           = ["pytest", "ruff", "hatchling"]
```

User stories:

- "Compute metrics on my pair": `pip install screconstruction-tools[metrics]`. One env. Tutorial runs.
- "Run scGPT → MLP decoder → metrics": `pip install screconstruction-tools[metrics,fm-scgpt]` inside the scGPT env.
- "Retrain everything": clone the repo, use `envs/*.yaml` conda files, run `experiments/<task>/submit/<model>.sh`.

## 7. Order of operations

1. **`make example-data` target + `data/example/extract.py`** — subset `analysis/frozen/` into three `(true, pred)` h5ad pairs (~5 MB each).
2. **Build `metrics/api.py`** light wrapper. Re-export from `metrics/__init__.py`. Rename `base_eval.py` → `_batch.py` and update internal imports.
3. **Write `01b_end_to_end_metrics.ipynb`** using `data/example/e2e/`. Validates the API end-to-end.
4. **Clone to `02b_fm_metrics.ipynb` and `03b_latent_shift_metrics.ipynb`** (only the loading cell differs).
5. **`adapters/fm_protocol.py` + thin facades** `adapters/fm_{scgpt,scconcept,scimilarity,se}.py` over the existing `models/recon*.py` classes.
6. **Lift training YAMLs + sbatch** from private `runs/configs/` + `runs/scripts/train/` → `experiments/<task>/`. Flatten Hydra group includes; replace `/lustre/...` with `$RECONEVAL_ROOT`.
7. **Lift eval scripts** from `runs/scripts/eval/` → `evaluation/<task>/`, same pattern.
8. **Write `envs/*.yaml`** conda files (one per env, mirroring what's working on the cluster).
9. **Extend `pyproject.toml`** with `[project.optional-dependencies]` extras.
10. **Update `README.rst`** — quickstart with the pip install line, links to tutorials, pointers to experiments/evaluation.
11. **Tag `v0.1.0-paper`** and push.
12. (Later, post paper revisions) Fill in setup tutorials 01a/02a/03a.

## 8. What's reused vs new

| Reused (no rewrite) | New |
|---|---|
| `metrics/utils.py` numpy helpers | `metrics/api.py` |
| `metrics/_cellcycle.py`, `_coexpression.py`, `_cytokine.py`, `_deg.py`, `_pathway.py`, `distributional.py`, `loss.py` | `adapters/fm_protocol.py` |
| `metrics/base_eval.py` → `metrics/_batch.py` (renamed only) | `adapters/fm_{scgpt,scconcept,scimilarity,se}.py` (thin facades) |
| `decoders/*` modules | `data/example/extract.py` + 3 (true, pred) pairs |
| `models/recon*.py` (refactored, not rewritten) | `experiments/` tree (lifted, paths rewritten) |
| All `analysis/` notebooks | `evaluation/` tree (lifted, paths rewritten) |
| `Makefile`, `pyproject.toml` skeleton | `envs/*.yaml` (7 files) |

## 9. Verification

After step 2 (light metrics API):
- `from sc_reconstruction.metrics import compute_all_metrics` works in a fresh env containing only `[metrics]` extras.
- `compute_all_metrics(adata_true, adata_pred)` returns a dict of floats matching the per-family functions called individually.

After step 3:
- `jupyter nbconvert --execute tutorials/01b_end_to_end_metrics.ipynb` runs end-to-end without errors on a fresh `[metrics]` env.

After step 4:
- Same nbconvert check on `02b_fm_metrics.ipynb` and `03b_latent_shift_metrics.ipynb`.

After step 5:
- `from sc_reconstruction.adapters.fm_scgpt import ScGPTAdapter; fm = ScGPTAdapter(); fm.load(...); adata2 = fm.embed(adata)` produces `.obsm["X_fm"]` of shape `(n_cells, 512)`.
- End-to-end smoke: embed → decode → metrics in one script.

After steps 6–7:
- One representative `experiments/<task>/submit/<model>.sh` runs on the cluster and writes a checkpoint.
- The matching `evaluation/<task>/submit/<model>.sh` reads that checkpoint and writes a CSV.

After step 10–11:
- A fresh clone + `pip install ".[metrics]"` lets a new user run the three metrics tutorials.

## 10. Notable risks / open questions

- **Heavy metric envs**: some "biological" metrics depend on `decoupler` + PROGENy weights + MSigDB — verify the `[metrics]` extra pulls everything; the README must point to the PROGENy / MSigDB downloads.
- **Conda yaml drift**: the private cluster's working envs may pin packages no longer on PyPI/conda-forge. Audit each `envs/*.yaml` and lock to a reproducible set.
- **License of FM-specific weights**: `fm-scgpt` etc. install the *code*; users still need to obtain the *weights* — document in `tutorials/02a_fm_setup.ipynb` placeholder (filled later).
- **Backward-compat of metrics**: if anyone reproduces the paper from the current `analysis/data/plots/fig*_clean.ipynb`, those notebooks call the heavy `MetricsBatchEvaluator` directly — renaming `base_eval.py` → `_batch.py` is fine because they import via `from sc_reconstruction.metrics.base_eval import ...`. Update the imports there too (one-line change per notebook).
