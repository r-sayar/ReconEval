# Tutorials

| Notebook | What it covers |
|---|---|
| [`metrics.ipynb`](metrics.ipynb) | The user-facing metrics API — every metric one at a time, the one-call wrapper, and the rank-percentile aggregation. Same API across all three benchmark settings. |
| [`end_to_end.ipynb`](end_to_end.ipynb) | End-to-end reconstruction. The 2-method Protocol any model must satisfy, AE as the reference implementation (train → reconstruct → score). |
| [`fm.ipynb`](fm.ipynb) | Foundation-model reconstruction as a two-step pipeline: FM-specific embed step + FM-agnostic decoder step. SE active, the other three FMs commented for easy switching. |
| [`latent_shift.ipynb`](latent_shift.ipynb) | Latent-shift (perturbation prediction). 2-method `PerturbationPredictor` Protocol, a small MLP predictor as the runnable reference (CellFlow / STATE shown but commented), KNN purity. |
