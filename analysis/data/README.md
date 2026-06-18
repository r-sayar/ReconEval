# Analysis

This folder contains the notebooks and helper scripts needed to reproduce the paper figures.

## Layout

- data/frozen/
  - Place the frozen datasets and result tables here (downloaded from Zenodo or prepared locally).
- data/extract/
  - Scripts that generate the frozen datasets from full raw data.
- data/plots/
  - Jupyter notebooks that generate figure panels. No subfolders are used here.

## Figure notebooks

- data/plots/fig2.ipynb
- data/plots/fig2_metric_scaling.ipynb
- data/plots/fig2_summary_panel.ipynb
- data/plots/decoder_eval.ipynb (Figure 3)
- data/plots/cf_scaling_eval.ipynb (Figure 4)
- data/plots/st_scaling_eval.ipynb (Figure 4)

## Frozen data (expected)

Place these under data/frozen/:

- tahoe_umap_input.h5ad
- tahoe_CVCL_0320_fig2b.zarr
- pbmc_umap_input.h5ad
- luca_umap_input.h5ad
- figs/results/*.csv

## Extract scripts

- data/extract/regenerate_summary_csvs.py
- data/extract/utils.py

Run the extract scripts to rebuild the frozen datasets, then execute the notebooks in plots/.
