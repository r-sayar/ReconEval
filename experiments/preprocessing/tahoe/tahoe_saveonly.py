import os, gc, h5py
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from collections import Counter
import time
from tqdm import tqdm
# ---- Config ----
N_PLATES = 13
PLATE_PATH = "/lustre/groups/ml01/workspace/ken_wang/100m/plate{ID}_filtered.h5ad"
OUT_DIR = "/lustre/groups/ml01/projects/big_perturbation/datasets"
MERGED_OUT = f"{OUT_DIR}/tahoe.h5ad"

SPARSE_CHUNK_SIZE = 1_000_000


# PATH = '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/raw/tahoe_3000wCCM.h5ad'
# with h5py.File(PATH, "r") as f:
#     adata = ad.AnnData(
#         obs=ad.io.read_elem(f["obs"]),
#         var=ad.io.read_elem(f["var"]),
#     )

# selected_genes = adata.var_names.tolist()

# print(f"Selected genes (union, first-seen order, no sort): {len(selected_genes)}")

# cell_cycle_genes = [x.strip() for x in open('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/sup/regev_lab_cell_cycle_genes.txt')]
# feature_genes = list(set(selected_genes + cell_cycle_genes))
# feature_genes = selected_genes
# print(f"Feature genes (added cell cycle marker genes): {len(feature_genes)}")

out_paths = {}
for i in tqdm(range(0, N_PLATES),' desc="Pass 2: per-plate filter/reorder→write"):'):
    pid = str(i + 1)
    in_path = PLATE_PATH.format(ID=pid)

    # with h5py.File(in_path, "r") as f:
    #     adata = ad.AnnData(
    #         obs=ad.io.read_elem(f["obs"]),
    #         var=ad.io.read_elem(f["var"]),
    #     )
    #     adata.X = ad.experimental.read_elem_as_dask(
    #         f["X"], chunks=(SPARSE_CHUNK_SIZE, adata.shape[1])
    #     )

    # # filter cells
    # row_mask = (adata.obs["pass_filter"] == "full").values
    # adata = adata[row_mask, :].copy()
    # # sc.pp.normalize_total(adata, target_sum = 1e4)
    # # sc.pp.log1p(adata)
    
    # adata = adata[:, feature_genes].copy()
    # import ast
    # s = adata.obs["drugname_drugconc"]  
    # dose_map = {lbl: float(ast.literal_eval(lbl)[0][1]) for lbl in s.cat.categories}
    # adata.obs["dosage"] = s.map(dose_map)    

    out_path = f"{OUT_DIR}/plate{pid}_preprocessed.h5ad"
    # adata.write_h5ad(out_path)
    out_paths[f"plate_{pid}"] = out_path

    # del adata
    gc.collect()

ad.experimental.concat_on_disk(
    out_paths,
    MERGED_OUT,
    label="plate",
)
print(f"Done → {MERGED_OUT}")