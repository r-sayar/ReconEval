#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zarr
import re
import os
import gc
import time

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da
import pandas as pd

from tqdm.notebook import tqdm
from dask.diagnostics import ProgressBar
from sklearn.model_selection import train_test_split


# In[2]:


def sanitize_key(key):
    return re.sub(r'[^\w]', '_', str(key)).strip('_')

# def init_combination_group(root_zarr, combination_key, n_vars, dtype):
#     if combination_key not in root_zarr:
#         group = root_zarr.create_group(combination_key)
#         group.create_dataset(
#             'X',
#             shape=(0, n_vars),
#             dtype=dtype,
#             compressor=zarr.Blosc(cname='zstd', clevel=3)
#         )
#         group.create_dataset(
#             'obs_index',
#             shape=(0,),
#             dtype='<U40',
#         )
#         group.attrs['current_row'] = 0
#         group.attrs['n_vars'] = n_vars
#     else:
#         group = root_zarr[combination_key]
#     return group


# In[3]:


PATH = "/lustre/groups/ml01/workspace/ten_million/data/final_data/adata_notebook_input.h5ad"
adata = sc.read_h5ad(PATH)


# In[2]:


# adata = sc.read_h5ad('/ictstr01/groups/ml01/workspace/manuel.lubetzki/MPII/hlca/dbb5ad81-1713-4aee-8257-396fbabe7c6e.h5ad')


# In[5]:


sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


# In[ ]:


adata.X


# In[8]:


gc.collect()
time.sleep(10)


# In[9]:


sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.25)
sc.pl.highly_variable_genes(adata)

adata.raw = adata


# In[10]:


adata = adata[:, adata.var.highly_variable].copy()


# In[5]:


adata


# In[7]:


adata.obs['cell_id'] = adata.obs.index
adata.obs = adata.obs.reset_index(drop=True)


# In[8]:


new_cell_type = pd.read_csv("/lustre/groups/ml01/workspace/ten_million/data/data_2024_12_16/new_cell_type_annotations.csv")
d = new_cell_type[["Unnamed: 0", "cell_type_new_coarse"]].drop_duplicates().set_index("Unnamed: 0")['cell_type_new_coarse'].to_dict()
adata.obs["cell_type_new"] = adata.obs["cell_id"].apply(lambda x: d[x])
print(len(d))


# In[9]:


del new_cell_type, d
gc.collect()
time.sleep(10)


# In[10]:


adata.obs.cell_type_new.value_counts()


# In[11]:


adata.obs.loc[:, 'combination_key'] = adata.obs.apply(
    lambda row: f"{sanitize_key(row['cell_type_new'])}-"
                f"{sanitize_key(row['donor'])}-"
                f"{sanitize_key(row['cytokine'])}",
    axis=1
)


# In[12]:


output_dir = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc_wg/comb_w_obs.zarr"


# In[13]:


os.makedirs(output_dir, exist_ok=True)
n_vars = adata.X.shape[1]
CHUNK_SIZE = 10_000
root_zarr = zarr.open(output_dir, mode='a')
root_zarr.attrs['var_names'] = adata.var.index.to_list() # adata.var['feature_name'].tolist()

current_combinations = set()

for combination_key, group in tqdm(adata.obs.groupby('combination_key')):
    current_combinations.add(combination_key)
    x_group = adata.X[group.index].toarray()

    da.from_array(x_group, chunks=(CHUNK_SIZE, n_vars)).astype(adata.X.dtype).to_zarr(output_dir+'/'+combination_key, component='X', overwrite=True)
    da.from_array(group['cell_id'].values, chunks=-1).astype('<U40').to_zarr(output_dir+'/'+combination_key, component='obs_index', overwrite=True)

existing_combinations = set(root_zarr.attrs.get('combinations', []))
existing_combinations.update(current_combinations)
root_zarr.attrs['combinations'] = sorted(existing_combinations)


# **best seed choice**

# In[52]:


for seed in range(30, 51):
    print("SEED:", seed)
    print(f"Processing combinations from {source_dir}")
    os.makedirs(output_dir, exist_ok=True)

    comb_dir = source_dir
    if not os.path.isdir(comb_dir):
        raise ValueError(f"Combination directory not found: {comb_dir}")

    combinations = [d for d in os.listdir(comb_dir) if not d.startswith('.')]
    cell_types = np.unique([d.split("-")[0] for d in combinations])
    print(f"Found {len(combinations)} combination directories")
    print(f"Found {len(cell_types)} cell types")

    train_val_ratio = train_ratio + val_ratio
    train_val_types, test_types = train_test_split(
        cell_types, 
        train_size=train_val_ratio,
        random_state=seed
    )

    train_types, val_types = train_test_split(
        train_val_types,
        train_size=train_ratio/train_val_ratio,
        random_state=seed
    )

    train_combs = []
    test_combs = []
    val_combs = []
    for combination in combinations:
        if combination.split("-")[0] in train_types:
            train_combs.append(combination)
        elif combination.split("-")[0] in test_types:
            test_combs.append(combination)
        elif combination.split("-")[0] in val_types:
            val_combs.append(combination)
        else:
            raise Exception(f"cell_type {combination.split('-')[0]} not found")

    print(f"Split: {len(train_combs)} train, {len(val_combs)} val, {len(test_combs)} test")

    val_arrays = []
    for comb in val_combs:
        comb_path = os.path.join(comb_dir, comb)
        val_arrays.append(da.from_zarr(comb_path, 'X'))

    val_data = da.concatenate(val_arrays, axis=0)
    print(f"Validation data shape: {val_data.shape}")

    test_arrays = []
    for comb in test_combs:
        comb_path = os.path.join(comb_dir, comb)
        test_arrays.append(da.from_zarr(comb_path, 'X'))

    test_data = da.concatenate(test_arrays,  axis=0)
    print(f"Test data shape: {test_data.shape}")
    print(f"Train data shape: ({len(adata) - test_data.shape[0] - val_data.shape[0]}, 5412)")
    print()


# 31, 33, 39

# In[ ]:





# **split01 - cell_type**

# In[19]:


CHUNK_SIZE = 50_000

source_dir = '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/comb_w_obs.zarr'
output_dir = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/split01"

train_ratio=0.7
val_ratio=0.15
test_ratio=0.15

seed=33


# In[20]:


print(f"Processing combinations from {source_dir}")
os.makedirs(output_dir, exist_ok=True)

comb_dir = source_dir
if not os.path.isdir(comb_dir):
    raise ValueError(f"Combination directory not found: {comb_dir}")

combinations = [d for d in os.listdir(comb_dir) if not d.startswith('.')]
cell_types = np.unique([d.split("-")[0] for d in combinations])
print(f"Found {len(combinations)} combination directories")
print(f"Found {len(cell_types)} cell types")

train_val_ratio = train_ratio + val_ratio
train_val_types, test_types = train_test_split(
    cell_types, 
    train_size=train_val_ratio,
    random_state=seed
)

train_types, val_types = train_test_split(
    train_val_types,
    train_size=train_ratio/train_val_ratio,
    random_state=seed
)

train_combs = []
test_combs = []
val_combs = []
for combination in combinations:
    if combination.split("-")[0] in train_types:
        train_combs.append(combination)
    elif combination.split("-")[0] in test_types:
        test_combs.append(combination)
    elif combination.split("-")[0] in val_types:
        val_combs.append(combination)
    else:
        raise Exception(f"cell_type {combination.split('-')[0]} not found")

print(f"Split: {len(train_combs)} train, {len(val_combs)} val, {len(test_combs)} test")


# In[21]:


meta_save_path = os.path.join(output_dir, "split_metadata.zarr")
meta_group = zarr.open_group(meta_save_path, mode='w')
meta_group.attrs['train_combinations'] = train_combs
meta_group.attrs['val_combinations'] = val_combs
meta_group.attrs['test_combinations'] = test_combs
meta_group.attrs['random_seed'] = seed

print("Processing validation data...")
val_arrays = []
for comb in val_combs:
    comb_path = os.path.join(comb_dir, comb)
    val_arrays.append(da.from_zarr(comb_path, 'X'))

val_data = da.concatenate(val_arrays, axis=0)
print(f"Validation data shape: {val_data.shape}")

val_save_path = os.path.join(output_dir, 'val.zarr')
print(f"Saving validation data to {val_save_path}")
with ProgressBar():
    val_data.rechunk((CHUNK_SIZE, val_data.shape[1])).to_zarr(val_save_path, component='X', overwrite=True)
print(f"Done saving validation data")

del val_arrays, val_data
gc.collect()
time.sleep(10)

print("Processing test data...")
test_arrays = []
for comb in test_combs:
    comb_path = os.path.join(comb_dir, comb)
    test_arrays.append(da.from_zarr(comb_path, 'X'))

test_data = da.concatenate(test_arrays,  axis=0)
print(f"Test data shape: {test_data.shape}")

test_save_path = os.path.join(output_dir, 'test.zarr')
print(f"Saving test data to {test_save_path}")
with ProgressBar():
    test_data.rechunk((CHUNK_SIZE, test_data.shape[1])).to_zarr(test_save_path, component='X', overwrite=True)
print(f"Done saving test data")

del test_arrays, test_data
gc.collect()
time.sleep(10)

print("Processing training data...")
train_arrays = []
for comb in train_combs:
    comb_path = os.path.join(comb_dir, comb)
    train_arrays.append(da.from_zarr(comb_path, 'X'))

train_data = da.concatenate(train_arrays, axis=0)
print(f"Training data shape: {train_data.shape}")

n_vars = train_data.shape[1]
train_data = train_data.rechunk((CHUNK_SIZE, n_vars))
print(f"Training data rechunked: {train_data.numblocks} blocks")

train_unsfd_save_path = os.path.join(output_dir, 'train_unsfd.zarr')
print(f"Saving unshuffled training data to {train_unsfd_save_path}")
with ProgressBar():
    train_data.to_zarr(train_unsfd_save_path, component='X', overwrite=True)
print(f"Done saving unshuffled training data")

print("Shuffling training data...")
rng = np.random.default_rng(seed=seed)
n = train_data.shape[0]
perm = rng.permutation(n)
indexer = [list(perm[i:i+CHUNK_SIZE]) for i in range(0, n, CHUNK_SIZE)]

train_data_shuffled = da.shuffle(train_data, indexer=indexer, axis=0, chunks='auto')
print(f"Training data shuffled: {train_data_shuffled.numblocks} blocks")

train_save_path = os.path.join(output_dir, 'train.zarr')
with ProgressBar():
    train_data_shuffled.to_zarr(train_save_path, component='X', overwrite=True)
print(f"Done saving shuffled training data")

del train_arrays, train_data, train_data_shuffled
gc.collect()
time.sleep(10)


# In[ ]:





# **best seed choice**

# In[31]:


for seed in range(40, 51):
    print("SEED:", seed)
    print(f"Processing combinations from {source_dir}")
    os.makedirs(output_dir, exist_ok=True)

    comb_dir = source_dir
    if not os.path.isdir(comb_dir):
        raise ValueError(f"Combination directory not found: {comb_dir}")

    combinations = [d for d in os.listdir(comb_dir) if not d.startswith('.')]
    donors = np.unique([d.split("-")[1] for d in combinations])
    print(f"Found {len(combinations)} combination directories")
    print(f"Found {len(donors)} donors")

    train_val_ratio = train_ratio + val_ratio
    train_val_donors, test_donors = train_test_split(
        donors, 
        train_size=train_val_ratio,
        random_state=seed
    )

    train_donors, val_donors = train_test_split(
        train_val_donors,
        train_size=train_ratio/train_val_ratio,
        random_state=seed
    )

    train_combs = []
    test_combs = []
    val_combs = []
    for combination in combinations:
        if combination.split("-")[1] in train_donors:
            train_combs.append(combination)
        elif combination.split("-")[1] in test_donors:
            test_combs.append(combination)
        elif combination.split("-")[1] in val_donors:
            val_combs.append(combination)
        else:
            raise Exception(f"cell_type {combination.split('-')[1]} not found")

    print(f"Split: {len(train_combs)} train, {len(val_combs)} val, {len(test_combs)} test")

    val_arrays = []
    for comb in val_combs:
        comb_path = os.path.join(comb_dir, comb)
        val_arrays.append(da.from_zarr(comb_path, 'X'))

    val_data = da.concatenate(val_arrays, axis=0)
    print(f"Validation data shape: {val_data.shape}")

    test_arrays = []
    for comb in test_combs:
        comb_path = os.path.join(comb_dir, comb)
        test_arrays.append(da.from_zarr(comb_path, 'X'))

    test_data = da.concatenate(test_arrays,  axis=0)
    print(f"Test data shape: {test_data.shape}")
    print(f"Train data shape: ({len(adata) - test_data.shape[0] - val_data.shape[0]}, 5412)")
    print()


# In[ ]:





# **split02 - donor**

# In[32]:


CHUNK_SIZE = 50_000

source_dir = '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/comb_w_obs.zarr'
output_dir = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/split02"

train_ratio=0.7
val_ratio=0.15
test_ratio=0.15

seed=47


# In[33]:


print(f"Processing combinations from {source_dir}")
os.makedirs(output_dir, exist_ok=True)

comb_dir = source_dir
if not os.path.isdir(comb_dir):
    raise ValueError(f"Combination directory not found: {comb_dir}")

combinations = [d for d in os.listdir(comb_dir) if not d.startswith('.')]
donors = np.unique([d.split("-")[1] for d in combinations])
print(f"Found {len(combinations)} combination directories")
print(f"Found {len(donors)} donors")

train_val_ratio = train_ratio + val_ratio
train_val_donors, test_donors = train_test_split(
    donors, 
    train_size=train_val_ratio,
    random_state=seed
)

train_donors, val_donors = train_test_split(
    train_val_donors,
    train_size=train_ratio/train_val_ratio,
    random_state=seed
)

train_combs = []
test_combs = []
val_combs = []
for combination in combinations:
    if combination.split("-")[1] in train_donors:
        train_combs.append(combination)
    elif combination.split("-")[1] in test_donors:
        test_combs.append(combination)
    elif combination.split("-")[1] in val_donors:
        val_combs.append(combination)
    else:
        raise Exception(f"cell_type {combination.split('-')[1]} not found")

print(f"Split: {len(train_combs)} train, {len(val_combs)} val, {len(test_combs)} test")


# In[34]:


meta_save_path = os.path.join(output_dir, "split_metadata.zarr")
meta_group = zarr.open_group(meta_save_path, mode='w')
meta_group.attrs['train_combinations'] = train_combs
meta_group.attrs['val_combinations'] = val_combs
meta_group.attrs['test_combinations'] = test_combs
meta_group.attrs['random_seed'] = seed

print("Processing validation data...")
val_arrays = []
for comb in val_combs:
    comb_path = os.path.join(comb_dir, comb)
    val_arrays.append(da.from_zarr(comb_path, 'X'))

val_data = da.concatenate(val_arrays, axis=0)
print(f"Validation data shape: {val_data.shape}")

val_save_path = os.path.join(output_dir, 'val.zarr')
print(f"Saving validation data to {val_save_path}")
with ProgressBar():
    val_data.rechunk((CHUNK_SIZE, val_data.shape[1])).to_zarr(val_save_path, component='X', overwrite=True)
print(f"Done saving validation data")

del val_arrays, val_data
gc.collect()
time.sleep(10)

print("Processing test data...")
test_arrays = []
for comb in test_combs:
    comb_path = os.path.join(comb_dir, comb)
    test_arrays.append(da.from_zarr(comb_path, 'X'))

test_data = da.concatenate(test_arrays,  axis=0)
print(f"Test data shape: {test_data.shape}")

test_save_path = os.path.join(output_dir, 'test.zarr')
print(f"Saving test data to {test_save_path}")
with ProgressBar():
    test_data.rechunk((CHUNK_SIZE, test_data.shape[1])).to_zarr(test_save_path, component='X', overwrite=True)
print(f"Done saving test data")

del test_arrays, test_data
gc.collect()
time.sleep(10)

print("Processing training data...")
train_arrays = []
for comb in train_combs:
    comb_path = os.path.join(comb_dir, comb)
    train_arrays.append(da.from_zarr(comb_path, 'X'))

train_data = da.concatenate(train_arrays, axis=0)
print(f"Training data shape: {train_data.shape}")

n_vars = train_data.shape[1]
train_data = train_data.rechunk((CHUNK_SIZE, n_vars))
print(f"Training data rechunked: {train_data.numblocks} blocks")

train_unsfd_save_path = os.path.join(output_dir, 'train_unsfd.zarr')
print(f"Saving unshuffled training data to {train_unsfd_save_path}")
with ProgressBar():
    train_data.to_zarr(train_unsfd_save_path, component='X', overwrite=True)
print(f"Done saving unshuffled training data")

print("Shuffling training data...")
rng = np.random.default_rng(seed=seed)
n = train_data.shape[0]
perm = rng.permutation(n)
indexer = [list(perm[i:i+CHUNK_SIZE]) for i in range(0, n, CHUNK_SIZE)]

train_data_shuffled = da.shuffle(train_data, indexer=indexer, axis=0, chunks='auto')
print(f"Training data shuffled: {train_data_shuffled.numblocks} blocks")

train_save_path = os.path.join(output_dir, 'train.zarr')
with ProgressBar():
    train_data_shuffled.to_zarr(train_save_path, component='X', overwrite=True)
print(f"Done saving shuffled training data")

del train_arrays, train_data, train_data_shuffled
gc.collect()
time.sleep(10)


# In[ ]:





# **split03 - combination**

# In[25]:


CHUNK_SIZE = 50_000

source_dir = '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/comb_w_obs.zarr'
output_dir = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/split03"

train_ratio=0.7
val_ratio=0.15
test_ratio=0.15

seed=42


# In[26]:


print(f"Processing combinations from {source_dir}")
os.makedirs(output_dir, exist_ok=True)

comb_dir = source_dir
if not os.path.isdir(comb_dir):
    raise ValueError(f"Combination directory not found: {comb_dir}")

combinations = [d for d in os.listdir(comb_dir) if not d.startswith('.')]
print(f"Found {len(combinations)} combination directories")

train_val_ratio = train_ratio + val_ratio
train_val_combs, test_combs = train_test_split(
    combinations, 
    train_size=train_val_ratio,
    random_state=seed
)

train_combs, val_combs = train_test_split(
    train_val_combs,
    train_size=train_ratio/train_val_ratio,
    random_state=seed
)

print(f"Split: {len(train_combs)} train, {len(val_combs)} val, {len(test_combs)} test")

meta_save_path = os.path.join(output_dir, "split_metadata.zarr")
meta_group = zarr.open_group(meta_save_path, mode='w')
meta_group.attrs['train_combinations'] = train_combs
meta_group.attrs['val_combinations'] = val_combs
meta_group.attrs['test_combinations'] = test_combs
meta_group.attrs['random_seed'] = seed

print("Processing validation data...")
val_arrays = []
for comb in val_combs:
    comb_path = os.path.join(comb_dir, comb)
    val_arrays.append(da.from_zarr(comb_path, 'X'))

val_data = da.concatenate(val_arrays, axis=0)
print(f"Validation data shape: {val_data.shape}")

val_save_path = os.path.join(output_dir, 'val.zarr')
print(f"Saving validation data to {val_save_path}")
with ProgressBar():
    val_data.rechunk((CHUNK_SIZE, val_data.shape[1])).to_zarr(val_save_path, component='X', overwrite=True)
print(f"Done saving validation data")

del val_arrays, val_data
gc.collect()
time.sleep(10)

print("Processing test data...")
test_arrays = []
for comb in test_combs:
    comb_path = os.path.join(comb_dir, comb)
    test_arrays.append(da.from_zarr(comb_path, 'X'))

test_data = da.concatenate(test_arrays,  axis=0)
print(f"Test data shape: {test_data.shape}")

test_save_path = os.path.join(output_dir, 'test.zarr')
print(f"Saving test data to {test_save_path}")
with ProgressBar():
    test_data.rechunk((CHUNK_SIZE, test_data.shape[1])).to_zarr(test_save_path, component='X', overwrite=True)
print(f"Done saving test data")

del test_arrays, test_data
gc.collect()
time.sleep(10)

print("Processing training data...")
train_arrays = []
for comb in train_combs:
    comb_path = os.path.join(comb_dir, comb)
    train_arrays.append(da.from_zarr(comb_path, 'X'))

train_data = da.concatenate(train_arrays, axis=0)
print(f"Training data shape: {train_data.shape}")

n_vars = train_data.shape[1]
train_data = train_data.rechunk((CHUNK_SIZE, n_vars))
print(f"Training data rechunked: {train_data.numblocks} blocks")

train_unsfd_save_path = os.path.join(output_dir, 'train_unsfd.zarr')
print(f"Saving unshuffled training data to {train_unsfd_save_path}")
with ProgressBar():
    train_data.to_zarr(train_unsfd_save_path, component='X', overwrite=True)
print(f"Done saving unshuffled training data")

print("Shuffling training data...")
rng = np.random.default_rng(seed=seed)
n = train_data.shape[0]
perm = rng.permutation(n)
indexer = [list(perm[i:i+CHUNK_SIZE]) for i in range(0, n, CHUNK_SIZE)]

train_data_shuffled = da.shuffle(train_data, indexer=indexer, axis=0, chunks='auto')
print(f"Training data shuffled: {train_data_shuffled.numblocks} blocks")

train_save_path = os.path.join(output_dir, 'train.zarr')
with ProgressBar():
    train_data_shuffled.to_zarr(train_save_path, component='X', overwrite=True)
print(f"Done saving shuffled training data")

del train_arrays, train_data, train_data_shuffled
gc.collect()
time.sleep(10)


# In[ ]:





# In[27]:


# CHUNK_SIZE = 100_000

# def split_and_save_combinations(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
#     """
#     Split combination zarr files into train/val/test and save them

#     Args:
#         source_dir: Directory containing comb_w_obs/cl_drg_dos.zarr directory
#         output_dir: Directory to save output files
#         train_ratio: Ratio of data for training
#         val_ratio: Ratio of data for validation
#         test_ratio: Ratio of data for testing
#         seed: Random seed for reproducibility
#     """
#     print(f"Processing combinations from {source_dir}")
#     os.makedirs(output_dir, exist_ok=True)

#     # The combination zarr files are in a subdirectory
#     comb_dir = source_dir
#     if not os.path.isdir(comb_dir):
#         raise ValueError(f"Combination directory not found: {comb_dir}")

#     # List all combination zarr directories
#     combinations = [d for d in os.listdir(comb_dir) if not d.startswith('.')]
#     print(f"Found {len(combinations)} combination directories")

#     # Split combinations into train/val/test sets
#     train_val_ratio = train_ratio + val_ratio
#     train_val_combs, test_combs = train_test_split(
#         combinations, 
#         train_size=train_val_ratio,
#         random_state=seed
#     )

#     train_combs, val_combs = train_test_split(
#         train_val_combs,
#         train_size=train_ratio/train_val_ratio,
#         random_state=seed
#     )

#     print(f"Split: {len(train_combs)} train, {len(val_combs)} val, {len(test_combs)} test")

#     # Save metadata
#     meta_save_path = os.path.join(output_dir, "split_metadata.json")
#     meta_group = zarr.open_group(meta_save_path, mode='w')
#     meta_group.attrs['train_combinations'] = train_combs
#     meta_group.attrs['val_combinations'] = val_combs
#     meta_group.attrs['test_combinations'] = test_combs
#     meta_group.attrs['random_seed'] = seed

#     # Load and concatenate validation data
#     print("Processing validation data...")
#     val_arrays = []
#     for comb in val_combs:
#         comb_path = os.path.join(comb_dir, comb)
#         # print(f"Loading validation data from: {comb_path}")
#         val_arrays.append(da.from_zarr(comb_path, 'X'))

#     val_data = da.concatenate(val_arrays, axis=0)
#     print(f"Validation data shape: {val_data.shape}")

#     # Save validation data
#     val_save_path = os.path.join(output_dir, 'val.zarr')
#     print(f"Saving validation data to {val_save_path}")
#     with ProgressBar():
#         val_data.to_zarr(val_save_path, overwrite=True)
#     print(f"Done saving validation data")

#     # Clear memory
#     del val_arrays, val_data
#     gc.collect()
#     time.sleep(10)  # Give time for memory to be released

#     # Load and concatenate test data
#     print("Processing test data...")
#     test_arrays = []
#     for comb in test_combs:
#         comb_path = os.path.join(comb_dir, comb)
#         # print(f"Loading test data from: {comb_path}")
#         test_arrays.append(da.from_zarr(comb_path, 'X'))

#     test_data = da.concatenate(test_arrays, axis=0)
#     print(f"Test data shape: {test_data.shape}")

#     # Save test data
#     test_save_path = os.path.join(output_dir, 'test.zarr')
#     print(f"Saving test data to {test_save_path}")
#     with ProgressBar():
#         test_data.to_zarr(test_save_path, overwrite=True)
#     print(f"Done saving test data")

#     # Clear memory
#     del test_arrays, test_data
#     gc.collect()
#     time.sleep(10)  # Give time for memory to be released

#     # Load and concatenate training data (unshuffled)
#     print("Processing training data...")
#     train_arrays = []
#     for comb in train_combs:
#         comb_path = os.path.join(comb_dir, comb)
#         # print(f"Loading training data from: {comb_path}")
#         train_arrays.append(da.from_zarr(comb_path, 'X'))

#     train_data = da.concatenate(train_arrays, axis=0)
#     print(f"Training data shape: {train_data.shape}")

#     # Rechunk for better performance
#     n_vars = train_data.shape[1]
#     train_data = train_data.rechunk((CHUNK_SIZE, n_vars))
#     print(f"Training data rechunked: {train_data.numblocks} blocks")

#     # Save unshuffled training data
#     train_unsfd_save_path = os.path.join(output_dir, 'train_unsfd.zarr')
#     print(f"Saving unshuffled training data to {train_unsfd_save_path}")
#     with ProgressBar():
#         train_data.to_zarr(train_unsfd_save_path, overwrite=True)
#     print(f"Done saving unshuffled training data")

#     # Shuffle and save training data
#     print("Shuffling training data...")
#     # Generate a permutation of indices
#     rng = np.random.default_rng(seed=seed)
#     n = train_data.shape[0]
#     perm = rng.permutation(n)

#     # Create indexers for chunked permutation
#     indexer = [list(perm[i:i+CHUNK_SIZE]) for i in range(0, n, CHUNK_SIZE)]
#     train_data_shuffled = da.shuffle(train_data, indexer=indexer, axis=0, chunks='auto')
#     print(f"Training data shuffled: {train_data_shuffled.numblocks} blocks")

#     # Save in segments to avoid memory issues
#     total_rows = train_data_shuffled.shape[0]
#     segment_count = 5
#     segment_size = total_rows // segment_count

#     temp_segments = []
#     for i in range(segment_count):
#         start = i * segment_size
#         end = total_rows if i == segment_count - 1 else (i + 1) * segment_size
#         temp_path = os.path.join(output_dir, f'train_temp_{i}.zarr')
#         temp_segments.append(temp_path)

#         print(f"Saving segment {i+1}/{segment_count} (rows {start} to {end}) to {temp_path}")
#         with ProgressBar():
#             train_data_shuffled[start:end].to_zarr(temp_path, overwrite=True)

#         print(f"Segment {i+1}/{segment_count} saved")
#         gc.collect()
#         time.sleep(10)  # Give time for memory to be released

#     # Combine all segments into the final train.zarr file
#     print("Combining all segments into train.zarr")
#     arrays = [da.from_zarr(segment) for segment in temp_segments]
#     combined = da.concatenate(arrays, axis=0)

#     train_save_path = os.path.join(output_dir, 'train.zarr')
#     with ProgressBar():
#         combined.to_zarr(train_save_path, overwrite=True)

#     # Clean up temporary files
#     print("Cleaning up temporary files")
#     for segment in temp_segments:
#         try:
#             import shutil
#             shutil.rmtree(segment)
#             print(f"Removed {segment}")
#         except Exception as e:
#             print(f"Error removing {segment}: {e}")

#     print(f"Process complete. Files saved to {output_dir}")


# In[28]:


# split_and_save_combinations(
#     source_dir="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/comb_w_obs",
#     output_dir="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/split01"
# )


# In[ ]:




