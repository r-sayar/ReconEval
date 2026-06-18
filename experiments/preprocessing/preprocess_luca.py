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

from tqdm.notebook import tqdm
from dask.diagnostics import ProgressBar
from sklearn.model_selection import train_test_split


# In[2]:


def sanitize_key(key):
    return re.sub(r'[^\w]', '_', str(key)).strip('_')


# In[ ]:


adata = sc.read_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/lucav2/core_240701.h5ad')


# In[32]:


adata.obs['cell_type'].unique()


# In[ ]:


# import cellxgene_census as census

# versions = census.get_census_version_directory()
# print(versions)  # shows available builds

# census_version = "2024-07-01"
# dataset_id = "232f6a5a-a04c-4758-a6e8-88ab2e3a6e69"

# census.download_source_h5ad(
#     dataset_id=dataset_id,
#     census_version=census_version,
#     to_path="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/lucav2/core_240701.h5ad",
# )


# In[5]:


adata


# In[17]:


adata = sc.read_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/lucav2/core_250130.h5ad')
adata.X = adata.layers['count']      
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
hv_mask = adata.var['is_highly_variable'] == 'True'
hv_idx  = np.nonzero(hv_mask)[0]
adata_hvg = adata[:, hv_idx]


# In[22]:


adata_hvg.write_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/luca.h5ad')


# In[25]:


adata_hvg.obs['cell_id'] = adata_hvg.obs.index.astype(str)
adata_hvg.obs.reset_index(drop=True, inplace=True)
for col in ['cell_type','dataset','origin']:
    adata_hvg.obs[col] = adata_hvg.obs[col].astype(str)

adata_hvg.obs['comb_key'] = adata_hvg.obs.apply(
    lambda r: "-".join([
        sanitize_key(r['cell_type']),
        sanitize_key(r['dataset']),
        sanitize_key(r['origin'])
    ]),
    axis=1
)


# In[28]:


adata_hvg.obs['comb_key']


# In[31]:


output_dir = '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/comb_w_obs.zarr'


# In[32]:


os.makedirs(output_dir, exist_ok=True)
n_vars = adata_hvg.X.shape[1]

root_zarr = zarr.open(output_dir, mode='a')
root_zarr.attrs['var_names'] = adata_hvg.var['feature_name'].tolist()


# In[33]:


current_combinations = set()

for combination_key, group in tqdm(adata_hvg.obs.groupby('comb_key')):
    current_combinations.add(combination_key)
    x_group = adata_hvg.X[group.index].toarray()

    da.from_array(x_group, chunks=-1).astype(adata_hvg.X.dtype).to_zarr(output_dir+'/'+combination_key, component='X', overwrite=True)
    da.from_array(group['cell_id'].values, chunks=-1).astype('<U40').to_zarr(output_dir+'/'+combination_key, component='obs_index', overwrite=True)

existing_combinations = set(root_zarr.attrs.get('combinations', []))
existing_combinations.update(current_combinations)
root_zarr.attrs['combinations'] = sorted(existing_combinations)


# In[35]:


#!/usr/bin/env python3
from pathlib import Path

root = Path("/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/comb_w_obs.zarr")
old = "multi_ciliated_epithelial"
new = "multiciliated_epithelial"

# dry run
targets = sorted([p for p in root.iterdir() if p.name.startswith(old)])
for p in targets:
    print(f"would rename: {p.name} -> {p.name.replace(old, new, 1)}")

# rename
for p in targets:
    dst = p.with_name(p.name.replace(old, new, 1))
    p.rename(dst)
    print(f"renamed: {p.name} -> {dst.name}")


# **split01 - combination**

# In[36]:


CHUNK_SIZE = 50_000

source_dir = '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/comb_w_obs.zarr'
output_dir = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split03"

train_ratio=0.7
val_ratio=0.15
test_ratio=0.15

seed=42
adata_hvg = sc.read_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/luca.h5ad')


# In[38]:


# print(f"Processing combinations from {source_dir}")
# os.makedirs(output_dir, exist_ok=True)

comb_dir = source_dir
# if not os.path.isdir(comb_dir):
#     raise ValueError(f"Combination directory not found: {comb_dir}")

# combinations = [d for d in os.listdir(comb_dir) if not d.startswith('.')]
# print(f"Found {len(combinations)} combination directories")

# train_val_ratio = train_ratio + val_ratio
# train_val_combs, test_combs = train_test_split(
#     combinations, 
#     train_size=train_val_ratio,
#     random_state=seed
# )

# train_combs, val_combs = train_test_split(
#     train_val_combs,
#     train_size=train_ratio/train_val_ratio,
#     random_state=seed
# )

# print(f"Split: {len(train_combs)} train, {len(val_combs)} val, {len(test_combs)} test")
meta_data = zarr.open('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/lucav2/split03/split_metadata.zarr')

meta_save_path = os.path.join(output_dir, "split_metadata.zarr")
meta_group = zarr.open_group(meta_save_path, mode='w')
meta_group.attrs['train_combinations'] = meta_data.attrs['train_combinations']
meta_group.attrs['val_combinations'] = meta_data.attrs['val_combinations']
meta_group.attrs['test_combinations'] = meta_data.attrs['test_combinations']
meta_group.attrs['random_seed'] = seed

print("Processing validation data...")
val_arrays = []
for comb in meta_data.attrs['val_combinations']:
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
for comb in meta_data.attrs['test_combinations']:
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
for comb in meta_data.attrs['train_combinations']:
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


# **split02 - cell_type**

# In[39]:


CHUNK_SIZE = 50_000

source_dir = '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/comb_w_obs.zarr'
output_dir = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split02"

train_ratio=0.7
val_ratio=0.15
test_ratio=0.15

seed=42


# In[40]:


# print(f"Processing combinations from {source_dir}")
# os.makedirs(output_dir, exist_ok=True)

# comb_dir = source_dir
# if not os.path.isdir(comb_dir):
#     raise ValueError(f"Combination directory not found: {comb_dir}")

# combinations = [d for d in os.listdir(comb_dir) if not d.startswith('.')]
# cell_types = np.unique([d.split("-")[0] for d in combinations])
# print(f"Found {len(combinations)} combination directories")
# print(f"Found {len(cell_types)} cell types")

# train_val_ratio = train_ratio + val_ratio
# train_val_types, test_types = train_test_split(
#     cell_types, 
#     train_size=train_val_ratio,
#     random_state=seed
# )

# train_types, val_types = train_test_split(
#     train_val_types,
#     train_size=train_ratio/train_val_ratio,
#     random_state=seed
# )

# train_combs = []
# test_combs = []
# val_combs = []
# for combination in combinations:
#     if combination.split("-")[0] in train_types:
#         train_combs.append(combination)
#     elif combination.split("-")[0] in test_types:
#         test_combs.append(combination)
#     elif combination.split("-")[0] in val_types:
#         val_combs.append(combination)
#     else:
#         raise Exception(f"cell_type {combination.split('-')[0]} not found")

# print(f"Split: {len(train_combs)} train, {len(val_combs)} val, {len(test_combs)} test")


# In[41]:


meta_data = zarr.open('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/lucav2/split02/split_metadata.zarr')
train_combs = meta_data.attrs['train_combinations']
val_combs = meta_data.attrs['val_combinations']
test_combs = meta_data.attrs['test_combinations']

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


# **split03 - dataset**

# In[42]:


CHUNK_SIZE = 50_000

source_dir = '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/comb_w_obs.zarr'
output_dir = "/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split01"

train_ratio=0.7
val_ratio=0.15
test_ratio=0.15

seed=42


# In[43]:


# print(f"Processing combinations from {source_dir}")
# os.makedirs(output_dir, exist_ok=True)

# comb_dir = source_dir
# if not os.path.isdir(comb_dir):
#     raise ValueError(f"Combination directory not found: {comb_dir}")

# combinations = [d for d in os.listdir(comb_dir) if not d.startswith('.')]
# datasets = np.unique([d.split("-")[1] for d in combinations])
# print(f"Found {len(combinations)} combination directories")
# print(f"Found {len(cell_types)} cell types")

# train_val_ratio = train_ratio + val_ratio
# train_val_datasets, test_datasets = train_test_split(
#     datasets, 
#     train_size=train_val_ratio,
#     random_state=seed
# )

# train_datasets, val_datasets = train_test_split(
#     train_val_datasets,
#     train_size=train_ratio/train_val_ratio,
#     random_state=seed
# )

# train_combs = []
# test_combs = []
# val_combs = []
# for combination in combinations:
#     if combination.split("-")[1] in train_datasets:
#         train_combs.append(combination)
#     elif combination.split("-")[1] in test_datasets:
#         test_combs.append(combination)
#     elif combination.split("-")[1] in val_datasets:
#         val_combs.append(combination)
#     else:
#         raise Exception(f"cell_type {combination.split('-')[0]} not found")

# print(f"Split: {len(train_combs)} train, {len(val_combs)} val, {len(test_combs)} test")


# In[44]:


meta_data = zarr.open('/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/lucav2/split01/split_metadata.zarr')
train_combs = meta_data.attrs['train_combinations']
val_combs = meta_data.attrs['val_combinations']
test_combs = meta_data.attrs['test_combinations']

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


# In[48]:


import zarr
root_paths = [  '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split01/test.zarr',
                '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split01/val.zarr',
                '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split01/train.zarr',
                '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split02/test.zarr',
                '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split02/val.zarr',
                '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split02/train.zarr',
                '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split03/test.zarr',
                '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split03/val.zarr',
                '/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/luca_counts/split03/train.zarr'
            ]
for root_path in root_paths:
    zarr_root = zarr.open(root_path, mode='r+')
    X = zarr_root['X'][:]
    sum_counts = X.sum(axis=(1))
    masked_log_sum = np.ma.log(sum_counts)
    log_counts = masked_log_sum.filled(0)
    library_log_mean = np.mean(log_counts)
    library_log_var = np.var(log_counts)
    zarr_root.attrs['library_log_mean'] = library_log_mean
    zarr_root.attrs['library_log_var'] = library_log_var


# In[51]:


library_log_var


# In[ ]:




