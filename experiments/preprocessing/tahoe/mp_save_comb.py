import os
import gc
import time
import re
import anndata as ad
import h5py
import zarr
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import numpy as np

CHUNK_SIZE = 100_000

def sanitize_key(key):
    return re.sub(r'[^\w]', '_', str(key)).strip('_')  

def init_combination_group(root_zarr, combination_key, n_vars, dtype):
    if combination_key not in root_zarr:
        group = root_zarr.create_group(combination_key)
        group.create_dataset(
            'X',
            shape=(0, n_vars),
            chunks=(CHUNK_SIZE, n_vars),
            dtype=dtype,
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )
        group.create_dataset(
            'obs_index',
            shape=(0,),
            dtype='<U40',
            chunks=(CHUNK_SIZE,)
        )
        group.create_dataset(
            'plate',
            shape=(0,),
            dtype='<U40',
            chunks=(CHUNK_SIZE,)
        )
        group.attrs['current_row'] = 0
        group.attrs['n_vars'] = n_vars
    else:
        group = root_zarr[combination_key]
    return group

def reorganize_by_combination_worker(shared_data, output_dir, part_index, total_parts):
    os.makedirs(output_dir, exist_ok=True)
    
    n_vars = shared_data['n_vars']
    dtype = shared_data['dtype']
    var_names = shared_data['var_names']
    original_chunk_size = shared_data['original_chunk_size']
    total_chunks = shared_data['total_chunks']
    
    chunks_per_part = total_chunks // total_parts
    start_chunk = part_index * chunks_per_part
    end_chunk = (part_index + 1) * chunks_per_part if part_index < (total_parts - 1) else total_chunks
    
    root_zarr = zarr.open(output_dir, mode='a')
    current_combinations = set()
    
    for chunk_idx in tqdm(range(start_chunk, end_chunk), desc=f"Processing chunks part {part_index}"):
        start = chunk_idx * original_chunk_size
        end = min(start + original_chunk_size, shared_data['n_obs'])
        
        x_chunk = np.frombuffer(shared_data['x_shared'], dtype=dtype, 
                               offset=start * n_vars * np.dtype(dtype).itemsize,
                               count=(end - start) * n_vars).reshape((end - start, n_vars))
        
        obs_chunk = shared_data['obs'][start:end].copy()
        
        for combination_key, group in obs_chunk.groupby('combination_key'):
            current_combinations.add(combination_key)
            relative_indices = group.index - start
            x_group = x_chunk[relative_indices]
            
            group_zarr = init_combination_group(root_zarr, combination_key, n_vars, dtype)
            current_row = group_zarr.attrs['current_row']
            new_rows = len(group)
            new_size = current_row + new_rows
            
            group_zarr['X'].resize((new_size, n_vars))
            group_zarr['obs_index'].resize(new_size)
            group_zarr['plate'].resize(new_size)
            
            group_zarr['X'][current_row:new_size] = x_group
            group_zarr['obs_index'][current_row:new_size] = group['cell_id'].values
            group_zarr['plate'][current_row:new_size] = group['plate'].values
            group_zarr.attrs['current_row'] = new_size
    
    existing_combinations = set(root_zarr.attrs.get('combinations', []))
    existing_combinations.update(current_combinations)
    root_zarr.attrs['combinations'] = sorted(existing_combinations)
    root_zarr.attrs['var_names'] = var_names

def reorganize_by_combination_parallel(adata, output_dir, total_parts=10, num_processes=None):

    n_obs, n_vars = adata.shape
    dtype = adata.X.dtype
    var_names = adata.var_names.tolist()
    
    x_dense = adata.X.todense() if hasattr(adata.X, 'todense') else adata.X
    x_shared = mp.RawArray(np.ctypeslib.as_ctypes_type(dtype), n_obs * n_vars)
    x_np = np.frombuffer(x_shared, dtype=dtype).reshape((n_obs, n_vars))
    np.copyto(x_np, x_dense)
    
    shared_data = {
        'n_obs': n_obs,
        'n_vars': n_vars,
        'dtype': dtype,
        'var_names': var_names,
        'original_chunk_size': adata.X.chunks[0][0] if hasattr(adata.X, 'chunks') else CHUNK_SIZE,
        'total_chunks': adata.X.numblocks[0] if hasattr(adata.X, 'numblocks') else (n_obs + CHUNK_SIZE - 1) // CHUNK_SIZE,
        'x_shared': x_shared,
        'obs': adata.obs  
    }
    
    num_processes = num_processes or mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        pool.map(partial(reorganize_by_combination_worker, shared_data, output_dir), 
                range(total_parts))
    
    print("All parts processed successfully!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Save cell line data by combination key')
    parser.add_argument('--h5-path', default='/lustre/groups/ml01/workspace/weixu.wang/100m/100m_processed_cfp_data.h5ad', help='Path to the input HDF5 file')
    parser.add_argument('--output-dir', default='/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/tahoe/comb_w_obs', help='Directory to save output files')
    parser.add_argument('--total-parts', type=int, default=10, help='Total number of parts')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: CPU count)')
    args = parser.parse_args()

    with h5py.File(args.h5_path, "r") as f:
        adata = ad.AnnData(
            X=ad.experimental.read_elem_as_dask(f["X"], chunks=(1_000_000, -1)),
            obs=ad.io.read_elem(f["obs"]),
        )
    
    adata.obs['cell_id'] = adata.obs.index
    adata.obs = adata.obs.reset_index(drop=True)
    adata.obs.loc[:, 'combination_key'] = adata.obs.apply(
        lambda row: f"{sanitize_key(row['cell_line'])}-"
                    f"{sanitize_key(row['drug'])}-"
                    f"{sanitize_key(f'{float(row['dosage']):g}')}",
        axis=1
    )
    
    reorganize_by_combination_parallel(
        adata, 
        args.output_dir, 
        total_parts=args.total_parts, 
        num_processes=args.processes
    )