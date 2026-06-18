import zarr
import numpy as np
import os
import argparse

def merge_parts(output_dir, total_parts=10):
    # Create final output directory
    final_output_dir = output_dir+'.zarr'
    os.makedirs(final_output_dir, exist_ok=True)
    final_zarr = zarr.open(final_output_dir, mode='w')
    
    all_combinations = set()
    var_names = None
    
    for i in range(total_parts):
        part_dir = f"{output_dir}_part_{i}.zarr"
        if not os.path.exists(part_dir):
            continue
            
        part_zarr = zarr.open(part_dir, mode='r')
        all_combinations.update(part_zarr.attrs.get('combinations', []))
        
        if var_names is None:
            var_names = part_zarr.attrs.get('var_names', [])
    
    # Set final attributes
    final_zarr.attrs['combinations'] = sorted(all_combinations)
    final_zarr.attrs['var_names'] = var_names
    
 
    for combination in sorted(all_combinations):
        print(f"Merging {combination}")
        
        # Initialize group in final zarr
        if combination not in final_zarr:
            for i in range(total_parts):
                part_dir = f"{output_dir}_part_{i}.zarr"
                if not os.path.exists(part_dir):
                    continue
                    
                part_zarr = zarr.open(part_dir, mode='r')
                if combination in part_zarr:
                    n_vars = part_zarr[combination].attrs['n_vars']
                    dtype = part_zarr[combination]['X'].dtype
                    
                    group = final_zarr.create_group(combination)
                    group.create_dataset(
                        'X',
                        shape=(0, n_vars),
                        chunks=(100000, n_vars),
                        dtype=dtype,
                        compressor=zarr.Blosc(cname='zstd', clevel=3)
                    )
                    group.create_dataset(
                        'obs_index',
                        shape=(0,),
                        dtype='<U40',
                        chunks=(100000,)
                    )
                    group.create_dataset(
                        'plate',
                        shape=(0,),
                        dtype='<U40',
                        chunks=(100000,)
                    )
                    group.attrs['current_row'] = 0
                    group.attrs['n_vars'] = n_vars
                    break
        

        for i in range(total_parts):
            part_dir = f"{output_dir}_part_{i}.zarr"
            if not os.path.exists(part_dir):
                continue
                
            part_zarr = zarr.open(part_dir, mode='r')
            if combination not in part_zarr:
                continue
                
            part_group = part_zarr[combination]
            final_group = final_zarr[combination]
            
            current_row = final_group.attrs['current_row']
            new_rows = part_group['X'].shape[0]
            new_size = current_row + new_rows
            
            final_group['X'].resize((new_size, final_group.attrs['n_vars']))
            final_group['obs_index'].resize(new_size)
            final_group['plate'].resize(new_size)
            
            final_group['X'][current_row:new_size] = part_group['X'][:]
            final_group['obs_index'][current_row:new_size] = part_group['obs_index'][:]
            final_group['plate'][current_row:new_size] = part_group['plate'][:]
            
            final_group.attrs['current_row'] = new_size

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Merge part results')
    parser.add_argument('--output-dir', required=True, help='Base directory containing part results')
    parser.add_argument('--total-parts', type=int, default=10, help='Total number of parts to merge')
    args = parser.parse_args()
    
    merge_parts(args.output_dir, args.total_parts)