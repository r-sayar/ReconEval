import os
import numpy as np
import dask.array as da
import zarr
import gc
import time
from dask.diagnostics import ProgressBar
from sklearn.model_selection import train_test_split

CHUNK_SIZE = 50_000

import dask
dask.config.set({'num_workers': os.cpu_count()})

def split_and_save_drugs(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split drug zarr files into train/val/test and save them
    
    Args:
        source_dir: Directory containing drug zarr files
        output_dir: Directory to save output files
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        seed: Random seed for reproducibility
    """
    print(f"Processing drugs from {source_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # List all drug zarr files
    combinations = [d for d in os.listdir(source_dir) if not d.startswith('.')]
    drugs = list(set(comb.split('-')[1] for comb in combinations))
    print(f"Found {len(drugs)} unique drugs")

    # Split drugs into train/val/test sets
    train_val_ratio = train_ratio + val_ratio
    train_val_drugs, test_drugs = train_test_split(
        drugs, 
        train_size=train_val_ratio,
        random_state=seed
    )
    
    train_drugs, val_drugs = train_test_split(
        train_val_drugs,
        train_size=train_ratio/train_val_ratio,
        random_state=seed
    )
    
    train_combs = [comb for comb in combinations if comb.split('-')[1] in train_drugs]
    val_combs = [comb for comb in combinations if comb.split('-')[1] in val_drugs]
    test_combs = [comb for comb in combinations if comb.split('-')[1] in test_drugs]
    
    print(f"Split: {len(train_drugs)} train drugs -> {len(train_combs)} combinations")
    print(f"Split: {len(val_drugs)} val drugs -> {len(val_combs)} combinations")
    print(f"Split: {len(test_drugs)} test drugs -> {len(test_combs)} combinations")
    
    # Save metadata
    meta_save_path = os.path.join(output_dir, "split_metadata.json")
    meta_group = zarr.open_group(meta_save_path, mode='w')
    meta_group.attrs['train_drugs'] = train_drugs
    meta_group.attrs['val_drugs'] = val_drugs
    meta_group.attrs['test_drugs'] = test_drugs
    meta_group.attrs['train_combinations'] = train_combs
    meta_group.attrs['val_combinations'] = val_combs
    meta_group.attrs['test_combinations'] = test_combs
    meta_group.attrs['random_seed'] = seed
    
    # Load and concatenate validation data
    print("Processing validation data...")
    val_arrays = []
    for comb in val_combs:
        comb_path = os.path.join(source_dir, comb)
        val_arrays.append(da.from_zarr(comb_path, 'X'))
    
    val_data = da.concatenate(val_arrays, axis=0)
    print(f"Validation data shape: {val_data.shape}")
    
    # Save validation data
    val_save_path = os.path.join(output_dir, 'val.zarr')
    if os.path.exists(val_save_path):
        print(f"Validation data already exists at {val_save_path}, skipping save.")
    else:
        print(f"Saving validation data to {val_save_path}")
        with ProgressBar():
            val_data.to_zarr(val_save_path, overwrite=True)
        print(f"Done saving validation data")
    
    # Clear memory
    del val_arrays, val_data
    gc.collect()
    time.sleep(10)  # Give time for memory to be released
    
    # Load and concatenate test data
    print("Processing test data...")
    test_arrays = []
    for comb in test_combs:
        comb_path = os.path.join(source_dir, comb)
        test_arrays.append(da.from_zarr(comb_path, 'X'))
    
    test_data = da.concatenate(test_arrays, axis=0)
    print(f"Test data shape: {test_data.shape}")
    
    # Save test data
    test_save_path = os.path.join(output_dir, 'test.zarr')
    if os.path.exists(test_save_path):
        print(f"Test data already exists at {test_save_path}, skipping save.")
    else:
        print(f"Saving test data to {test_save_path}")
        with ProgressBar():
            test_data.to_zarr(test_save_path, overwrite=True)
        print(f"Done saving test data")
    
    # Clear memory
    del test_arrays, test_data
    gc.collect()
    time.sleep(10)  # Give time for memory to be released
    
    # Load and concatenate training data (unshuffled)
    print("Processing training data...")
    train_arrays = []
    for comb in train_combs:
        comb_path = os.path.join(source_dir, comb)
        train_arrays.append(da.from_zarr(comb_path, 'X'))
    
    train_data = da.concatenate(train_arrays, axis=0)
    print(f"Training data shape: {train_data.shape}")
    
    # Rechunk for better performance
    n_vars = train_data.shape[1]
    train_data = train_data.rechunk((CHUNK_SIZE, n_vars))
    print(f"Training data rechunked: {train_data.numblocks} blocks")
    
    # Save unshuffled training data
    train_unsfd_save_path = os.path.join(output_dir, 'train_unsfd.zarr')
    if os.path.exists(train_unsfd_save_path):
        print(f"Unshuffled training data already exists at {train_unsfd_save_path}, skipping save.")
    else:
        print(f"Saving unshuffled training data to {train_unsfd_save_path}")
        with ProgressBar():
            train_data.to_zarr(train_unsfd_save_path, overwrite=True)
        print(f"Done saving unshuffled training data")
    
    # Shuffle and save training data
    print("Shuffling training data...")
    # Generate a permutation of indices
    rng = np.random.default_rng(seed=seed)
    n = train_data.shape[0]
    perm = rng.permutation(n)
    
    # Create indexers for chunked permutation
    indexer = [list(perm[i:i+CHUNK_SIZE]) for i in range(0, n, CHUNK_SIZE)]
    train_data_shuffled = da.shuffle(train_data, indexer=indexer, axis=0, chunks='auto')
    print(f"Training data shuffled: {train_data_shuffled.numblocks} blocks")
    
    # Save in segments to avoid memory issues
    total_rows = train_data_shuffled.shape[0]
    segment_count = 10
    segment_size = total_rows // segment_count
    
    temp_segments = []
    for i in range(segment_count):
        start = i * segment_size
        end = total_rows if i == segment_count - 1 else (i + 1) * segment_size
        temp_path = os.path.join(output_dir, f'train_temp_{i}.zarr')
        temp_segments.append(temp_path)

        if os.path.exists(temp_path):
            print(f"Segment {i+1}/{segment_count} already exists at {temp_path}, skipping save.")
            continue
        else:
            print(f"Segment {i+1}/{segment_count} does not exist, saving to {temp_path}")
            print(f"Saving segment {i+1}/{segment_count} (rows {start} to {end}) to {temp_path}")
            with ProgressBar():
                train_data_shuffled[start:end].to_zarr(temp_path, overwrite=True)
        
        print(f"Segment {i+1}/{segment_count} saved")
        gc.collect()
        time.sleep(10)  # Give time for memory to be released
    
    # Combine all segments into the final train.zarr file
    print("Combining all segments into train.zarr")
    arrays = [da.from_zarr(segment) for segment in temp_segments]
    combined = da.concatenate(arrays, axis=0)
    
    train_save_path = os.path.join(output_dir, 'train.zarr')
    with ProgressBar():
        combined.to_zarr(train_save_path, overwrite=True)
    
    # Clean up temporary files
    print("Cleaning up temporary files")
    for segment in temp_segments:
        try:
            import shutil
            shutil.rmtree(segment)
            print(f"Removed {segment}")
        except Exception as e:
            print(f"Error removing {segment}: {e}")
    
    print(f"Process complete. Files saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split drug data into train/val/test sets')
    parser.add_argument('--source-dir', required=True, help='Path to source directory with drug zarr files')
    parser.add_argument('--output-dir', required=True, help='Directory to save output files')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Ratio of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Ratio of data for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Ensure ratios sum to 1
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        args.train_ratio /= total
        args.val_ratio /= total
        args.test_ratio /= total
        print(f"Adjusted ratios to sum to 1: {args.train_ratio:.3f}, {args.val_ratio:.3f}, {args.test_ratio:.3f}")
    
    split_and_save_drugs(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    ) 