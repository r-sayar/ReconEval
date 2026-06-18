from __future__ import annotations

import numpy as np
import os
import glob


def read_csv_pandas_flexible(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        header = [col for col in first_line.strip().split(',') if col != '']
        expected_columns = len(header)
    
    
    df = pd.read_csv(
        file_path,
        engine='python',
        on_bad_lines=lambda bad_line: bad_line[:expected_columns] +
                                    [''] * max(0, expected_columns - len(bad_line)),
        skipinitialspace=True
    )
    
    df = df.iloc[:, :expected_columns]
    df.columns = header
    
    return df

def find_model_paths(base_dir):
    model_paths = {}
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            ckpt_files = glob.glob(os.path.join(item_path, "**/*.ckpt"), recursive=True)
            
            if ckpt_files:
                model_paths[item] = ckpt_files[0]
            else:
                print(f"No .ckpt file found in {item_path}")
    
    return model_paths

def sample_list(items, fraction: float | None, rng: np.random.Generator):
    items = list(items)
    n = len(items)
    if fraction is None:
        return items
    if n == 0:
        return []
    if fraction <= 0:
        return []
    if fraction >= 1:
        return items
    k = max(1, int(n * fraction))
    idx = rng.choice(n, size=k, replace=False)
    return [items[i] for i in sorted(idx)]