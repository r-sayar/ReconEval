from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
import dask.array as da
import os
import warnings
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sc_reconstruction.dataloaders.datasets import *
import zarr
from typing import Optional, Dict, Any


def transform_to_csr(path, chunk_size=1_000_000):
    train_data = da.from_zarr(path)
    train_data = train_data.rechunk((chunk_size, -1))
    blocks = []
    n_blocks = train_data.numblocks[0]  

    for i in range(n_blocks):
        block = train_data.blocks[i, :].compute()
        csr_block = csr_matrix(block)
        blocks.append(csr_block)

    global_csr = vstack(blocks)
    adata = ad.AnnData(X=global_csr)
    adata.write_h5ad(path.replace('.zarr', '.h5ad'))
    return path.replace('.zarr', '.h5ad')

class IterDaskDataModule(LightningDataModule):
    '''
    Iterative datamodule to align with the iterative dataloader, can be used for scVI standard training
    '''
    def __init__(
            self,
            train_path, 
            val_path = None, 
            test_path = None, 
            chunk_size = 100_000, 
            minibatch_size=4096,
            num_workers=10,
            max_workers_percentage=0.8,
            chunks_per_worker=5,
            random_seed=42):
        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.chunk_size = chunk_size
        self.minibatch_size = minibatch_size
        self.max_workers_percentage = max_workers_percentage
        self.num_workers = num_workers
        self.n_vars = None  
        self.chunks_per_worker = chunks_per_worker
        # no batch effect
        self.n_batch = 1
        self.random_seed = random_seed
        if not os.path.isdir(self.train_path):
            warnings.warn(f"Train directory not found: {self.train_path}")
        if not os.path.isdir(self.val_path):
            warnings.warn(f"Validation directory not found: {self.val_path}")
        if not os.path.isdir(self.test_path):
            warnings.warn(f"Test directory not found: {self.test_path}")
        
        self.X = self._load_zarr_data(self.train_path, description="Prep data")
        self.n_vars = self.X.shape[1]

    @staticmethod
    def _load_zarr_data(path, description="Non-specified data"):

        if not os.path.isdir(path):
            warnings.warn(f"{description} doesn't exist: {path}")
            return None
        
        try:
            data = da.from_zarr(path)
            print(f"{description} loading directly from: {path}")
            return data
        except Exception as e:
            try:
                data = da.from_zarr(path, 'X')
                print(f"{description} loading from 'X': {path}")
                return data
            except Exception as e2:
                warnings.warn(f"Failing loading {description}: {path}. Error: {e2}")
                return None

    def prepare_data(self):
        '''
        Load data from zarr files using Dask
        '''
        self.train_data = self._load_zarr_data(self.train_path, description="Train data")
        self.val_data = self._load_zarr_data(self.val_path, description="Val data") if self.val_path is not None else None
        self.test_data = self._load_zarr_data(self.test_path, description="Test data") if self.test_path is not None else None

    def setup(self, stage=None):

        if self.train_data is not None:
            self.train_data = self.train_data.rechunk((self.chunk_size, self.n_vars))
        if self.val_data is not None:
            self.val_data = self.val_data.rechunk((self.chunk_size, self.n_vars))
        if self.test_data is not None:
            self.test_data = self.test_data.rechunk((self.chunk_size, self.n_vars))
        
        print(f"Train data: shape={self.train_data.shape}, chunks={self.train_data.numblocks}")
        print(f"Val data: shape={self.val_data.shape}, chunks={self.val_data.numblocks}")
        print(f"Test data: shape={self.test_data.shape}, chunks={self.test_data.numblocks}")

        if self.train_data is not None:
            self.train_dataset = DaskSCVIIterableDataset(self.train_data, chunk_shuffle=True, batch_size=self.minibatch_size, seed=self.random_seed, chunks_per_worker=self.chunks_per_worker)
        if self.val_data is not None:
            self.val_dataset = DaskSCVIIterableDataset(self.val_data, chunk_shuffle=False, batch_size=self.minibatch_size, seed=self.random_seed, chunks_per_worker=self.chunks_per_worker)
        if self.test_data is not None:
            self.test_dataset = DaskSCVIIterableDataset(self.test_data, chunk_shuffle=False, batch_size=self.minibatch_size, seed=self.random_seed, chunks_per_worker=self.chunks_per_worker)
    
    def shuffle_train_data(self):
        self.train_dataset.shuffle_chunks()
        print("Train dataset chunks shuffled.")

        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None, 
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None, 
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=None, 
            num_workers=self.num_workers,
            pin_memory=True
        )
    
from lightning.pytorch.callbacks import Callback
class DatasetEpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if dm is not None and hasattr(dm, "set_epoch"):
            dm.set_epoch(trainer.current_epoch)
        elif dm is not None and hasattr(dm, "train_dataset") and hasattr(dm.train_dataset, "set_epoch"):
            dm.train_dataset.set_epoch(trainer.current_epoch)

import h5py
import anndata as ad
class DaskPCADataModule():
    '''
    Mini-class for PCA with Dask
    Uses h5ad train/val/test for PCA fitting and evaluation
    val/test does not do anything
    '''
    def __init__(self, train_path, val_path = None, test_path = None, chunk_size=1_000_000):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.chunk_size = chunk_size
        
    def prepare_data(self):
        if not os.path.isfile(self.train_path) and not os.path.isdir(self.train_path):
            raise FileNotFoundError(f"Train file not found: {self.train_path}, at least one of zarr or h5ad for training data is required")

        if not self.train_path.endswith('.h5ad'):
            if not os.path.isfile(self.train_path.replace('.zarr', '.h5ad')):
                print(f"Provided train data is not in .h5ad format, transforming {self.train_path} to .h5ad")
                self.train_path = transform_to_csr(self.train_path, self.chunk_size)
            else:
                self.train_path = self.train_path.replace('.zarr', '.h5ad')

        with h5py.File(self.train_path, "r") as f:
            self.adata = ad.AnnData(
                X=ad.experimental.read_elem_lazy(
                    f["X"], 
                    chunks=(self.chunk_size, -1) 
                )
            )
    def train_dataloader(self):
        return self.adata

        

# Datamodule for decode-only training
class DecodeOnlyDataModule(LightningDataModule):
    def __init__(
            self,
            train_path, 
            val_path = None, 
            test_path = None, 
            expression_key = 'X',
            latent_key = 'scANVI',
            batch_key = None,
            create_pseudo_batch: bool = False,
            minibatch_size = 128,
            num_workers = 10,
            max_workers_percentage = 0.8,
            random_seed = 42):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.expression_key = expression_key
        self.latent_key = latent_key
        self.batch_key = batch_key
        self.create_pseudo_batch = create_pseudo_batch
        self.minibatch_size = minibatch_size
        self.num_workers = num_workers
        self.max_workers_percentage = max_workers_percentage
        self.random_seed = random_seed


    def prepare_data(self):
        pass

    def _create_data_dict(self, zarr_store: zarr.Group) -> Dict[str, Any]:
        data_dict = {
            self.expression_key: zarr_store[self.expression_key][:],
            self.latent_key: zarr_store[self.latent_key][:],
        }
        
        if self.batch_key and self.batch_key in zarr_store:
            data_dict[self.batch_key] = zarr_store[self.batch_key][:]
            
        return data_dict

    def _create_dataset(self, zarr_store: zarr.Group) -> DecodeOnlyDataset:
        """Create dataset from Zarr store"""
        data_dict = self._create_data_dict(zarr_store)
        
        return DecodeOnlyDataset(
            data_dict=data_dict,
            expression_key=self.expression_key,
            latent_key=self.latent_key,
            batch_key=self.batch_key,
            pseudo_batch=self.create_pseudo_batch
        )

    def setup(self, stage=None):
        if os.path.isdir(self.train_path):
            self.train_zarr = zarr.open(self.train_path, mode='r')
        else:
            self.train_zarr = None

        if self.val_path is not None and os.path.isdir(self.val_path):
            self.val_zarr = zarr.open(self.val_path, mode='r')
        else:
            self.val_zarr = None

        if self.test_path is not None and os.path.isdir(self.test_path):
            self.test_zarr = zarr.open(self.test_path, mode='r')
        else:
            self.test_zarr = None

        if stage == "fit" or stage is None:
            if self.train_zarr is not None:
                self.train_dataset = self._create_dataset(self.train_zarr)
                print(f"  - Train samples: {len(self.train_dataset)}")
                
            if self.val_zarr is not None:
                self.val_dataset = self._create_dataset(self.val_zarr)
                print(f"  - Validation samples: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            if self.test_zarr is not None:
                self.test_dataset = self._create_dataset(self.test_zarr)
                print(f"  - Test samples: {len(self.test_dataset)}")
                
        if self.train_dataset and self.train_dataset.batch is not None:
            batch_type = "real" if self.train_dataset.has_real_batch else "pseudo"
            print(f"  - Batch type: {batch_type}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.minibatch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.minibatch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.minibatch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )


class DaskDecodeOnlyDataModule(LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path=None,
        test_path=None,
        expression_key='X',
        latent_key='scANVI',
        batch_key=None,  
        chunk_size=100_000,
        minibatch_size=4096,
        num_workers=10,
        chunks_per_worker=5,
        random_seed=42
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.expression_key = expression_key
        self.latent_key = latent_key
        self.batch_key = batch_key
        self.chunk_size = chunk_size
        self.minibatch_size = minibatch_size
        self.num_workers = num_workers
        self.chunks_per_worker = chunks_per_worker
        self.random_seed = random_seed
        
        # Data attributes
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_latent = None
        self.val_latent = None
        self.test_latent = None
        self.train_batch = None
        self.val_batch = None
        self.test_batch = None
        self.n_vars = None

    def _load_zarr_arrays(self, path, description="Data"):
        """Load expression, latent, and optional batch arrays from Zarr"""
        if not os.path.isdir(path):
            warnings.warn(f"{description} directory not found: {path}")
            return None, None, None
        
        try:
            store = zarr.open(path, mode='r')
            
            if self.expression_key in store:
                X_data = da.from_zarr(path, component=self.expression_key)
            else:
                raise KeyError(f"Expression key '{self.expression_key}' not found")
            
            if self.latent_key in store:
                latent_data = da.from_zarr(path, component=self.latent_key)
            else:
                raise KeyError(f"Latent key '{self.latent_key}' not found")
            
            batch_data = None
            if self.batch_key and self.batch_key in store:
                batch_data = da.from_zarr(path, component=self.batch_key)
            
            print(f"{description} loaded from {path}")
            print(f"  X shape: {X_data.shape}, latent shape: {latent_data.shape}")
            if batch_data is not None:
                print(f"  batch shape: {batch_data.shape}")
            
            return X_data, latent_data, batch_data
            
        except Exception as e:
            warnings.warn(f"Failed loading {description} from {path}: {e}")
            return None, None, None

    def prepare_data(self):
        """Load all data arrays"""
        print("Loading data arrays...")
        print("Train data path:", self.train_path)
        print("Validation data path:", self.val_path)
        print("Test data path:", self.test_path)
        
        self.train_data, self.train_latent, self.train_batch = self._load_zarr_arrays(
            self.train_path, "Train data"
        )
        
        if self.val_path:
            self.val_data, self.val_latent, self.val_batch = self._load_zarr_arrays(
                self.val_path, "Validation data"
            )
        
        if self.test_path:
            self.test_data, self.test_latent, self.test_batch = self._load_zarr_arrays(
                self.test_path, "Test data"
            )

    def setup(self, stage=None):
        """Setup datasets with proper chunking"""
        # In DDP, prepare_data() only runs on local rank 0.
        # Re-load here so every rank has the data references.
        if self.train_data is None:
            self.prepare_data()

        if self.train_data is not None:
            self.n_vars = self.train_data.shape[1]
            
            self.train_data = self.train_data.rechunk((self.chunk_size, self.n_vars))
            self.train_latent = self.train_latent.rechunk((self.chunk_size, self.train_latent.shape[1]))
            if self.train_batch is not None:
                self.train_batch = self.train_batch.rechunk((self.chunk_size, self.train_batch.shape[1]))
            
            self.train_dataset = DaskDecodeOnlyIterableDataset(
                X_data=self.train_data,
                latent_data=self.train_latent,
                batch_data=self.train_batch,
                chunk_shuffle=True,
                batch_size=self.minibatch_size,
                seed=self.random_seed,
                chunks_per_worker=self.chunks_per_worker
            )

        if self.val_data is not None:
            self.val_data = self.val_data.rechunk((self.chunk_size, self.n_vars))
            self.val_latent = self.val_latent.rechunk((self.chunk_size, self.val_latent.shape[1]))
            if self.val_batch is not None:
                self.val_batch = self.val_batch.rechunk((self.chunk_size, self.val_batch.shape[1]))
            
            self.val_dataset = DaskDecodeOnlyIterableDataset(
                X_data=self.val_data,
                latent_data=self.val_latent,
                batch_data=self.val_batch,
                chunk_shuffle=False,  # No shuffle for validation
                batch_size=self.minibatch_size,
                seed=self.random_seed,
                chunks_per_worker=self.chunks_per_worker
            )

        if self.test_data is not None:
            self.test_data = self.test_data.rechunk((self.chunk_size, self.n_vars))
            self.test_latent = self.test_latent.rechunk((self.chunk_size, self.test_latent.shape[1]))
            if self.test_batch is not None:
                self.test_batch = self.test_batch.rechunk((self.chunk_size, self.test_batch.shape[1]))
            
            self.test_dataset = DaskDecodeOnlyIterableDataset(
                X_data=self.test_data,
                latent_data=self.test_latent,
                batch_data=self.test_batch,
                chunk_shuffle=False,  # No shuffle for test
                batch_size=self.minibatch_size,
                seed=self.random_seed,
                chunks_per_worker=self.chunks_per_worker
            )

        # Print dataset info
        print(f"\nDataset Info:")
        if self.train_data is not None:
            print(f"Train: X{self.train_data.shape} -> latent{self.train_latent.shape}, chunks={self.train_data.numblocks}")
        if self.val_data is not None:
            print(f"Val: X{self.val_data.shape} -> latent{self.val_latent.shape}, chunks={self.val_data.numblocks}")
        if self.test_data is not None:
            print(f"Test: X{self.test_data.shape} -> latent{self.test_latent.shape}, chunks={self.test_data.numblocks}")

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling"""
        if hasattr(self, 'train_dataset'):
            self.train_dataset.set_epoch(epoch)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=None,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None

    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=None,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None

class SubsetDaskDecodeOnlyDataModule(DaskDecodeOnlyDataModule):
    def __init__(
        self,
        target_feature_indices: np.ndarray,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.target_feature_indices = target_feature_indices

    def _load_zarr_arrays(self, path, description="Data"):
        """Load expression, latent, and optional batch arrays from Zarr"""
        # Call parent method to load data
        X_data, latent_data, batch_data = super()._load_zarr_arrays(path, description)
        
        # If data was successfully loaded, subset the features
        if X_data is not None:
            print(f"  Original X shape: {X_data.shape}")
            X_data = X_data[:, self.target_feature_indices]
            print(f"  Subset X shape: {X_data.shape}")
        
        return X_data, latent_data, batch_data

    def prepare_data(self):
        """Load all data arrays with feature subsetting"""
        super().prepare_data()
        
        if self.train_data is not None:
            self.n_vars = len(self.target_feature_indices)
            print(f"\nFeature subsetting applied:")
            print(f"  Using {self.n_vars} features from original {self.train_data.shape[1]}")


# H5adReconstructionDataModule lives in ``h5ad_datamodule.py`` — kept out of
# this module so it can be imported without the Dask/Zarr/Lightning/omegaconf
# stack that the Zarr classes above require.