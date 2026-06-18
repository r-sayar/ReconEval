from __future__ import annotations

import lightning as L
import torch.nn as nn
import torch
import numpy as np
import warnings
from collections.abc import Iterable
import os

class BaseDecoder(nn.Module):
    def __init__(self, 
                 n_latent: int, 
                 n_hidden: list, 
                 output_dim: int,
                 output_activation: nn.Module):
        super().__init__()
        layers = []
        prev_dim = n_latent
        for hidden_dim in n_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
        self.output_activation = output_activation
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, library_size=None):
        reconstruction = self.decoder(x)
        if library_size is not None:
            reconstruction = self.softmax(reconstruction) * library_size
        else:
            reconstruction = self.output_activation(reconstruction)
        return reconstruction

class MLPDecoder(L.LightningModule):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: list,
        n_cat_list: Iterable[int] = None,
        distribution: str = 'normal',
        learning_rate: float = 0.001,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 5,
        lr_threshold: float = 1e-3,
        lr_min: float = 0.0,
        library_size_mode: str = 'none',
        decoder_output_activation: str = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_cat_list = n_cat_list
        self.distribution = distribution
        self.learning_rate = learning_rate
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_threshold = lr_threshold
        self.lr_min = lr_min
        self.library_size_mode = library_size_mode
        self.decoder_output_activation = decoder_output_activation

        print(f"Initializing MLPDecoder with n_input={n_input}, n_output={n_output}, n_hidden={n_hidden}")

        total_input_dim = n_input
        if n_cat_list is not None:
            total_input_dim += sum(n_cat_list)
        
        activation_fn = self._get_activation_fn(decoder_output_activation)
        self.decoder = BaseDecoder(n_latent = total_input_dim, 
                                   n_hidden = n_hidden, 
                                   output_dim = n_output, 
                                   output_activation = activation_fn
                                   )

        # For normal_mle_gene distribution
        if self.distribution == 'normal_mle_gene':
            initial = torch.zeros(n_output).normal_(mean=0.0, std=0.1)
            self.px_r = nn.Parameter(initial)


    def _get_activation_fn(self, name: str | None) -> nn.Module:
        """Map string name to activation function module"""
        if name is None:
            return nn.Identity()
        name = name.lower()
        if name == 'linear' or name == 'identity':
            return nn.Identity()
        elif name == 'softmax':
            return nn.Softmax(dim=-1)
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'softplus':
            return nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, z: torch.Tensor, *cat_list: torch.Tensor) -> torch.Tensor:
        """Forward pass - supports both unconditional and conditional decoding"""
        if cat_list and len(cat_list) > 0:
            z_cat = torch.cat([z] + list(cat_list), dim=1)
        else:
            z_cat = z

        reconstruction = self.decoder(z_cat)
        
        if self.library_size_mode != 'none':
            warnings.warn("library_size_mode is set but not implemented for decoding only. Ignoring library size mode.")
            
        return reconstruction

    def _shared_step(self, batch, stage='train'):
        x = batch['x']
        z = batch['z']
        
        # Handle conditional vs unconditional
        if 'batch_onehot' in batch and batch['batch_onehot'] is not None:
            batch_onehot = batch['batch_onehot']
            reconstruction = self.forward(z, batch_onehot)
        else:
            reconstruction = self.forward(z)

        loss = self.compute_loss(x, reconstruction)
        self.log_metrics({'loss': loss}, stage)
        return loss

    def training_step(self, batch, batch_idx):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True)
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    def compute_loss(self, x, reconstruction):
        """Reconstruction loss matching ``Autoencoder.compute_loss`` in reconae.py."""
        if self.distribution == 'normal':
            return nn.MSELoss()(reconstruction, x)
        elif self.distribution == 'huber':
            return nn.HuberLoss()(reconstruction, x)
        elif self.distribution == 'l1':
            return nn.L1Loss()(reconstruction, x)
        elif self.distribution == 'normal_mle_fixed':
            sigma = torch.tensor(1e-1, device=x.device)  
            dist = torch.distributions.Normal(loc=reconstruction, scale=sigma)
            log_prob = dist.log_prob(x)      
            neg_log_likelihood = -log_prob.sum(dim=1).mean()
            return neg_log_likelihood
        elif self.distribution == 'normal_mle_gene':
            sigma = torch.exp(torch.clamp(self.px_r, min=-10.0, max=10.0))
            sigma_batch = sigma.unsqueeze(0)    
            dist = torch.distributions.Normal(loc=reconstruction, scale=sigma_batch)
            log_prob = dist.log_prob(x)       
            neg_log_likelihood = -log_prob.sum(dim=1).mean()
            return neg_log_likelihood
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.reduce_lr_on_plateau:
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.lr_patience,
                    factor=self.lr_factor,
                    threshold=self.lr_threshold,
                    min_lr=self.lr_min,
                    threshold_mode="abs"
                ),
                'monitor': 'val/loss_epoch',
                'interval': 'epoch',   
                'frequency': 1,        
                'name': 'lr_scheduler' 
            }
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        else:
            return optimizer

    def log_metrics(self, loss_dict, stage='train'):
        for key, value in loss_dict.items():
            self.log(f'{stage}/{key}', value, on_step=True, on_epoch=True, prog_bar=True)


from sc_reconstruction.decoders._base_decoder import BaseReconstructionDecoder

class ReconMLPDecoder(BaseReconstructionDecoder):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        distribution: str = 'normal',
        learning_rate: float = 0.001,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 5,
        lr_threshold: float = 1e-3,
        lr_min: float = 0.0,
        library_size_mode: str = 'none',
        decoder_output_activation: str = None,
        **kwargs
    ):
        # Build hidden layers configuration
        hidden_dims = [n_hidden] * n_layers
        
        self.model_params = {
            "n_input": n_input,
            "n_output": n_output,
            "n_hidden": hidden_dims,
            "n_cat_list": n_cat_list,
            "distribution": distribution,
            "learning_rate": learning_rate,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "lr_threshold": lr_threshold,
            "lr_min": lr_min,
            "library_size_mode": library_size_mode,
            "decoder_output_activation": decoder_output_activation,
        }
        
        print(f"ReconMLPDecoder params: n_input={n_input}, n_output={n_output}, n_hidden={hidden_dims}")
        self.module = MLPDecoder(**self.model_params)

    def train(self, datamodule: L.LightningDataModule = None, **train_kwargs) -> None:
        trainer = L.Trainer(**train_kwargs)
        trainer.fit(self.module, datamodule=datamodule)

    def decode(self, z: np.ndarray, *cat_list: np.ndarray) -> np.ndarray:
        """Decode latent representations to data space"""
        device = next(self.module.parameters()).device
        self.module.eval()
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().to(device)
            if cat_list and len(cat_list) > 0:
                cat_tensors = [torch.from_numpy(cat).float().to(device) for cat in cat_list]
                reconstruction = self.module(z_tensor, *cat_tensors)
            else:
                reconstruction = self.module(z_tensor)  
        return reconstruction.cpu().numpy()
    
    # def predict(self, z: np.ndarray, *cat_list: np.ndarray) -> np.ndarray:
    #     """Alias for decode"""
    #     return self.decode(z, *cat_list)
    
    def save(self, path: str):
        if not path.endswith('.pt') and not path.endswith('.ckpt'):
            path = path + '.pt'

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.module.state_dict(), path)
        print(f"Model weights saved to {path}")
    
    def load(self, path: str, map_location=None) -> None:
        """Load the model weights"""
        if path.endswith('.pt'):
            self.module = torch.load(path, map_location=map_location, weights_only=False)
        elif path.endswith('.ckpt'):
            self.module = MLPDecoder.load_from_checkpoint(path, map_location=map_location)
        print(f"MLP decoder loaded from {path}")