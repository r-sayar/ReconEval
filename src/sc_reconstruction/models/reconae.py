from __future__ import annotations

import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import warnings

'''
Autoencoder model for data reconstruction:
- Autoencoder class: Main model class inheriting from LightningModule.
- ReconAE class: Wrapper for reconstruction tasks inheriting from BaseReconstructionModel.
'''



from scvi.distributions import (
            NegativeBinomial
        )
        
class BaseEncoder(nn.Module):
    def __init__(self, input_dim: int, n_hidden: list, n_latent: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in n_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_latent))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
        
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

class Autoencoder(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        n_hidden: list,
        n_latent: int,
        distribution: str = 'normal',
        learning_rate: float = 0.001,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 5,
        lr_threshold: float = 1e-3,
        lr_min: float = 0.0,
        library_size_mode: str = "none", # "none", "observed", or "modeled"
        decoder_output_activation: str | nn.Module | None = None,
        **trainer_kwargs
    ):
        super().__init__()

        if library_size_mode not in ["none", "observed", "modeled"]:
            raise ValueError("library_size_mode must be 'none', 'observed', or 'modeled'")
        
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.distribution = distribution
        self.learning_rate = learning_rate
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_threshold = lr_threshold
        self.lr_min = lr_min
        self.lr_patience = lr_patience
        self.library_size_mode = library_size_mode


        if not isinstance(decoder_output_activation, nn.Module):
            self.output_activation = self._get_activation_fn(decoder_output_activation)
            if decoder_output_activation is None and 'nb' in self.distribution:
                warnings.warn("For 'nb' distribution with no specified decoder output activation, using 'softplus' activation.")
                self.output_activation = self._get_activation_fn('softplus')
        else:
            self.output_activation = decoder_output_activation
        

        self.encoder = BaseEncoder(self.input_dim, self.n_hidden, self.n_latent)
        
        if self.library_size_mode == "modeled":
            self.l_encoder = BaseEncoder(self.input_dim, [self.n_hidden[0]], 1)

        if library_size_mode == "none":
            self.forward_fn = self._forward_none
        elif library_size_mode == "observed":
            self.forward_fn = self._forward_observed
        else:  # modeled
            self.forward_fn = self._forward_modeled

        self.decoder = BaseDecoder(self.n_latent, 
                                   self.n_hidden[::-1], 
                                   self.input_dim, 
                                   self.output_activation
                                   )
        
        if self.distribution in ['normal_mle_gene', 'nb_gene']:
            initial = torch.zeros(self.input_dim).normal_(mean=0.0, std=0.1)
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
        elif name == 'softplus':
            return nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation: {name}")
        
    def forward(self, x):
        return self.forward_fn(x)

    def _forward_none(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def _forward_observed(self, x):
        z = self.encoder(x)
        library_size = torch.sum(x, dim=1).unsqueeze(1) 
        return self.decoder(z, library_size)

    def _forward_modeled(self, x):
        z = self.encoder(x)
        library_size = torch.exp(self.l_encoder(x))
        return self.decoder(z, library_size)

    def encode(self, x):
        return self.encoder(x)

    def _shared_step(self, batch, stage='train'):
        x = batch['X']
        reconstruction = self(x,)
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
        elif self.distribution == 'nb_gene':
            mu = reconstruction
            theta = torch.exp(torch.clamp(self.px_r, min=-10.0, max=10.0))
            theta_batch = theta.unsqueeze(0)
            nb = NegativeBinomial(mu=mu, theta=theta_batch)
            log_prob = nb.log_prob(x)
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


        

from sc_reconstruction.models._base_model import BaseReconstructionModel
from anndata import AnnData
import os

class ReconAE(BaseReconstructionModel):
    def __init__(
        self,
        input_dim: int,
        n_hidden: list,
        n_latent: int,
        distribution: str = 'normal', # 'normal' for mse, 'nb_gene' for negative binomial with gene-wise dispersion
        library_size_mode: str = "none", # "none", "observed", or "modeled"
        learning_rate: float = 0.001,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 5,
        lr_threshold: float = 1e-3,
        lr_min: float = 0.0,
        decoder_output_activation: str | nn.Module | None = None,
    ):
        self.model_params = {
            'input_dim': input_dim,
            'n_hidden': n_hidden,
            'n_latent': n_latent,
            'distribution': distribution,
            'library_size_mode': library_size_mode,
            'learning_rate': learning_rate,
            'reduce_lr_on_plateau': reduce_lr_on_plateau,
            'lr_factor': lr_factor,
            'lr_patience': lr_patience,
            'lr_threshold': lr_threshold,
            'lr_min': lr_min,
            'decoder_output_activation': decoder_output_activation,
        }
        self.module = Autoencoder(**self.model_params)

    def prepare(self, adata: AnnData | None = None, **kwargs):
        if adata is not None:
            self.adata = adata

    def train(self, datamodule: L.LightningDataModule = None, **train_kwargs):
        trainer = L.Trainer(
            **train_kwargs
        )
        trainer.fit(
            self.module, 
            datamodule=datamodule
        )

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:

        device = next(self.module.parameters()).device
        self.module.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).to(device)
            z = self.module.encode(x_tensor)
        return z.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray: # not optimal
        device = next(self.module.parameters()).device
        self.module.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).to(device)
            reconstruction = self.module(x_tensor)
        return reconstruction.cpu().numpy()

    def predict_relu(self, X: np.ndarray) -> np.ndarray: # not optimal
        device = next(self.module.parameters()).device
        self.module.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).to(device)
            # if self.model_params['library_size_mode'] == "observed":
            #     library_size = torch.sum(x_tensor, dim=1)
            # elif self.model_params['library_size_mode'] == "modeled":
            #     library_size = self.module.l_encoder(x_tensor)
            # else:
            #     library_size = None
            # reconstruction = self.module(x_tensor, library_size)

            reconstruction = self.module(x_tensor)
            reconstruction = torch.relu(reconstruction)

        return reconstruction.cpu().numpy()

    def save(self, path: str):
        if not path.endswith('.pt') and not path.endswith('.ckpt'):
            path = path + '.pt'
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.module, path)
        print(f"Model checkpoint saved to {path}")

    def load(self, path: str, map_location=None) -> None:
        """Load the model"""
        self.module = Autoencoder.load_from_checkpoint(path, map_location=map_location)
        print(f"Model loaded from {path}")

        

