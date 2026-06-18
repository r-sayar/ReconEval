"""Frozen decoder adapters for end-to-end reconstruction models (PCA, AE, nlscVI).

Each adapter wraps the model's own decoder so it can replace
State Transition's internal gene_decoder during training / inference.
All parameters are frozen and eval mode is enforced.
"""

import numpy as np
import torch
import torch.nn as nn
import zarr


class FrozenPCADecoderAdapter(nn.Module):
    """Linear PCA decoder: z @ pc.T + mean.

    Parameters
    ----------
    mean_path : str
        Path to the zarr file containing the PCA mean vector (n_genes,).
    pc_path : str
        Path to the zarr file containing the PCA components (n_genes, n_components).
    """

    def __init__(self, mean_path: str, pc_path: str, map_location="cpu"):
        super().__init__()
        mean = zarr.load(mean_path).astype(np.float32)
        pc = zarr.load(pc_path).astype(np.float32)
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("pc", torch.from_numpy(pc))
        self._n_input = pc.shape[1]
        self._n_output = pc.shape[0]

    def train(self, mode=True):
        return super().train(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape
        flat = z.reshape(-1, shape[-1])
        out = flat @ self.pc.T + self.mean
        return out.reshape(*shape[:-1], -1)

    @property
    def n_input(self) -> int:
        return self._n_input

    @property
    def n_output(self) -> int:
        return self._n_output

    def gene_dim(self) -> int:
        return self._n_output


class FrozenAEDecoderAdapter(nn.Module):
    """Wraps the BaseDecoder from a trained Autoencoder checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Lightning ``.ckpt`` file for the ``Autoencoder`` model.
    """

    def __init__(self, checkpoint_path: str, map_location="cpu"):
        super().__init__()
        from sc_reconstruction.models.reconae import Autoencoder

        # weights_only=False: PyTorch 2.6 changed the default to True, which blocks
        # omegaconf types (ListConfig, DictConfig, ContainerMetadata, ...) stored in
        # Lightning checkpoint hyperparameters. These are our own trusted checkpoints.
        model = Autoencoder.load_from_checkpoint(
            checkpoint_path, map_location=map_location, weights_only=False
        )
        self.ae_decoder = model.decoder
        self._n_input = model.n_latent
        self._n_output = model.input_dim
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape
        flat = z.reshape(-1, shape[-1])
        out = self.ae_decoder(flat)
        return out.reshape(*shape[:-1], -1)

    @property
    def n_input(self) -> int:
        return self._n_input

    @property
    def n_output(self) -> int:
        return self._n_output

    def gene_dim(self) -> int:
        return self._n_output


class FrozenNLscVIDecoderAdapter(nn.Module):
    """Wraps the NormalDecoderSCVI from a trained nlscVI (NormalVAE) module.

    NormalDecoderSCVI uses Identity activation (no softmax), so
    ``px_rate = px_scale`` -- the direct network output with no
    library-size dependency.

    Parameters
    ----------
    module_path : str
        ``.pt`` file saved via ``torch.save(module, path)``.
    """

    def __init__(self, module_path: str, map_location="cpu"):
        super().__init__()
        module = torch.load(module_path, map_location=map_location, weights_only=False)
        self.nlscvi_decoder = module.decoder
        self._dispersion = module.dispersion
        self._n_input = module.n_latent
        self._n_output = module.px_r.shape[0]  # px_r: (n_genes,) for "gene" dispersion
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape
        flat = z.reshape(-1, shape[-1])
        n = flat.shape[0]
        batch_idx = torch.zeros(n, 1, dtype=torch.long, device=flat.device)
        px_scale, px_r, px_rate, px_dropout = self.nlscvi_decoder(
            self._dispersion, flat, None, batch_idx
        )
        return px_rate.reshape(*shape[:-1], -1)

    @property
    def n_input(self) -> int:
        return self._n_input

    @property
    def n_output(self) -> int:
        return self._n_output

    def gene_dim(self) -> int:
        return self._n_output
