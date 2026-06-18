"""Encoder adapters for extracting pretrained encoder weights.

Used to initialize JiTFlow's LearnableAutoEncoder with pretrained weights
from AE/nlscVI/PCA checkpoints. These extract PyTorch weights as numpy arrays
for transfer to JAX/Flax models.
"""

import numpy as np
import torch


class AEEncoderAdapter:
    """Extract encoder from a pretrained Autoencoder Lightning checkpoint.

    The pretrained encoder is: Linear → BatchNorm → ReLU → Dropout → ... → Linear
    JiTFlow's LearnableEncoder is: Dense → activation → ... → Dense

    We extract the Linear layer weights/biases for initialization.
    BatchNorm statistics are NOT transferred (JiTFlow doesn't use BN).

    Parameters
    ----------
    checkpoint_path
        Path to the Lightning .ckpt file.
    map_location
        Device for loading (default: "cpu").
    """

    def __init__(self, checkpoint_path: str, map_location: str = "cpu"):
        import torch
        from sc_reconstruction.models.reconae import Autoencoder
        try:
            torch.serialization.add_safe_globals([__import__("omegaconf").listconfig.ListConfig])
            model = Autoencoder.load_from_checkpoint(checkpoint_path, map_location=map_location)
        except Exception:
            # Fallback for older PyTorch or different pickle issues
            model = Autoencoder.load_from_checkpoint(
                checkpoint_path, map_location=map_location, weights_only=False)
        model.eval()
        self._model = model
        self._encoder = model.encoder
        self._decoder = model.decoder

    @property
    def n_input(self):
        return self._model.input_dim

    @property
    def n_latent(self):
        return self._model.n_latent

    @property
    def n_output(self):
        return self._model.input_dim

    def get_encoder_weights(self):
        """Extract encoder Linear layer weights as list of (weight, bias) numpy arrays.

        Returns list of tuples: [(W1, b1), (W2, b2), ...] where each Wi has
        shape (out_features, in_features) in PyTorch convention.
        For JAX Dense layers, transpose to (in_features, out_features).
        """
        layers = []
        for module in self._encoder.encoder:
            if isinstance(module, torch.nn.Linear):
                w = module.weight.detach().cpu().numpy()  # (out, in)
                b = module.bias.detach().cpu().numpy()    # (out,)
                layers.append((w.T, b))  # Transpose for JAX: (in, out)
        return layers

    def get_decoder_weights(self):
        """Extract decoder Linear layer weights as list of (weight, bias) numpy arrays."""
        layers = []
        for module in self._decoder.decoder:
            if isinstance(module, torch.nn.Linear):
                w = module.weight.detach().cpu().numpy()
                b = module.bias.detach().cpu().numpy()
                layers.append((w.T, b))  # Transpose for JAX
        return layers

    def encode(self, x):
        """Run encoder forward pass (PyTorch). For validation only."""
        with torch.no_grad():
            return self._encoder(torch.as_tensor(x, dtype=torch.float32)).numpy()

    def decode(self, z):
        """Run decoder forward pass (PyTorch). For validation only."""
        with torch.no_grad():
            return self._decoder(torch.as_tensor(z, dtype=torch.float32)).numpy()


class PCAEncoderAdapter:
    """PCA encoder/decoder using mean and principal components.

    Parameters
    ----------
    mean_path
        Path to mean.zarr.
    pc_path
        Path to pc_{dim}.zarr.
    """

    def __init__(self, mean_path: str, pc_path: str):
        import zarr
        self._mean = zarr.load(mean_path).astype(np.float32)
        self._pc = zarr.load(pc_path).astype(np.float32)

    @property
    def n_input(self):
        return self._mean.shape[0]

    @property
    def n_latent(self):
        return self._pc.shape[1]

    def encode(self, x):
        """Encode: z = (x - mean) @ pc"""
        return (np.asarray(x, dtype=np.float32) - self._mean) @ self._pc

    def decode(self, z):
        """Decode: x = z @ pc.T + mean"""
        return np.asarray(z, dtype=np.float32) @ self._pc.T + self._mean
