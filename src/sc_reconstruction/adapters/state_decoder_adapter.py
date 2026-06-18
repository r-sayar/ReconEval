"""Frozen MLP decoder adapter for State Transition evaluation.

Wraps a pre-trained MLPDecoder checkpoint so that it can be called as a
standard nn.Module in the inference / evaluation scripts.  All parameters
are frozen (requires_grad=False) and the module is kept in eval mode.
"""

import torch
import torch.nn as nn


class FrozenMLPDecoderAdapter(nn.Module):
    """Adapter that loads a frozen MLPDecoder from a Lightning checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.ckpt`` file produced by Lightning's ``ModelCheckpoint``.
    map_location : str or torch.device, optional
        Device to map weights onto (default: cpu).
    """

    def __init__(self, checkpoint_path: str, map_location="cpu"):
        super().__init__()
        from sc_reconstruction.decoders.reconmlp import MLPDecoder

        # weights_only=False: PyTorch 2.6 changed the default to True, which blocks
        # omegaconf types stored in Lightning checkpoint hyperparameters.
        # These are our own trusted checkpoints.
        self.decoder = MLPDecoder.load_from_checkpoint(
            checkpoint_path, map_location=map_location, weights_only=False
        )
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()

        self._n_input = self.decoder.n_input
        self._n_output = self.decoder.n_output

    # keep eval mode even if .train() is called on a parent module
    def train(self, mode=True):
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to gene expression.

        Accepts arbitrary leading batch dimensions, e.g. ``(B, S, D)``
        from the ST model's set-level output.
        """
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = self.decoder(flat)
        return out.reshape(*shape[:-1], -1)

    @property
    def n_input(self) -> int:
        return self._n_input

    @property
    def n_output(self) -> int:
        return self._n_output

    def gene_dim(self) -> int:
        return self._n_output
