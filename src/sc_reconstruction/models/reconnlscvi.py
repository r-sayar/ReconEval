from __future__ import annotations

import collections
from collections.abc import Callable, Iterable
from typing import Literal, Any, Dict
import torch
import numpy as np
from anndata import AnnData
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import ModuleList

from scvi.nn._utils import ExpActivation
from scvi.nn import FCLayers
import os
from collections.abc import Callable
from typing import Literal

from torch.distributions import Distribution


# Decoder
class NormalDecoderSCVI(nn.Module):
    """
    change the decoder of scvi to normal distribution without softmax
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        else:
            px_scale_activation = nn.Identity()

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )
        
        # dispersion: gene-cell case only
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor | None = None,
        *cat_list: int,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library_size
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = px_scale
        
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout



import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn.functional import one_hot

from scvi import REGISTRY_KEYS, settings
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import (
    BaseMinifiedModeModuleClass,
    EmbeddingModuleMixin,
    LossOutput,
    auto_move_data,
)
from scvi.utils import unsupported_if_adata_minified



logger = logging.getLogger(__name__)


class NormalVAE(EmbeddingModuleMixin, BaseMinifiedModeModuleClass):

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        extra_payload_autotune: bool = False,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        batch_embedding_kwargs: dict | None = None,
    ):
        from scvi.nn import DecoderSCVI, Encoder

        self.__class__.__name__  = "VAE"
        super().__init__()

        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        self.extra_payload_autotune = extra_payload_autotune

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "`dispersion` must be one of 'gene', 'gene-batch', 'gene-label', 'gene-cell'."
            )

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        if self.batch_representation == "embedding":
            n_input_encoder += batch_dim * encode_covariates
            cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        n_input_decoder = n_latent + n_continuous_cov
        if self.batch_representation == "embedding":
            n_input_decoder += batch_dim

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = NormalDecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation=None,
            **_extra_decoder_kwargs,
        )

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        if full_forward_pass or self.minified_data_type is None:
            loader = "full_data"
        elif self.minified_data_type in [
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR_WITH_COUNTS,
        ]:
            loader = "minified_data"
        else:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")

        if loader == "full_data":
            return {
                MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
                MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
                MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
                MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            }
        else:
            return {
                MODULE_KEYS.QZM_KEY: tensors[REGISTRY_KEYS.LATENT_QZM_KEY],
                MODULE_KEYS.QZV_KEY: tensors[REGISTRY_KEYS.LATENT_QZV_KEY],
                REGISTRY_KEYS.OBSERVED_LIB_SIZE: tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE],
            }

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.LIBRARY_KEY: inference_outputs[MODULE_KEYS.LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.Y_KEY: tensors[REGISTRY_KEYS.LABELS_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
        }

    def _compute_local_library_params(
        self,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        from torch.nn.functional import linear

        n_batch = self.library_log_means.shape[1]
        local_library_log_means = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
        )

        local_library_log_vars = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
        )

        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            qz, z = self.z_encoder(encoder_input, *categorical_input)
        else:
            qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(
                    encoder_input, batch_index, *categorical_input
                )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
        }

    @auto_move_data
    def _cached_inference(
        self,
        qzm: torch.Tensor,
        qzv: torch.Tensor,
        observed_lib_size: torch.Tensor,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | None]:
        """Run the cached inference process."""
        from torch.distributions import Normal

        qz = Normal(qzm, qzv.sqrt())
        # use dist.sample() rather than rsample because we aren't optimizing the z here
        untran_z = qz.sample() if n_samples == 1 else qz.sample((n_samples,))
        z = self.z_encoder.z_transformation(untran_z)
        library = torch.log(observed_lib_size)
        if n_samples > 1:
            library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: None,
            MODULE_KEYS.LIBRARY_KEY: library,
        }

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process."""
        from torch.nn.functional import linear

        from scvi.distributions import (
            NegativeBinomial,
            Normal,
            Poisson,
            ZeroInflatedNegativeBinomial,
        )

        # TODO: refactor forward function to not rely on y
        # Likelihood distribution
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_input = torch.cat([decoder_input, batch_rep], dim=-1)
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                *categorical_input,
                y,
            )
        else:
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                batch_index,
                *categorical_input,
                y,
            )

        if self.dispersion == "gene-label":
            px_r = linear(
                one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = linear(one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(rate=px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            px = Normal(px_rate, px_r, normal_mu=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
        }

    @unsupported_if_adata_minified
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: torch.Tensor | float = 1.0,
    ) -> LossOutput:
        """Compute the loss."""
        from torch.distributions import kl_divergence

        x = tensors[REGISTRY_KEYS.X_KEY]
        kl_divergence_z = kl_divergence(
            inference_outputs[MODULE_KEYS.QZ_KEY], generative_outputs[MODULE_KEYS.PZ_KEY]
        ).sum(dim=-1)
        if not self.use_observed_lib_size:
            kl_divergence_l = kl_divergence(
                inference_outputs[MODULE_KEYS.QL_KEY], generative_outputs[MODULE_KEYS.PL_KEY]
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        reconst_loss = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        # a payload to be used during autotune
        if self.extra_payload_autotune:
            extra_metrics_payload = {
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
        else:
            extra_metrics_payload = {}

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local={
                MODULE_KEYS.KL_L_KEY: kl_divergence_l,
                MODULE_KEYS.KL_Z_KEY: kl_divergence_z,
            },
            extra_metrics=extra_metrics_payload,
        )

    @torch.inference_mode()
    def sample(
        self,
        tensors: dict[str, torch.Tensor],
        n_samples: int = 1,
        max_poisson_rate: float = 1e8,
        generative_kwargs: dict | None = None,
    ) -> torch.Tensor:
        r"""Generate predictive samples from the posterior predictive distribution.

        The posterior predictive distribution is denoted as :math:`p(\hat{x} \mid x)`, where
        :math:`x` is the input data and :math:`\hat{x}` is the sampled data.

        We sample from this distribution by first sampling ``n_samples`` times from the posterior
        distribution :math:`q(z \mid x)` for a given observation, and then sampling from the
        likelihood :math:`p(\hat{x} \mid z)` for each of these.

        Parameters
        ----------
        tensors
            Dictionary of tensors passed into :meth:`~scvi.module.VAE.forward`.
        n_samples
            Number of Monte Carlo samples to draw from the distribution for each observation.
        max_poisson_rate
            The maximum value to which to clip the ``rate`` parameter of
            :class:`~scvi.distributions.Poisson`. Avoids numerical sampling issues when the
            parameter is very large due to the variance of the distribution.
        generative_kwargs
            Keyword args for ``generative()`` in fwd pass

        Returns
        -------
        Tensor on CPU with shape ``(n_obs, n_vars)`` if ``n_samples == 1``, else
        ``(n_obs, n_vars,)``.
        """
        from scvi.distributions import Poisson

        inference_kwargs = {"n_samples": n_samples}
        _, generative_outputs = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            generative_kwargs=generative_kwargs,
            compute_loss=False,
        )

        dist = generative_outputs[MODULE_KEYS.PX_KEY]
        if self.gene_likelihood == "poisson":
            # TODO: NEED TORCH MPS FIX for 'aten::poisson'
            dist = (
                Poisson(torch.clamp(dist.rate.to("cpu"), max=max_poisson_rate))
                if self.device.type == "mps"
                else Poisson(torch.clamp(dist.rate, max=max_poisson_rate))
            )

        # (n_obs, n_vars) if n_samples == 1, else (n_samples, n_obs, n_vars)
        samples = dist.sample()
        # (n_samples, n_obs, n_vars) -> (n_obs, n_vars, n_samples)
        samples = torch.permute(samples, (1, 2, 0)) if n_samples > 1 else samples

        return samples.cpu()

    @torch.inference_mode()
    @auto_move_data
    def marginal_ll(
        self,
        tensors: dict[str, torch.Tensor],
        n_mc_samples: int,
        return_mean: bool = False,
        n_mc_samples_per_pass: int = 1,
    ):
        """Compute the marginal log-likelihood of the data under the model.

        Parameters
        ----------
        tensors
            Dictionary of tensors passed into :meth:`~scvi.module.VAE.forward`.
        n_mc_samples
            Number of Monte Carlo samples to use for the estimation of the marginal log-likelihood.
        return_mean
            Whether to return the mean of marginal likelihoods over cells.
        n_mc_samples_per_pass
            Number of Monte Carlo samples to use per pass. This is useful to avoid memory issues.
        """
        from torch import logsumexp
        from torch.distributions import Normal

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = []
        if n_mc_samples_per_pass > n_mc_samples:
            warnings.warn(
                "Number of chunks is larger than the total number of samples, setting it to the "
                "number of samples",
                RuntimeWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            n_mc_samples_per_pass = n_mc_samples
        n_passes = int(np.ceil(n_mc_samples / n_mc_samples_per_pass))
        for _ in range(n_passes):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(
                tensors,
                inference_kwargs={"n_samples": n_mc_samples_per_pass},
                get_inference_input_kwargs={"full_forward_pass": True},
            )
            qz = inference_outputs[MODULE_KEYS.QZ_KEY]
            ql = inference_outputs[MODULE_KEYS.QL_KEY]
            z = inference_outputs[MODULE_KEYS.Z_KEY]
            library = inference_outputs[MODULE_KEYS.LIBRARY_KEY]

            # Reconstruction Loss
            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            # Log-probabilities
            p_z = (
                Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale)).log_prob(z).sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(library)
                    .sum(dim=-1)
                )
                q_l_x = ql.log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x
            if n_mc_samples_per_pass == 1:
                log_prob_sum = log_prob_sum.unsqueeze(0)

            to_sum.append(log_prob_sum)
        to_sum = torch.cat(to_sum, dim=0)
        batch_log_lkl = logsumexp(to_sum, dim=0) - np.log(n_mc_samples)
        if return_mean:
            batch_log_lkl = torch.mean(batch_log_lkl).item()
        else:
            batch_log_lkl = batch_log_lkl.cpu()
        return batch_log_lkl


from scvi.model import SCVI
from sc_reconstruction.models._base_model import BaseReconstructionModel

class ReconNLSCVI(SCVI, BaseReconstructionModel):
    """
    A structured SCVI model for reconstruction that adheres to the base class interface.
    """
    _module_cls = NormalVAE
    
    def __init__(
        self,
        adata: AnnData | None = None,
        registry: Dict | None = None,
        **kwargs 
    ):
        self.__class__.__name__ = "SCVI"
        super().__init__(adata=adata, registry=registry, **kwargs)

    def prepare(self, adata: Any, **kwargs) -> None:
        """Register adata with the SCVI data manager and cache the reference."""
        self.setup_anndata(adata, **kwargs)
        self.adata = adata

    def train(self, **train_kwargs) -> None:
        """Drop the unused ``save_path`` kwarg and delegate to ``SCVI.train``."""
        train_kwargs.pop('save_path', None)
        super().train(**train_kwargs)

    def _prepare_batch(self, X: np.ndarray) -> dict:
        """
        Convert a raw NumPy array into the mini_batch dictionary
        required by self.module.forward.
        """
        return {
            REGISTRY_KEYS.X_KEY: torch.from_numpy(X),
            REGISTRY_KEYS.BATCH_KEY: torch.zeros((X.shape[0], 1), dtype=torch.int64),
            REGISTRY_KEYS.LABELS_KEY: torch.zeros((X.shape[0], 1), dtype=torch.int64),
        }

    def _predict_batch(self, mini_batch: dict, inference_kwargs: dict) -> torch.Tensor:
        """
        Runs the model forward pass on a mini_batch and returns the reconstructed output.
        """
        self.module.eval()
        with torch.no_grad():
            _, generative_outputs = self.module.forward(
                tensors=mini_batch,
                inference_kwargs=inference_kwargs,
                compute_loss=False
            )
        return generative_outputs["px"].loc

    def _predict_batch_deterministic(self, mini_batch: dict, inference_kwargs: dict) -> torch.Tensor:
        """
        Runs the model forward pass on a mini_batch and returns the reconstructed output.
        """
        self.module.eval()
        with torch.no_grad():
            inference_outputs, generative_outputs = self.module.forward(
                tensors=mini_batch,
                inference_kwargs=inference_kwargs,
                compute_loss=False
            )
            qz = inference_outputs["qz"]  
            z_mean = qz.loc
            ql = inference_outputs["ql"]  
            l_mean = ql.loc
            
            generative_inputs = self.module._get_generative_input(
                mini_batch, inference_outputs

            )
            generative_inputs[MODULE_KEYS.Z_KEY] = z_mean
            if not self.use_observed_lib_size:
                generative_inputs[MODULE_KEYS.LIBRARY_KEY] = l_mean
            generative_outputs_deterministic = self.module.generative(**generative_inputs)
        
        return generative_outputs_deterministic["px"].loc

    def _predict_deterministic(self, 
                X: np.ndarray,
                inference_kwargs: dict = None) -> np.ndarray:
        """
        deterministic prediction for scvi style models
        """ 
        mini_batch = self._prepare_batch(X)
        pred_tensor = self._predict_batch_deterministic(mini_batch, inference_kwargs)
        return pred_tensor.cpu().numpy()

    def _predict_random(self, 
                X: np.ndarray,
                inference_kwargs: dict = None) -> np.ndarray:
        """
        random prediction for scvi style models
        
        Parameters:
        X: np.ndarray - Raw input data.
        inference_kwargs: Optional dictionary of inference parameters (default: {"n_samples": 1, "return_mean": True}).
        Returns:
        np.ndarray: The predicted/reconstructed output.
        """


        mini_batch = self._prepare_batch(X)
        pred_tensor = self._predict_batch(mini_batch, inference_kwargs)
        return pred_tensor.cpu().numpy()
        
    def predict(self, 
                X: np.ndarray,
                inference_kwargs: dict = None,
                pred_type = "random") -> np.ndarray:
        """
        Parameters:
        X: np.ndarray - Raw input data.
        inference_kwargs: Optional dictionary of inference parameters (default: {"n_samples": 1, "return_mean": True}).
        pred_type: Optional string indicating the type of prediction ("random" or "deterministic").
        """ 

        if pred_type == "random":
            return self._predict_random(X, inference_kwargs)
        elif pred_type == "deterministic":
            return self._predict_deterministic(X, inference_kwargs)
        else:
            raise ValueError(f"Invalid prediction type: {pred_type}")
            
    def get_latent_representation(self, X: np.ndarray, **inference_kwargs) -> np.ndarray:
        mini_batch = self._prepare_batch(X)
        self.module.eval()
        with torch.no_grad():
            inference_outputs, generative_outputs = self.module.forward(
                tensors=mini_batch,
                inference_kwargs=inference_kwargs,
                compute_loss=False
            )
            qz = inference_outputs["qz"]  
            z_mean = qz.loc
        return z_mean.cpu().numpy()

    def save(self, path: str, **kwargs) -> None:
        """Save the model module to ``path``.

        Args:
            path: Destination path; ``.pt`` is appended if no extension is set.
            kwargs: Forwarded to the parent class (e.g. ``save_anndata``,
                ``overwrite``).
        """
        # Add .pt extension if no .pt or .ckpt extension is present
        if not path.endswith('.pt') and not path.endswith('.ckpt'):
            path = path + '.pt'
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.module, path)
        print(f"Model saved to {path}")

    def load(self, path: str, map_location=None) -> None:
        """Load the model"""
        self.module = torch.load(path, map_location=map_location, weights_only=False)