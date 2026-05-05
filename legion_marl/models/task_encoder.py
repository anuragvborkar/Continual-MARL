"""
TaskEncoder: maps a mean-pooled joint observation vector to a task latent z.

In Phase 2 the encoder outputs a mean and log-variance (VAE-style), and z
is sampled via the reparameterisation trick during training.  At eval time
(or when gradients are not needed) the mean is used directly.

A paired Decoder reconstructs the input obs from z.  Training the
encoder+decoder jointly with a reconstruction loss gives the encoder
a gradient signal from day 1, before the DPMM has formed any clusters.
This forces z to be informative about the observation content (which
differs between tasks via zero-padding and semantics), enabling the DPMM
to cluster z into per-task components.

Input:  (batch, GLOBAL_OBS_DIM)
Output: z (batch, z_dim)  — sampled during training, mean at eval
        Also exposes: self.z_mean, self.z_logvar for KL loss computation.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskEncoder(nn.Module):
    """
    VAE-style task encoder with a paired reconstruction decoder.

    Args:
        obs_dim:    input dimension (padded observation, default 19)
        z_dim:      latent dimension (default 32)
        hidden_dim: hidden layer width (default 64)
    """

    def __init__(self, obs_dim: int = 19, z_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
        )
        self.mean_head   = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)

        # Decoder: reconstructs obs from z (used for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Exposed after each forward for KL loss computation
        self.z_mean:   torch.Tensor | None = None
        self.z_logvar: torch.Tensor | None = None

    def forward(self, joint_obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Args:
            joint_obs:     (N, obs_dim)
            deterministic: if True, always return mean (no sampling).
                           Used for KL updates to avoid stochastic gradients.

        Returns:
            z: (N, z_dim)
        """
        h = self.backbone(joint_obs)
        mean   = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-2.0, 2.0)

        self.z_mean   = mean
        self.z_logvar = logvar

        if self.training and not deterministic:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def reconstruction_loss(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """
        VAE-style reconstruction loss: MSE between input obs and decoded obs,
        plus the KL divergence of q(z|x) from N(0,I) as a prior regulariser.

        This is entirely self-supervised and requires no DPMM clusters.
        It gives the encoder a gradient signal from step 1, forcing z to
        encode information that distinguishes tasks.

        Args:
            joint_obs: (N, obs_dim) — with gradient

        Returns:
            scalar loss
        """
        h = self.backbone(joint_obs)
        mean   = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-2.0, 2.0)

        # Reparameterise
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z   = mean + eps * std

        # Reconstruction
        obs_recon = self.decoder(z)
        recon_loss = F.mse_loss(obs_recon, joint_obs)

        # KL from N(0,I) prior: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        kl_prior = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()

        return recon_loss + 0.1 * kl_prior