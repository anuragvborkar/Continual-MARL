"""
DPMM: Dirichlet Process Mixture Model for task clustering.

Implements a Gaussian DPMM with memoized variational Bayes-style updates,
closely following the LEGION paper's approach.

Lifecycle (mirrors LEGION):
  - E-step (soft_assign):  runs every forward pass — assigns z to clusters
  - KL loss (kl_loss):     runs every ~50 optimizer steps — regularises encoder
  - M-step (update):       runs every ~100 optimizer steps — updates cluster
                           parameters and spawns new clusters if needed

The DPMM is NOT a nn.Module and has NO gradient — it is a pure numpy/torch
inference object. Only the KL loss flows gradients back through the encoder.

Cluster representation:
  Each cluster k has:
    - mean:  (z_dim,)
    - cov:   (z_dim, z_dim)  — diagonal for efficiency
    - n:     float           — memoized sufficient statistic (effective count)

New cluster creation:
  If the best cluster's log-likelihood for a z sample is below a threshold
  (controlled by log_alpha, the log DP concentration parameter), a new
  cluster is spawned at that z's location.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


class GaussianCluster:
    """Single Gaussian cluster with memoized sufficient statistics."""

    def __init__(self, mean: torch.Tensor, z_dim: int, device: str = "cpu"):
        self.mean = mean.clone().to(device)                         # (z_dim,)
        self.cov_diag = torch.ones(z_dim, device=device) * 0.5     # diagonal cov
        self.n = 1.0                                                # effective count
        self.z_dim = z_dim
        self.device = device

    def log_likelihood(self, z: torch.Tensor) -> torch.Tensor:
        """
        Diagonal Gaussian log-likelihood.
        z: (N, z_dim)  →  returns (N,)
        """
        diff = z - self.mean.unsqueeze(0)           # (N, z_dim)
        log_det = self.cov_diag.log().sum()         # scalar
        maha = (diff ** 2 / self.cov_diag.unsqueeze(0)).sum(-1)  # (N,)
        return -0.5 * (self.z_dim * 1.8379 + log_det + maha)     # 1.8379 = log(2π)

    def update(self, z_assigned: torch.Tensor, momentum: float = 0.95):
        """
        M-step: update mean and diagonal covariance from assigned z samples.
        z_assigned: (M, z_dim)

        Covariance is capped at max=4.0 per dim so the cluster stays tight enough
        to detect when a genuinely different-distribution z arrives (new task).
        Count accumulates additively so pruning thresholds remain meaningful.
        """
        if z_assigned.shape[0] == 0:
            return
        new_mean = z_assigned.mean(0)
        if z_assigned.shape[0] < 2:
            new_var = self.cov_diag
        else:
            new_var = z_assigned.var(0).clamp(min=1e-4, max=4.0)
        self.mean = momentum * self.mean + (1 - momentum) * new_mean
        cov_momentum = min(momentum, 0.9)
        self.cov_diag = cov_momentum * self.cov_diag + (1 - cov_momentum) * new_var
        self.n += z_assigned.shape[0]


class DPMM:
    """
    Dirichlet Process Mixture Model for task clustering.

    Args:
        z_dim:           dimensionality of task latent space
        alpha:           DP concentration parameter — higher = more clusters
        new_cluster_threshold: minimum log-likelihood below which a new
                         cluster is spawned (more negative = harder to spawn)
        min_cluster_size: clusters with n < this are pruned on M-step
        device:          torch device string
    """

    def __init__(
        self,
        z_dim: int = 32,
        alpha: float = 1.0,
        new_cluster_threshold: float = -100.0,  # fallback only; adaptive threshold is used
        min_cluster_size: float = 20.0,
        max_clusters: int = 50,
        device: str = "cpu",
    ):
        self.z_dim = z_dim
        self.log_alpha = torch.tensor(alpha).log()
        self.new_cluster_threshold = new_cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.device = device

        self.clusters: List[GaussianCluster] = []
        self.total_n: float = 0.0

        # Adaptive threshold: track running mean and std of best_ll across M-steps.
        # Spawn a new cluster when batch mean_ll drops more than `n_sigma` standard
        # deviations below the running mean. This tracks the encoder's current scale
        # and fires on genuine distributional shifts (task changes) rather than
        # within-task outliers.
        self._ll_running_mean: Optional[float] = None
        self._ll_running_var: float = 1.0
        self._ll_ema_alpha: float = 0.05   # smoothing; ~20 M-steps to adapt
        self._spawn_n_sigma: float = 2.5   # how many sigma below mean to trigger spawn

    # ------------------------------------------------------------------
    # E-step: soft cluster assignment
    # ------------------------------------------------------------------

    def soft_assign(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft cluster responsibilities for each z.

        Args:
            z: (N, z_dim)

        Returns:
            responsibilities: (N, K) — softmax over log-likelihoods
            cluster_ids:      (N,)   — argmax cluster assignment
        """
        z = z.to(self.device)

        if len(self.clusters) == 0:
            # No clusters yet — spawn first cluster at mean of z
            self._spawn_cluster(z.mean(0))

        # Compute log-likelihoods for each cluster: (N, K)
        log_liks = torch.stack(
            [c.log_likelihood(z) for c in self.clusters], dim=1
        )  # (N, K)

        # CRP prior: log P(k) ∝ log(n_k) for existing, log(alpha) for new
        cluster_counts = torch.tensor(
            [c.n for c in self.clusters], device=self.device
        )  # (K,)
        log_prior = torch.log(cluster_counts + 1e-8)  # (K,)

        log_posteriors = log_liks + log_prior.unsqueeze(0)  # (N, K)
        responsibilities = F.softmax(log_posteriors, dim=-1)  # (N, K)
        cluster_ids = log_posteriors.argmax(dim=-1)           # (N,)

        return responsibilities, cluster_ids

    # ------------------------------------------------------------------
    # KL loss (runs every ~50 steps, gradients flow through z)
    # ------------------------------------------------------------------

    def kl_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        KL divergence loss that regularises the encoder toward cluster means.

        For each z, compute soft responsibilities, then penalise distance
        to the responsibility-weighted cluster mean. This is a simplified
        version of LEGION's KL term that is fully differentiable.

        Args:
            z: (N, z_dim)  — with gradients (from encoder)

        Returns:
            scalar loss
        """
        if len(self.clusters) == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        z_dev = z.to(self.device)

        # Compute responsibilities (detached — DPMM params have no grad)
        with torch.no_grad():
            log_liks = torch.stack(
                [c.log_likelihood(z_dev) for c in self.clusters], dim=1
            )  # (N, K)
            cluster_counts = torch.tensor(
                [c.n for c in self.clusters], device=self.device
            )
            log_prior = torch.log(cluster_counts + 1e-8)
            responsibilities = F.softmax(log_liks + log_prior.unsqueeze(0), dim=-1)  # (N, K)

        # Weighted cluster means: (N, z_dim)
        cluster_means = torch.stack(
            [c.mean for c in self.clusters], dim=0
        )  # (K, z_dim)
        weighted_means = responsibilities @ cluster_means  # (N, z_dim)

        # KL loss: MSE between z and its responsibility-weighted target
        # This pulls the encoder output toward the nearest cluster centre
        loss = F.mse_loss(z, weighted_means.detach())
        return loss

    # ------------------------------------------------------------------
    # M-step: update cluster parameters
    # ------------------------------------------------------------------

    def update(self, z: torch.Tensor, momentum: float = 0.9):
        """
        M-step: update cluster parameters from a batch of z samples.

        Cluster spawning uses an ADAPTIVE threshold based on a running estimate
        of the mean and std of the batch best_ll. A new cluster is spawned when
        the *batch mean* best_ll drops more than `_spawn_n_sigma` standard
        deviations below the running mean — this fires on genuine task changes
        (mean ll drops 7-10 units) but not on within-task outliers (individual
        z values with low ll but a healthy batch mean).

        Args:
            z: (N, z_dim)  — detached (no grad needed)
        """
        z = z.detach().to(self.device)

        if len(self.clusters) == 0:
            self._spawn_cluster(z.mean(0))
            return

        # --- Assign each z to its best cluster ---
        log_liks = torch.stack(
            [c.log_likelihood(z) for c in self.clusters], dim=1
        )  # (N, K)
        best_log_liks, assignments = log_liks.max(dim=1)  # (N,), (N,)

        batch_mean_ll = best_log_liks.mean().item()
        batch_min_ll  = best_log_liks.min().item()

        # --- Update running mean/var of batch_mean_ll (Welford-style EMA) ---
        if self._ll_running_mean is None:
            self._ll_running_mean = batch_mean_ll
            self._ll_running_var  = 1.0
        else:
            delta = batch_mean_ll - self._ll_running_mean
            self._ll_running_mean += self._ll_ema_alpha * delta
            self._ll_running_var   = (1 - self._ll_ema_alpha) * (
                self._ll_running_var + self._ll_ema_alpha * delta ** 2
            )

        running_std = max(math.sqrt(self._ll_running_var), 0.5)
        adaptive_threshold = self._ll_running_mean - self._spawn_n_sigma * running_std

        # --- Spawn a new cluster if batch mean ll drops well below running mean ---
        # This fires on task changes (large sustained mean drop) not individual outliers.
        spawn_triggered = (
            batch_mean_ll < adaptive_threshold
            and len(self.clusters) < self.max_clusters
        )
        if spawn_triggered:
            # Use the poorly-fit z as the new cluster centre
            poor_fit_mask = best_log_liks < (batch_mean_ll - running_std)
            new_center = z[poor_fit_mask].mean(0) if poor_fit_mask.any() else z.mean(0)
            self._spawn_cluster(new_center)
            assignments[poor_fit_mask] = len(self.clusters) - 1

        # --- Diagnostic print ---
        n_poor = (best_log_liks < (self._ll_running_mean - running_std)).sum().item()
        print(
            f"[DPMM M-step] n_clusters={len(self.clusters)} | "
            f"best_ll mean={batch_mean_ll:.1f} min={batch_min_ll:.1f} | "
            f"running_mean={self._ll_running_mean:.1f} "
            f"std={running_std:.1f} "
            f"adaptive_thresh={adaptive_threshold:.1f} | "
            f"n_poor_fit(>1σ)={int(n_poor)} | "
            f"spawn={'YES' if spawn_triggered else 'no'}"
        )

        # --- Update each cluster with its assigned samples ---
        for k, cluster in enumerate(self.clusters):
            mask = assignments == k
            if mask.any():
                cluster.update(z[mask], momentum=momentum)

        # --- Prune tiny clusters ---
        self.clusters = [c for c in self.clusters if c.n >= self.min_cluster_size]
        self.total_n = sum(c.n for c in self.clusters)

        if len(self.clusters) == 0:
            self._spawn_cluster(z.mean(0))

    # ------------------------------------------------------------------
    # State management (for persistence across tasks)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise cluster state for cross-task persistence."""
        return {
            "clusters": [
                {
                    "mean": c.mean.cpu(),
                    "cov_diag": c.cov_diag.cpu(),
                    "n": c.n,
                }
                for c in self.clusters
            ],
            "total_n": self.total_n,
            "ll_running_mean": self._ll_running_mean,
            "ll_running_var":  self._ll_running_var,
        }

    def load_state_dict(self, state: dict):
        """Restore cluster state."""
        self.clusters = []
        for cs in state["clusters"]:
            c = GaussianCluster(cs["mean"], self.z_dim, self.device)
            c.cov_diag = cs["cov_diag"].to(self.device)
            c.n = cs["n"]
            self.clusters.append(c)
        self.total_n = state["total_n"]
        self._ll_running_mean = state.get("ll_running_mean", None)
        self._ll_running_var  = state.get("ll_running_var", 1.0)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def get_cluster_means(self) -> Optional[torch.Tensor]:
        """Returns (K, z_dim) tensor of cluster means, or None if no clusters."""
        if not self.clusters:
            return None
        return torch.stack([c.mean for c in self.clusters], dim=0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spawn_cluster(self, center: torch.Tensor):
        new_cluster = GaussianCluster(center, self.z_dim, self.device)
        # Start with enough effective count to survive the pruning step
        new_cluster.n = self.min_cluster_size * 2
        self.clusters.append(new_cluster)
        self.total_n += new_cluster.n