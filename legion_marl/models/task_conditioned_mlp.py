"""
TaskConditionedMlp: a BenchMARL Model that:

  1. Reads per-agent padded observations  (*leading, n_agents, GLOBAL_OBS_DIM)
  2. In oracle mode: builds a one-hot z of size n_tasks internally
     In encoder mode: mean-pools obs across agents, runs TaskEncoder → z
  3. Concatenates z to each agent's obs  (*leading, n_agents, obs_dim + z_dim)
  4. Passes the result through a shared MultiAgentMLP

The *leading dimensions vary between call sites:
  - Collection time: (n_envs, T, n_agents, obs_dim)
  - Optimizer time:  (batch, n_agents, obs_dim)
All shape handling is done by flattening leading dims before computation
and restoring them afterwards.

In Phase 1 (use_oracle=True):
  - task_id is set at construction and encodes the current task as a one-hot.
  - No external injection into the tensordict is needed.
  - The TaskEncoder is still instantiated (for weight transfer continuity)
    but is bypassed during the forward pass.

In Phase 2+ (use_oracle=False):
  - z is computed by the TaskEncoder on every forward pass.
  - self.last_z stores the latest batch of z values (detached) for the
    DPMM to consume asynchronously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Type

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig

from legion_marl.models.task_encoder import TaskEncoder
from legion_marl.tasks.task_config import GLOBAL_OBS_DIM


class TaskConditionedMlp(Model):
    """
    Task-conditioned multi-agent MLP policy.
    See module docstring for full details.
    """

    def __init__(
        self,
        # Task encoder / z config
        z_dim: int,
        task_encoder_hidden_dim: int,
        n_tasks: int,       # total tasks in sequence; one-hot size in oracle mode
        task_id: int,       # index of the current task (0-based), used in oracle mode
        use_oracle: bool,
        # Policy MLP config
        num_cells: Sequence[int],
        activation_class: Type[nn.Module],
        # BenchMARL Model base kwargs
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.z_dim = z_dim
        self.n_tasks = n_tasks
        self.task_id = task_id
        self.use_oracle = use_oracle

        obs_dim = GLOBAL_OBS_DIM

        # --- Task encoder (always built; bypassed in oracle mode) ---
        self.task_encoder = TaskEncoder(
            obs_dim=obs_dim,
            z_dim=z_dim,
            hidden_dim=task_encoder_hidden_dim,
        )

        # --- Policy MLP: per-agent input is obs + z ---
        self.policy_mlp = MultiAgentMLP(
            n_agent_inputs=obs_dim + z_dim,
            n_agent_outputs=self.output_leaf_spec.shape[-1],
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            num_cells=list(num_cells),
            activation_class=activation_class,
        )

        # Stores last z batch for DPMM (detached, shape (N, z_dim))
        self.last_z: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # BenchMARL checks
    # ------------------------------------------------------------------
    def _perform_checks(self):
        super()._perform_checks()
        obs_spec = self.input_spec.get(("agents", "observation"), None)
        if obs_spec is None:
            obs_spec = self.input_spec.get("observation", None)
        if obs_spec is not None:
            assert obs_spec.shape[-1] == GLOBAL_OBS_DIM, (
                f"Expected padded obs dim {GLOBAL_OBS_DIM}, "
                f"got {obs_spec.shape[-1]}. Make sure PaddedVmasTask is used."
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # obs shape: (*leading, n_agents, obs_dim)
        # leading is (batch,) at optimizer time, (n_envs, T) at collection time
        obs = tensordict.get((self.agent_group, "observation"))
        leading_shape = obs.shape[:-2]          # everything before (n_agents, obs_dim)
        N = 1
        for d in leading_shape:
            N *= d                              # flattened leading size

        # Flatten to (N, n_agents, obs_dim) for computation
        obs_flat = obs.reshape(N, self.n_agents, obs.shape[-1])

        # --- Compute z (N, z_dim) ---
        if self.use_oracle:
            z_flat = torch.zeros(N, self.z_dim, device=obs.device, dtype=obs.dtype)
            z_flat[:, self.task_id] = 1.0
        else:
            joint_obs = obs_flat.mean(dim=1)        # (N, obs_dim)
            z_flat = self.task_encoder(joint_obs)   # (N, z_dim)

        # Store for DPMM
        self.last_z = z_flat.detach()

        # Detach z before passing to policy: the encoder is trained
        # exclusively by the KL loss via its own optimizer. Allowing
        # PPO gradients to also flow through the encoder causes gradient
        # accumulation from two optimizers on the same weights -> NaN.
        z_flat_detached = z_flat.detach()

        # Restore leading dims: (N, z_dim) -> (*leading, z_dim)
        z = z_flat_detached.reshape(*leading_shape, self.z_dim)

        # Expand to per-agent: (*leading, n_agents, z_dim)
        z_expanded = z.unsqueeze(-2).expand(*leading_shape, self.n_agents, self.z_dim)

        # Safety: replace any NaN in z with zeros to prevent env crash
        if not torch.isfinite(z_expanded).all():
            z_expanded = torch.nan_to_num(z_expanded, nan=0.0)

        # Concatenate with obs: (*leading, n_agents, obs_dim + z_dim)
        policy_input = torch.cat([obs, z_expanded], dim=-1)

        # Diagnostics: detect NaN source before it reaches VMAS
        if not torch.isfinite(obs).all():
            print(f'[NaN] obs contains NaN/Inf, shape={obs.shape}')
        if not torch.isfinite(z_flat_detached).all():
            print(f'[NaN] z contains NaN/Inf')
        if not torch.isfinite(policy_input).all():
            print(f'[NaN] policy_input contains NaN/Inf')

        # Policy forward
        res = self.policy_mlp(policy_input)

        if not torch.isfinite(res).all():
            print(f'[NaN] policy_mlp output NaN/Inf — checking weights:')
            for name, p in self.policy_mlp.named_parameters():
                if not torch.isfinite(p).all():
                    print(f'  param {name}: NaN/Inf detected, norm={p.norm():.2f}')
            # Replace NaN output with zeros to avoid env crash and get more steps
            res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)

        tensordict.set(self.out_key, res)
        return tensordict


# ------------------------------------------------------------------
# Config dataclass (BenchMARL convention)
# ------------------------------------------------------------------
@dataclass
class TaskConditionedMlpConfig(ModelConfig):
    """
    Configuration for TaskConditionedMlp.

    task_id and n_tasks are set per-task by ContinualRunner before each
    Experiment is constructed.

    Args:
        z_dim: dimensionality of the task embedding.
            In oracle mode this equals n_tasks (one-hot size).
        task_encoder_hidden_dim: hidden layer width of the task encoder.
        n_tasks: total tasks in the sequence (one-hot size in oracle mode).
        task_id: index of the current task (set by ContinualRunner).
        use_oracle: True = Phase 1 (one-hot z), False = Phase 2+ (learned z).
        num_cells: hidden layer widths for the policy MLP.
        activation_class: activation function for the policy MLP.
    """

    z_dim: int = 32
    task_encoder_hidden_dim: int = 64
    n_tasks: int = 2
    task_id: int = 0
    use_oracle: bool = True

    num_cells: Sequence[int] = field(default_factory=lambda: [256, 256])
    activation_class: Type[nn.Module] = nn.Tanh

    @staticmethod
    def associated_class():
        return TaskConditionedMlp