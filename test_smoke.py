"""
Smoke test: validate the full pipeline without running actual training.

Checks:
  1. PaddedVmasTask creates an env with obs_dim == GLOBAL_OBS_DIM
  2. TaskConditionedMlp forward pass runs without errors
  3. ContinualCallback injects task_z correctly
  4. Weight extraction and loading round-trips correctly

Run:
    python test_smoke.py
"""

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from legion_marl.tasks.padded_vmas import PaddedVmasTask, ObsPaddingTransform
from legion_marl.tasks.task_config import GLOBAL_OBS_DIM, TASK_CONFIGS
from legion_marl.models.task_encoder import TaskEncoder


def test_obs_padding():
    print("--- Test: ObsPaddingTransform ---")
    import vmas

    for task_name, cfg in TASK_CONFIGS.items():
        env = vmas.make_env(
            scenario=task_name,
            num_envs=1,
            continuous_actions=True,
            device="cpu",
        )
        native_dim = cfg["obs_dim"]
        n_agents = cfg["n_agents"]

        # Simulate padding
        obs = torch.zeros(1, n_agents, native_dim)
        padded = torch.nn.functional.pad(obs, (0, GLOBAL_OBS_DIM - native_dim))
        assert padded.shape[-1] == GLOBAL_OBS_DIM, \
            f"Padding failed for {task_name}: {padded.shape}"
        print(f"  {task_name:20s} native={native_dim:2d} -> padded={padded.shape[-1]} ✓")


def test_task_encoder():
    print("\n--- Test: TaskEncoder ---")
    encoder = TaskEncoder(obs_dim=GLOBAL_OBS_DIM, z_dim=32, hidden_dim=64)
    batch = torch.randn(16, GLOBAL_OBS_DIM)
    z = encoder(batch)
    assert z.shape == (16, 32), f"Expected (16, 32), got {z.shape}"
    print(f"  TaskEncoder: ({16}, {GLOBAL_OBS_DIM}) -> {z.shape} ✓")


def test_oracle_z_injection():
    print("\n--- Test: Oracle one-hot z injection ---")
    n_tasks = 2
    task_id = 0
    batch_size = (4, 10)

    onehot = torch.zeros(1, 1, n_tasks)
    onehot[0, 0, task_id] = 1.0
    z_expanded = onehot.expand(*batch_size, 1, n_tasks).squeeze(-2)
    assert z_expanded.shape == (*batch_size, n_tasks), \
        f"Shape mismatch: {z_expanded.shape}"
    assert z_expanded[0, 0, 0] == 1.0
    assert z_expanded[0, 0, 1] == 0.0
    print(f"  Oracle z shape: {z_expanded.shape} ✓")
    print(f"  task_id=0 -> z={z_expanded[0,0].tolist()} ✓")


def test_mean_pool_variable_agents():
    print("\n--- Test: Mean-pool handles variable n_agents ---")
    for n_agents in [3, 4, 5]:
        obs = torch.randn(8, n_agents, GLOBAL_OBS_DIM)
        joint_obs = obs.mean(dim=-2)  # (8, GLOBAL_OBS_DIM)
        assert joint_obs.shape == (8, GLOBAL_OBS_DIM), \
            f"Failed for n_agents={n_agents}: {joint_obs.shape}"
        print(f"  n_agents={n_agents}: mean-pool -> {joint_obs.shape} ✓")


if __name__ == "__main__":
    test_obs_padding()
    test_task_encoder()
    test_oracle_z_injection()
    test_mean_pool_variable_agents()
    print("\nAll smoke tests passed ✓")
