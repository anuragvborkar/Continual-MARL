"""
run_phase3_agent_ablation.py
============================
Phase 3 ablation study: effect of varying number of agents on a single task.

The task sequence holds the environment CONSTANT (navigation) but changes
the number of agents across tasks.  This isolates whether the continual
learning machinery (task encoder / DPMM) can distinguish tasks that share
the same reward structure but differ only in team size.

Default agent sequence: [2, 3, 4, 5, 6]
  → 5 "tasks", each being VMAS navigation with a different n_agents.

Design choices
--------------
* TASK_CONFIGS is extended at runtime with synthetic entries so that
  ContinualRunner / ContinualCallback see proper task_id / obs_dim metadata.
* Observations are already padded to GLOBAL_OBS_DIM by PaddedVmasTask, so no
  architectural change is needed.
* n_agents_per_task is configurable so you can run smaller ablations quickly.
* Both Phase 1 (oracle) and Phase 2 (learned encoder) are supported via
  USE_ORACLE flag.
* Results (eval matrix + CL metrics) are saved to JSON in output_dir.

Usage
-----
    # from the repo root (same dir as benchmarl/)
    python legion_marl/run_phase3_agent_ablation.py

Flags (edit the CONFIG block below):
    N_AGENTS_SEQUENCE   list of agent counts that define the task sequence
    USE_ORACLE          True = Phase 1 one-hot z,  False = Phase 2 learned z
    MAX_N_FRAMES        training frames per task
    N_ENVS              parallel environments
    OUTPUT_DIR          where results are saved
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Make sure legion_marl is importable when running from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import ExperimentConfig

from legion_marl.experiment.continual_callback import ContinualCallback
from legion_marl.experiment.continual_runner import ContinualRunner
from legion_marl.models.task_conditioned_mlp import TaskConditionedMlpConfig
from legion_marl.tasks.padded_vmas import PaddedVmasTask, PaddedVmasClass
from legion_marl.tasks import task_config as _tc

# ===========================================================================
# CONFIG  — edit here
# ===========================================================================

# Agent counts that form the task sequence.
# Each entry becomes one "task" (same navigation scenario, different team size).
N_AGENTS_SEQUENCE: List[int] = [2, 3, 4, 5, 6]

# Set True for Phase 1 (oracle one-hot z),  False for Phase 2 (learned z)
USE_ORACLE: bool = False

# Training budget per task
MAX_N_FRAMES: int = 300_000          # reduce to 50_000 for a quick smoke test

# Parallel VMAS environments
N_ENVS: int = 20

# Frames collected per PPO iteration
FRAMES_PER_BATCH: int = 2_000

# PPO epochs per collected batch
N_EPOCHS: int = 4

# Eval every N frames
EVAL_INTERVAL: int = 20_000

# Device
DEVICE: str = "cpu"          # "cuda" if GPU available

# Random seed
SEED: int = 42

# Where to save results
OUTPUT_DIR: str = "./continual_results/phase3_agent_ablation"

# ===========================================================================
# END CONFIG
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_key(n_agents: int) -> str:
    """Canonical string key for a navigation-N-agent task."""
    return f"navigation_n{n_agents}"


def _register_agent_ablation_tasks(n_agents_sequence: List[int]) -> None:
    """
    Inject synthetic entries into TASK_CONFIGS so the rest of the
    codebase (runner, callback, metrics) can look up obs_dim / task_id.

    Navigation native obs_dim from original TASK_CONFIGS = 18.
    GLOBAL_OBS_DIM = 19 (pads up to this).
    n_agents doesn't change the per-agent obs_dim for navigation.
    """
    base_obs_dim = _tc.TASK_CONFIGS["navigation"]["obs_dim"]   # 18
    for idx, n_agents in enumerate(n_agents_sequence):
        key = _task_key(n_agents)
        if key not in _tc.TASK_CONFIGS:
            _tc.TASK_CONFIGS[key] = {
                "n_agents":    n_agents,
                "obs_dim":     base_obs_dim,
                "task_id":     idx,
                "reward_scale": 1.0,
            }


def _make_navigation_task(n_agents: int):
    """
    Build a PaddedVmasTask for navigation with a custom n_agents.

    BenchMARL's YAML for navigation is used as the base config;
    we override n_agents and max_steps in task.config.
    """
    task = PaddedVmasTask.NAVIGATION.get_from_yaml()
    # Override the number of agents in the environment config dict
    task.config["n_agents"] = n_agents
    # Optionally cap episode length for faster iteration
    # task.config["max_steps"] = 100
    task.reward_scale = 1.0
    return task


# ---------------------------------------------------------------------------
# Subclass ContinualRunner to support the agent-count ablation
# ---------------------------------------------------------------------------

class AgentAblationRunner(ContinualRunner):
    """
    Thin subclass of ContinualRunner that overrides _make_task() so that
    task names of the form "navigation_nN" are dispatched to a navigation
    environment with N agents, instead of looking up an enum entry.
    """

    def __init__(self, n_agents_sequence: List[int], **kwargs):
        self.n_agents_sequence = n_agents_sequence
        # Build task_sequence as string keys
        task_sequence = [_task_key(n) for n in n_agents_sequence]
        super().__init__(task_sequence=task_sequence, **kwargs)

    # ------------------------------------------------------------------
    # Override task factory
    # ------------------------------------------------------------------
    def _make_task(self, task_name: str):
        # task_name is e.g. "navigation_n4"
        if task_name.startswith("navigation_n"):
            n_agents = int(task_name.split("navigation_n")[1])
            return _make_navigation_task(n_agents)
        # Fallback: delegate to parent for standard task names
        return super()._make_task(task_name)

    # ------------------------------------------------------------------
    # Slightly richer console summary
    # ------------------------------------------------------------------
    def _print_cl_metrics(self):
        super()._print_cl_metrics()
        print("\n[Ablation info]")
        for n_agents, task_name in zip(self.n_agents_sequence, self.task_sequence):
            print(f"  {task_name}  →  navigation with {n_agents} agents")

    # ------------------------------------------------------------------
    # Save results to JSON
    # ------------------------------------------------------------------
    def save_results(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "ablation": "agent_count",
            "task": "navigation",
            "n_agents_sequence": self.n_agents_sequence,
            "task_sequence": self.task_sequence,
            "use_oracle": self.use_oracle,
            "eval_matrix": {
                k: {k2: round(v2, 6) for k2, v2 in v.items()}
                for k, v in self._eval_matrix.items()
            },
            "peak_returns": {k: round(v, 6) for k, v in self._peak_returns.items()},
            "forgetting": self._compute_forgetting(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        out_path = output_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Ablation] Results saved → {out_path}")
        return out_path

    def _compute_forgetting(self) -> Dict[str, float]:
        forgetting = {}
        final_task = self.task_sequence[-1]
        final_evals = self._eval_matrix.get(final_task, {})
        for task_name in self.task_sequence[:-1]:
            peak = self._peak_returns.get(task_name)
            final = final_evals.get(task_name)
            if peak is not None and final is not None:
                forgetting[task_name] = round(peak - final, 6)
        return forgetting


# ---------------------------------------------------------------------------
# Build configs
# ---------------------------------------------------------------------------

def build_mappo_config() -> MappoConfig:
    return MappoConfig.get_from_yaml()


def build_experiment_config() -> ExperimentConfig:
    cfg = ExperimentConfig.get_from_yaml()

    cfg.max_n_frames = MAX_N_FRAMES
    cfg.on_policy_collected_frames_per_batch = FRAMES_PER_BATCH
    cfg.on_policy_n_envs_per_worker = N_ENVS
    cfg.on_policy_n_minibatch_iters = N_EPOCHS

    cfg.evaluation = True
    cfg.evaluation_interval = EVAL_INTERVAL
    cfg.evaluation_episodes = 10
    cfg.render = False

    cfg.train_device = DEVICE
    cfg.sampling_device = DEVICE

    cfg.loggers = ["csv"]
    cfg.create_json = True

    return cfg


def build_model_config(n_tasks: int) -> TaskConditionedMlpConfig:
    return TaskConditionedMlpConfig(
        z_dim=n_tasks if USE_ORACLE else 32,
        task_encoder_hidden_dim=64,
        n_tasks=n_tasks,
        task_id=0,          # overwritten per-task by ContinualRunner
        use_oracle=USE_ORACLE,
        num_cells=[256, 256],
        activation_class=torch.nn.Tanh,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  LEGION-MARL  |  Phase 3 Agent-Count Ablation")
    print("=" * 65)
    print(f"  Task             : navigation (VMAS)")
    print(f"  Agent sequence   : {N_AGENTS_SEQUENCE}")
    print(f"  Oracle mode      : {USE_ORACLE}")
    print(f"  Frames per task  : {MAX_N_FRAMES:,}")
    print(f"  Device           : {DEVICE}")
    print(f"  Seed             : {SEED}")
    print(f"  Output dir       : {OUTPUT_DIR}")
    print("=" * 65)

    # ----------------------------------------------------------------
    # 1. Register synthetic task configs so downstream code can look
    #    them up by name (e.g. for task_id, obs_dim).
    # ----------------------------------------------------------------
    _register_agent_ablation_tasks(N_AGENTS_SEQUENCE)

    n_tasks = len(N_AGENTS_SEQUENCE)

    # ----------------------------------------------------------------
    # 2. Build BenchMARL / model configs
    # ----------------------------------------------------------------
    mappo_cfg = build_mappo_config()
    exp_cfg = build_experiment_config()
    model_cfg = build_model_config(n_tasks)

    # ----------------------------------------------------------------
    # 3. Build and run the ablation runner
    # ----------------------------------------------------------------
    runner = AgentAblationRunner(
        n_agents_sequence=N_AGENTS_SEQUENCE,
        algorithm_config=mappo_cfg,
        model_config=model_cfg,
        experiment_config=exp_cfg,
        seed=SEED,
        use_oracle=USE_ORACLE,
        output_dir=OUTPUT_DIR,
    )

    runner.run()

    # ----------------------------------------------------------------
    # 4. Save results
    # ----------------------------------------------------------------
    runner.save_results(OUTPUT_DIR)

    print("\n✓ Ablation complete.")


if __name__ == "__main__":
    main()