"""
Phase 1 entry point: sequential training on navigation → balance
with oracle task conditioning (one-hot z).

Run from the repo root:
    python run_phase1.py

This validates:
  - PaddedVmasTask observation padding is correct
  - TaskConditionedMlp builds and runs without errors
  - ContinualCallback correctly injects task_z and transfers weights
  - Sequential training loop completes
  - Forgetting metric is computed
"""

import torch
import torch.nn as nn

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import ExperimentConfig

from legion_marl.experiment.continual_runner import ContinualRunner
from legion_marl.models.task_conditioned_mlp import TaskConditionedMlpConfig
from legion_marl.tasks.task_config import PHASE_1_SEQUENCE, PHASE_2_SEQUENCE


def main():
    # ------------------------------------------------------------------
    # 1.  Experiment config
    #     Use small values here to verify the pipeline runs end-to-end.
    #     Scale up for real training (1M+ frames per task).
    # ------------------------------------------------------------------
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.train_device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.buffer_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Frames per task — set low for a smoke test, raise to ~1_000_000 for real runs
    experiment_config.max_n_frames = 150_000
    experiment_config.on_policy_collected_frames_per_batch = 6_000   # ~60 envs * 100 steps
    experiment_config.on_policy_n_envs_per_worker = 60
    experiment_config.on_policy_n_minibatch_iters = 45
    experiment_config.on_policy_minibatch_size = 400

    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 30_000
    experiment_config.evaluation_episodes = 10
    experiment_config.evaluation_deterministic_actions = True

    experiment_config.loggers = ["wandb"]      # switch to ["wandb"] for real runs
    experiment_config.create_json = False
    experiment_config.checkpoint_at_end = True
    experiment_config.checkpoint_interval = 96_000

    # ------------------------------------------------------------------
    # 2.  Algorithm config (MAPPO)
    # ------------------------------------------------------------------
    algorithm_config = MappoConfig.get_from_yaml()

    # ------------------------------------------------------------------
    # 3.  Model config
    #     Phase 1: use_oracle=True, z_dim will be overridden to n_tasks=2
    # ------------------------------------------------------------------
    model_config = TaskConditionedMlpConfig(
        z_dim=2,            # will be overridden to n_tasks in ContinualRunner
        task_encoder_hidden_dim=64,
        n_tasks=2,
        use_oracle=True,
        num_cells=[256, 256],
        activation_class=nn.Tanh,
    )

    # ------------------------------------------------------------------
    # 4.  Run
    # ------------------------------------------------------------------
    runner = ContinualRunner(
        task_sequence=PHASE_2_SEQUENCE,   # ["navigation", "balance"]
        algorithm_config=algorithm_config,
        model_config=model_config,
        experiment_config=experiment_config,
        seed=24,
        use_oracle=True,
        output_dir="./outputs/phase2",
    )

    runner.run()

    # ------------------------------------------------------------------
    # 5.  Print final metrics
    # ------------------------------------------------------------------
    runner.metrics_tracker.print_summary()

    print("\nPhase 1 complete. Shared model weights have been carried forward.")
    print("To proceed to Phase 2, change use_oracle=False and add the DPMM module.")


if __name__ == "__main__":
    main()
