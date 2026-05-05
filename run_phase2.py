"""
Phase 2 entry point: sequential training on navigation → balance
with learned task encoder and DPMM clustering (use_oracle=False).

Changes from Phase 1:
  - use_oracle=False: z is computed by TaskEncoder, not a fixed one-hot
  - z_dim=32: task embedding is now a 32-dim learned vector
  - DPMM is active: clusters z samples, KL loss regularises encoder
  - ARI metric logged at end: measures cluster quality vs ground truth

Run from repo root:
    python run_phase2.py
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
    # 1. Experiment config — identical to Phase 1 except output dir
    # ------------------------------------------------------------------
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.train_device    = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.buffer_device   = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_config.max_n_frames = 300_000
    experiment_config.on_policy_collected_frames_per_batch = 6_000
    experiment_config.on_policy_n_envs_per_worker = 60
    experiment_config.on_policy_n_minibatch_iters = 45
    experiment_config.on_policy_minibatch_size = 400
    # Gradient clipping: use norm clipping (safer than value clipping for PPO)
    # clip_grad_val=5 with value clipping allows large global norms on wide layers
    experiment_config.clip_grad_norm = True
    experiment_config.clip_grad_val = 0.5

    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 30_000
    experiment_config.evaluation_episodes = 10
    experiment_config.evaluation_deterministic_actions = True

    experiment_config.loggers = ["csv"]
    experiment_config.create_json = False
    experiment_config.checkpoint_at_end = True
    experiment_config.checkpoint_interval = 96_000

    # ------------------------------------------------------------------
    # 2. Algorithm config
    # ------------------------------------------------------------------
    algorithm_config = MappoConfig.get_from_yaml()

    # ------------------------------------------------------------------
    # 3. Model config — Phase 2: use_oracle=False, z_dim=32
    # ------------------------------------------------------------------
    model_config = TaskConditionedMlpConfig(
        z_dim=32,
        task_encoder_hidden_dim=64,
        n_tasks=2,
        task_id=0,
        use_oracle=False,           # KEY DIFFERENCE from Phase 1
        num_cells=[256, 256],
        activation_class=nn.Tanh,
    )

    # ------------------------------------------------------------------
    # 4. Run
    # ------------------------------------------------------------------
    runner = ContinualRunner(
        task_sequence=PHASE_2_SEQUENCE,   # ["navigation", "balance"]
        algorithm_config=algorithm_config,
        model_config=model_config,
        experiment_config=experiment_config,
        seed=0,
        use_oracle=False,                  # KEY DIFFERENCE from Phase 1
        output_dir="./outputs/phase2",
    )

    runner.run()

    # ------------------------------------------------------------------
    # 5. Print metrics + ARI
    # ------------------------------------------------------------------
    runner.metrics_tracker.print_summary()

    # ARI: measure how well DPMM clusters match ground-truth task IDs
    # Collect z buffers and true task labels
    z_buffers = runner.get_z_buffers()
    if all(v is not None for v in z_buffers.values()):
        from legion_marl.tasks.task_config import TASK_CONFIGS
        z_all, labels_all = [], []
        for task_name, z_buf in z_buffers.items():
            z_all.append(z_buf)
            task_id = TASK_CONFIGS[task_name]["task_id"]
            labels_all.append(torch.full((z_buf.shape[0],), task_id, dtype=torch.long))
        z_cat = torch.cat(z_all, dim=0)
        labels_cat = torch.cat(labels_all, dim=0)

        if runner.dpmm is not None and runner.dpmm.n_clusters > 0:
            _, dpmm_assignments = runner.dpmm.soft_assign(z_cat)
            ari = runner.metrics_tracker.ari_score(z_cat, labels_cat, dpmm_assignments)
            print(f"\nDPMM Cluster Quality — ARI: {ari:.4f}  "
                  f"(1.0=perfect, 0.0=random, n_clusters={runner.dpmm.n_clusters})")

    # ------------------------------------------------------------------
    # 6. Save z buffers and produce visualisations
    # ------------------------------------------------------------------
    from legion_marl.visualize_z import (
        save_z_buffers, visualize_z_buffers, plot_cluster_counts
    )

    # Save buffers so you can re-run visualisation without re-training
    save_z_buffers(z_buffers, path="outputs/phase2/z_buffers.pt")

    # UMAP + t-SNE scatter (requires: pip install umap-learn scikit-learn)
    visualize_z_buffers(
        z_buffers,
        save_path="outputs/phase2/z_visualization.pdf",
        method="both",          # change to "umap" or "tsne" if you prefer one
        max_per_task=5000,
    )

    # Cluster count bar chart — read spawn counts from runner logs
    # Fill this list from the "[DPMM After <task>]" lines printed during training
    # e.g. cluster_log = [("balance", 1), ("navigation", 2), ...]
    cluster_log = [
        (task, runner.dpmm.n_clusters)   # approximation: final count per task
        for task in runner.task_sequence
    ]
    plot_cluster_counts(
        cluster_log,
        save_path="outputs/phase2/cluster_counts.pdf",
    )

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()