"""
ContinualRunner: orchestrates sequential training over a list of VMAS tasks
using BenchMARL's Experiment infrastructure.

For each task in the sequence:
  1. Build a fresh Experiment with a ContinualCallback carrying shared weights
  2. Run the experiment (BenchMARL's standard training loop)
  3. Extract trained weights and pass them to the next task
  4. After each task, run cross-task evaluation and compute CL metrics

This class does NOT subclass Experiment — it wraps it.  This keeps all of
BenchMARL's internals untouched.

Phase 1 usage:
    runner = ContinualRunner(
        task_sequence=["navigation", "balance"],
        algorithm_config=mappo_config,
        model_config=task_conditioned_mlp_config,   # use_oracle=True
        experiment_config=exp_config,
        seed=0,
        use_oracle=True,
    )
    runner.run()
    results = runner.get_cl_metrics()
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

import torch

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models import MlpConfig

from legion_marl.experiment.continual_callback import ContinualCallback
from legion_marl.metrics.cl_metrics import CLMetricsTracker
from legion_marl.models.dpmm import DPMM
from legion_marl.models.task_conditioned_mlp import TaskConditionedMlpConfig
from legion_marl.tasks.padded_vmas import PaddedVmasTask
from legion_marl.tasks.task_config import TASK_CONFIGS


class CaptureCallback(Callback):
    """Captures eval rollouts from _evaluation_loop for return computation."""
    def __init__(self):
        super().__init__()
        self.rollouts = None

    def on_evaluation_end(self, rollouts):
        self.rollouts = rollouts


class ContinualRunner:
    """
    Sequential continual learning runner.

    Args:
        task_sequence: list of task name strings, e.g. ["navigation", "balance"]
        algorithm_config: a BenchMARL AlgorithmConfig (e.g. MappoConfig instance)
        model_config: a TaskConditionedMlpConfig instance
        experiment_config: a BenchMARL ExperimentConfig instance
        seed: random seed
        use_oracle: Phase 1 = True; Phase 2+ = False
        output_dir: where to save results and checkpoints
    """

    def __init__(
        self,
        task_sequence: List[str],
        algorithm_config: MappoConfig,
        model_config: TaskConditionedMlpConfig,
        experiment_config: ExperimentConfig,
        seed: int = 0,
        use_oracle: bool = True,
        output_dir: str = "./continual_results",
    ):
        self.task_sequence = task_sequence
        self.algorithm_config = algorithm_config
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.seed = seed
        self.use_oracle = use_oracle
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_tasks = len(task_sequence)

        # Shared model weights carried forward between tasks (None = first task)
        self._shared_model_state: Optional[Dict] = None

        # Accumulated z buffers per task (for DPMM in Phase 2+)
        self._z_buffers: Dict[str, Optional[torch.Tensor]] = {}

        # CL metrics tracker
        self.metrics_tracker = CLMetricsTracker(task_sequence)

        # Per-task peak performance (needed for forgetting computation)
        # { task_name: float }
        self._peak_returns: Dict[str, float] = {}

        # After-training per-task evaluations
        # { trained_up_to_task: { eval_task: mean_return } }
        self._eval_matrix: Dict[str, Dict[str, float]] = {}

        # Shared DPMM instance — persists across all tasks
        # Owned here so cluster state accumulates over the full sequence
        # DPMM device: use train_device from experiment_config if available
        _dpmm_device = getattr(experiment_config, 'train_device', 'cpu') or 'cpu'
        self.dpmm: Optional[DPMM] = (
            None if use_oracle else
            DPMM(
                z_dim=model_config.z_dim,
                alpha=1.0,
                new_cluster_threshold=-100.0,  # fallback only; adaptive threshold used
                min_cluster_size=20.0,
                max_clusters=50,               # raised: adaptive logic prevents explosion
                device=_dpmm_device,
            )
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """Run the full sequential training loop."""
        for task_idx, task_name in enumerate(self.task_sequence):
            print(f"\n{'='*60}")
            print(f"  Task {task_idx + 1}/{self.n_tasks}: {task_name}")
            print(f"{'='*60}\n")

            self._train_task(task_idx, task_name)

            # After training: evaluate on ALL tasks seen so far
            eval_returns = self._evaluate_all_seen_tasks(task_idx)
            self._eval_matrix[task_name] = eval_returns

            # Update peak returns
            if task_name in eval_returns:
                self._peak_returns[task_name] = max(
                    self._peak_returns.get(task_name, float("-inf")),
                    eval_returns[task_name],
                )

        print("\n\nSequential training complete.")
        self._print_cl_metrics()

    # ------------------------------------------------------------------
    # Per-task training
    # ------------------------------------------------------------------

    def _train_task(self, task_idx: int, task_name: str):
        """Build and run a BenchMARL Experiment for one task."""

        # --- Build model_config with correct n_tasks and use_oracle ---
        model_config = copy.deepcopy(self.model_config)
        model_config.n_tasks = self.n_tasks
        model_config.task_id = task_idx
        model_config.use_oracle = self.use_oracle
        # In oracle mode z_dim == n_tasks (one-hot)
        if self.use_oracle:
            model_config.z_dim = self.n_tasks

        # --- Build callback ---
        callback = ContinualCallback(
            task_name=task_name,
            task_id=task_idx,
            n_tasks=self.n_tasks,
            z_dim=model_config.z_dim,
            use_oracle=self.use_oracle,
            shared_model_state=self._shared_model_state,
            dpmm=self.dpmm,
        )

        # --- Build task ---
        task = self._make_task(task_name)

        # --- Build critic config (plain MLP — no task conditioning needed) ---
        critic_config = MlpConfig(num_cells=[256, 256], layer_class=torch.nn.Linear, activation_class=model_config.activation_class)

        # --- Build experiment ---
        exp = Experiment(
            task=task,
            algorithm_config=copy.deepcopy(self.algorithm_config),
            model_config=model_config,
            critic_model_config=critic_config,
            seed=self.seed,
            config=copy.deepcopy(self.experiment_config),
            callbacks=[callback],
        )

        # --- Run ---
        exp.run()

        # --- Extract shared weights for next task ---
        self._shared_model_state = callback.extract_shared_weights()

        # --- Store z buffer (Phase 2+) ---
        self._z_buffers[task_name] = callback.extract_z_buffer()

        # --- Log DPMM state after task ---
        if self.dpmm is not None:
            print(f'[DPMM] After {task_name}: '
                  f'n_clusters={self.dpmm.n_clusters}, '
                  f'cluster_sizes={[round(c.n,1) for c in self.dpmm.clusters]}')

    # ------------------------------------------------------------------
    # Cross-task evaluation
    # ------------------------------------------------------------------

    def _evaluate_all_seen_tasks(self, trained_up_to_idx: int) -> Dict[str, float]:
        """
        After training on tasks[0..trained_up_to_idx], evaluate performance
        on each of those tasks using the current shared weights.

        Returns: { task_name: mean_episode_return }
        """
        if self._shared_model_state is None:
            return {}

        returns: Dict[str, float] = {}
        for eval_idx in range(trained_up_to_idx + 1):
            eval_task_name = self.task_sequence[eval_idx]
            mean_ret = self._evaluate_single_task(eval_task_name, eval_idx)
            returns[eval_task_name] = mean_ret
            print(f"  Eval [{eval_task_name}]: mean_return = {mean_ret:.4f}")

        return returns

    def _evaluate_single_task(self, task_name: str, task_id: int) -> float:
        """
        Build a full Experiment for one task, load shared weights, run
        evaluation, and return mean episode return.

        We run a full experiment with minimal frames so that BenchMARL sets
        up the test_env and policy correctly, then call exp.evaluate() and
        extract the return directly from the rollouts via a capture callback.
        """
        model_config = copy.deepcopy(self.model_config)
        model_config.n_tasks = self.n_tasks
        model_config.task_id = task_id
        model_config.use_oracle = self.use_oracle
        if self.use_oracle:
            model_config.z_dim = self.n_tasks

        eval_config = copy.deepcopy(self.experiment_config)
        eval_config.max_n_frames = eval_config.on_policy_collected_frames_per_batch
        eval_config.evaluation = True
        eval_config.evaluation_interval = eval_config.on_policy_collected_frames_per_batch
        eval_config.render = False
        # Use minimal loggers for eval to avoid polluting training logs
        eval_config.loggers = []
        eval_config.create_json = False

        capture_cb = CaptureCallback()

        continual_cb = ContinualCallback(
            task_name=task_name,
            task_id=task_id,
            n_tasks=self.n_tasks,
            z_dim=model_config.z_dim,
            use_oracle=self.use_oracle,
            shared_model_state=self._shared_model_state,
        )

        task = self._make_task(task_name)
        critic_config = MlpConfig(
            num_cells=[256, 256],
            layer_class=torch.nn.Linear,
            activation_class=model_config.activation_class,
        )

        exp = Experiment(
            task=task,
            algorithm_config=copy.deepcopy(self.algorithm_config),
            model_config=model_config,
            critic_model_config=critic_config,
            seed=self.seed,
            config=eval_config,
            callbacks=[continual_cb, capture_cb],
        )

        exp.run()  # runs one collection step + triggers evaluation at the end

        # Compute mean return directly from captured rollouts.
        # Each rollout td has shape (n_episodes, T) with reward at
        # ("next", group_name, "reward") shape (n_episodes, T, n_agents, 1)
        # or ("next", "agents", "reward"). We sum over T and mean over
        # episodes and agents to get a scalar per rollout.
        mean_return = 0.0
        if capture_cb.rollouts is not None and len(capture_cb.rollouts) > 0:
            episode_returns = []
            for td in capture_cb.rollouts:
                # Try common reward key locations used by BenchMARL
                reward = None
                for key in [
                    ("next", "agents", "reward"),
                    ("next", "reward"),
                ]:
                    try:
                        reward = td[key]
                        break
                    except KeyError:
                        continue
                # Also try group-named key
                if reward is None:
                    try:
                        group = list(exp.group_map.keys())[0]
                        reward = td["next", group, "reward"]
                    except (KeyError, StopIteration):
                        pass
                if reward is not None:
                    # reward shape: (n_episodes, T, n_agents, 1) or (T, n_agents, 1)
                    # sum over time, mean over episodes and agents
                    ep_return = reward.sum(-3).mean()  # sum T, then mean rest
                    episode_returns.append(ep_return.item())

            if episode_returns:
                mean_return = float(sum(episode_returns) / len(episode_returns))
            else:
                # Last fallback: use the collection mean_return (not eval but better than 0)
                mean_return = float(exp.mean_return) if exp.mean_return else 0.0
                print(f"[ContinualRunner] Warning: could not find reward key in rollout for {task_name}. "
                      f"Keys: {list(capture_cb.rollouts[0].keys(True, True)) if capture_cb.rollouts else 'none'}")
        else:
            mean_return = float(exp.mean_return) if exp.mean_return else 0.0

        exp.close()
        return float(mean_return)

    # ------------------------------------------------------------------
    # CL metrics
    # ------------------------------------------------------------------

    def _print_cl_metrics(self):
        print("\n--- Continual Learning Metrics ---")
        print("Eval matrix (rows=after training on task, cols=eval task):")
        header = "               " + "  ".join(f"{t[:8]:>8}" for t in self.task_sequence)
        print(header)
        for train_task in self.task_sequence:
            if train_task not in self._eval_matrix:
                continue
            row = f"{train_task[:14]:>14} "
            for eval_task in self.task_sequence:
                val = self._eval_matrix[train_task].get(eval_task, None)
                row += f"  {val:8.4f}" if val is not None else "       --"
            print(row)

        # Forgetting
        print("\nForgetting (per task, computed at end of sequence):")
        final_task = self.task_sequence[-1]
        final_evals = self._eval_matrix.get(final_task, {})
        for task_name in self.task_sequence[:-1]:
            peak = self._peak_returns.get(task_name, None)
            final = final_evals.get(task_name, None)
            if peak is not None and final is not None:
                forgetting = peak - final
                print(f"  {task_name}: forgetting = {forgetting:.4f}  (peak={peak:.4f}, final={final:.4f})")

        # Forward transfer
        print("\nForward Transfer (per task, vs zero-shot baseline):")
        print("  (Requires zero-shot baseline — run standalone baselines to compare)")

    def get_eval_matrix(self) -> Dict[str, Dict[str, float]]:
        return self._eval_matrix

    def get_z_buffers(self) -> Dict[str, Optional[torch.Tensor]]:
        return self._z_buffers

    # ------------------------------------------------------------------
    # Task factory
    # ------------------------------------------------------------------

    def _make_task(self, task_name: str):
        """Instantiate a PaddedVmasTask for the given task name."""
        from legion_marl.tasks.task_config import TASK_CONFIGS
        task_enum_name = task_name.upper()
        task_enum = getattr(PaddedVmasTask, task_enum_name)
        task = task_enum.get_from_yaml()
        # Inject reward scale so PaddedVmasClass.get_env_transforms applies it
        reward_scale = TASK_CONFIGS.get(task_name, {}).get("reward_scale", 1.0)
        task.reward_scale = reward_scale
        return task