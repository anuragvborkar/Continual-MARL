"""
CLMetricsTracker: computes and stores continual learning evaluation metrics.

Metrics (following LEGION / standard LRL conventions):
  - Forgetting (F): peak performance on task i minus performance on task i
    after training on all subsequent tasks.  Lower is better.  F in [-1, 1].
  - Forward Transfer (FT): performance on task i+1 after training on tasks
    0..i relative to a zero-shot baseline.  Higher is better.  FT in [0, 1].
  - Average Return: mean team reward across all tasks at the end of training.
  - ARI (Phase 2+): Adjusted Rand Index between DPMM cluster assignments
    and ground-truth task IDs.  Measures how well the DPMM separates tasks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch, sklearn


class CLMetricsTracker:
    """
    Accumulates per-task performance data and computes CL metrics.

    Usage:
        tracker = CLMetricsTracker(task_sequence=["navigation", "balance"])
        # After each task training, record per-task eval returns:
        tracker.record(trained_on="navigation", eval_returns={"navigation": 0.8})
        tracker.record(trained_on="balance", eval_returns={"navigation": 0.75, "balance": 0.85})
        print(tracker.forgetting())
        print(tracker.average_return())
    """

    def __init__(self, task_sequence: List[str]):
        self.task_sequence = task_sequence
        # Matrix: R[i][j] = performance on task j after training up to task i
        # Stored as { trained_on_task_name: { eval_task_name: float } }
        self._matrix: Dict[str, Dict[str, float]] = {}
        self._peak: Dict[str, float] = {}

    def record(self, trained_on: str, eval_returns: Dict[str, float]):
        """Record evaluation returns after training on `trained_on`."""
        self._matrix[trained_on] = eval_returns
        for task, ret in eval_returns.items():
            if ret > self._peak.get(task, float("-inf")):
                self._peak[task] = ret

    def forgetting(self) -> Dict[str, float]:
        """
        Forgetting for each non-final task:
          F_i = peak_i - R[last_task][task_i]

        Returns dict { task_name: forgetting_value }
        """
        if not self._matrix:
            return {}
        last_task = self.task_sequence[-1]
        final_evals = self._matrix.get(last_task, {})
        result = {}
        for task in self.task_sequence[:-1]:
            peak = self._peak.get(task)
            final = final_evals.get(task)
            if peak is not None and final is not None:
                result[task] = peak - final
        return result

    def average_forgetting(self) -> Optional[float]:
        f = self.forgetting()
        if not f:
            return None
        return float(np.mean(list(f.values())))

    def forward_transfer(
        self, zero_shot_returns: Dict[str, float]
    ) -> Dict[str, float]:
        """
        FT_i = R[task_{i-1}][task_i] - zero_shot_returns[task_i]

        zero_shot_returns: performance on each task trained from scratch
        with no prior tasks (your Phase 0 baselines).

        Returns dict { task_name: ft_value }
        """
        result = {}
        for idx, task in enumerate(self.task_sequence[1:], start=1):
            prior_task = self.task_sequence[idx - 1]
            prior_evals = self._matrix.get(prior_task, {})
            r_after_prior = prior_evals.get(task)
            zs = zero_shot_returns.get(task)
            if r_after_prior is not None and zs is not None:
                result[task] = r_after_prior - zs
        return result

    def average_return(self) -> Optional[float]:
        """Mean return across all tasks at the end of the sequence."""
        last_task = self.task_sequence[-1]
        final_evals = self._matrix.get(last_task, {})
        if not final_evals:
            return None
        return float(np.mean(list(final_evals.values())))

    @staticmethod
    def ari_score(
        z_samples: "torch.Tensor",
        true_task_ids: "torch.Tensor",
        dpmm_assignments: "torch.Tensor",
    ) -> float:
        """
        Adjusted Rand Index between DPMM cluster assignments and ground-truth
        task IDs.  Both inputs should be 1-D integer tensors of the same length.

        Requires sklearn.
        """
        try:
            from sklearn.metrics import adjusted_rand_score
            return float(
                adjusted_rand_score(
                    true_task_ids.cpu().numpy(),
                    dpmm_assignments.cpu().numpy(),
                )
            )
        except ImportError:
            raise ImportError("scikit-learn required for ARI: pip install scikit-learn")

    def print_summary(self):
        """Pretty-print the evaluation matrix and key metrics."""
        print("\n" + "=" * 60)
        print("CL METRICS SUMMARY")
        print("=" * 60)

        # Eval matrix
        print("\nEval matrix (row = trained on, col = eval on):")
        col_width = 12
        header = " " * col_width + "".join(
            f"{t[:col_width-1]:>{col_width}}" for t in self.task_sequence
        )
        print(header)
        for train_task in self.task_sequence:
            if train_task not in self._matrix:
                continue
            row = f"{train_task[:col_width-1]:>{col_width}}"
            for eval_task in self.task_sequence:
                val = self._matrix[train_task].get(eval_task)
                row += f"  {val:8.4f}" if val is not None else "        --"
            print(row)

        # Average return
        avg_ret = self.average_return()
        if avg_ret is not None:
            print(f"\nAverage Return (end of sequence): {avg_ret:.4f}")

        # Forgetting
        f = self.forgetting()
        if f:
            print("\nForgetting:")
            for task, val in f.items():
                print(f"  {task}: {val:+.4f}  {'(forgetting)' if val > 0 else '(improvement)'}")
            avg_f = self.average_forgetting()
            print(f"  Average: {avg_f:+.4f}")
