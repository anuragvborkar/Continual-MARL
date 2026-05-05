"""
visualize_z.py — UMAP and t-SNE visualisation of task encoder z buffers.

Run AFTER a ContinualRunner experiment completes.  Pass the runner object
directly, or save/load the z buffers from disk.

Usage (inline after run_phase2.py):
    from visualize_z import visualize_z_buffers
    visualize_z_buffers(runner.get_z_buffers(), save_path="imgs/z_umap.pdf")

Requirements:
    pip install umap-learn matplotlib scikit-learn
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba


# ── colour palette (one per task, colour-blind friendly) ─────────────────────
TASK_COLOURS = [
    "#4C72B0",  # blue        — balance
    "#DD8452",  # orange      — navigation
    "#55A868",  # green       — flocking
    "#C44E52",  # red         — transport
    "#8172B2",  # purple      — reverse_transport
    "#937860",  # brown       — discovery
    "#DA8BC3",  # pink        — spare
    "#8C8C8C",  # grey        — spare
]


def _subsample(z: torch.Tensor, max_samples: int = 5000) -> torch.Tensor:
    """Randomly subsample if buffer is very large (speeds up embedding)."""
    if z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0])[:max_samples]
        return z[idx]
    return z


def _prepare_data(
    z_buffers: Dict[str, Optional[torch.Tensor]],
    max_per_task: int = 5000,
):
    """
    Stack z buffers from all tasks into a single numpy array with labels.

    Returns
    -------
    X       : np.ndarray  (N, z_dim)
    labels  : np.ndarray  (N,)  integer task index
    names   : list[str]   task names in order
    colours : np.ndarray  (N, 4)  per-sample RGBA
    """
    arrays, label_list, names = [], [], []
    for task_idx, (task_name, z_buf) in enumerate(z_buffers.items()):
        if z_buf is None or z_buf.shape[0] == 0:
            print(f"  [warn] z buffer for '{task_name}' is empty, skipping.")
            continue
        z_sub = _subsample(z_buf.cpu().float(), max_per_task).numpy()
        arrays.append(z_sub)
        label_list.append(np.full(z_sub.shape[0], task_idx, dtype=int))
        names.append(task_name)
        print(f"  {task_name}: {z_sub.shape[0]} samples, z_dim={z_sub.shape[1]}")

    if not arrays:
        raise ValueError("All z buffers are empty.")

    X      = np.concatenate(arrays, axis=0)
    labels = np.concatenate(label_list, axis=0)
    colours = np.array([to_rgba(TASK_COLOURS[l % len(TASK_COLOURS)])
                        for l in labels])
    return X, labels, colours, names


def _make_legend(names: list, ax) -> None:
    patches = [
        mpatches.Patch(color=TASK_COLOURS[i % len(TASK_COLOURS)], label=name)
        for i, name in enumerate(names)
    ]
    ax.legend(handles=patches, loc="best", fontsize=8, framealpha=0.8)


def visualize_z_buffers(
    z_buffers: Dict[str, Optional[torch.Tensor]],
    save_path: str = "imgs/z_visualization.pdf",
    method: str = "both",          # "umap", "tsne", or "both"
    max_per_task: int = 5000,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 15.0,
    random_state: int = 42,
    dpi: int = 150,
    title_prefix: str = "",
) -> None:
    """
    Produce UMAP and/or t-SNE scatter plots of z embeddings coloured by task.

    Parameters
    ----------
    z_buffers      : dict mapping task_name → (N, z_dim) tensor or None
    save_path      : output file path (.pdf or .png)
    method         : "umap", "tsne", or "both"
    max_per_task   : max z samples per task (subsampled randomly if exceeded)
    umap_n_neighbors, umap_min_dist : UMAP hyperparameters
    tsne_perplexity : t-SNE perplexity
    random_state   : reproducibility seed
    dpi            : figure resolution
    title_prefix   : optional prefix for figure titles
    """
    print("[visualize_z] Preparing data...")
    X, labels, colours, names = _prepare_data(z_buffers, max_per_task)
    print(f"  Total samples: {X.shape[0]}, z_dim: {X.shape[1]}")

    do_umap = method in ("umap", "both")
    do_tsne = method in ("tsne", "both")

    n_plots = int(do_umap) + int(do_tsne)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    # ── UMAP ─────────────────────────────────────────────────────────────────
    if do_umap:
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn:  pip install umap-learn")

        print("[visualize_z] Running UMAP...")
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            n_components=2,
            random_state=random_state,
            verbose=False,
        )
        X_2d = reducer.fit_transform(X)
        ax = axes[ax_idx]
        ax.scatter(X_2d[:, 0], X_2d[:, 1],
                   c=colours, s=4, alpha=0.5, linewidths=0)
        _make_legend(names, ax)
        ax.set_title(f"{title_prefix}UMAP of task embeddings $z$", fontsize=10)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.set_xticks([]); ax.set_yticks([])
        ax_idx += 1

    # ── t-SNE ────────────────────────────────────────────────────────────────
    if do_tsne:
        from sklearn.manifold import TSNE

        print("[visualize_z] Running t-SNE (may take a moment)...")
        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            random_state=random_state,
            max_iter=2000,
            verbose=0,
        )
        X_2d = tsne.fit_transform(X)
        ax = axes[ax_idx]
        ax.scatter(X_2d[:, 0], X_2d[:, 1],
                   c=colours, s=4, alpha=0.5, linewidths=0)
        _make_legend(names, ax)
        ax.set_title(f"{title_prefix}t-SNE of task embeddings $z$", fontsize=10)
        ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[visualize_z] Saved to {save_path}")


def save_z_buffers(
    z_buffers: Dict[str, Optional[torch.Tensor]],
    path: str = "z_buffers.pt",
) -> None:
    """Save z buffers to disk so you can visualise without re-running training."""
    serialisable = {k: v.cpu() if v is not None else None
                    for k, v in z_buffers.items()}
    torch.save(serialisable, path)
    print(f"[visualize_z] z buffers saved to {path}")


def load_z_buffers(path: str = "z_buffers.pt") -> Dict[str, Optional[torch.Tensor]]:
    """Load previously saved z buffers."""
    return torch.load(path, map_location="cpu")


# ── Convenience: cluster count over time bar chart ────────────────────────────

def plot_cluster_counts(
    cluster_log: list[tuple[str, int]],
    save_path: str = "imgs/cluster_counts.pdf",
    dpi: int = 150,
) -> None:
    """
    Bar chart of cluster count after each task.

    Parameters
    ----------
    cluster_log : list of (task_name, n_clusters_after) tuples
                  e.g. [("balance", 1), ("navigation", 2), ...]
    """
    names   = [t for t, _ in cluster_log]
    counts  = [c for _, c in cluster_log]
    colours = [TASK_COLOURS[i % len(TASK_COLOURS)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.2), 3.5))
    bars = ax.bar(names, counts, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("DPMM cluster count", fontsize=9)
    ax.set_xlabel("Task (in sequence order)", fontsize=9)
    ax.set_title("DPMM cluster count after each task", fontsize=10)
    ax.set_yticks(range(0, max(counts) + 2))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(count), ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[visualize_z] Cluster count chart saved to {save_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise task encoder z buffers")
    parser.add_argument("--buffers", default="z_buffers.pt",
                        help="Path to saved z_buffers.pt file")
    parser.add_argument("--out",     default="imgs/z_visualization.pdf",
                        help="Output figure path")
    parser.add_argument("--method",  default="both",
                        choices=["umap", "tsne", "both"])
    parser.add_argument("--max",     type=int, default=5000,
                        help="Max samples per task")
    args = parser.parse_args()

    z_buffers = load_z_buffers(args.buffers)
    print(f"Loaded z buffers for tasks: {list(z_buffers.keys())}")
    visualize_z_buffers(z_buffers, save_path=args.out,
                        method=args.method, max_per_task=args.max)