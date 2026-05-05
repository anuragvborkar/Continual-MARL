"""
Task configuration constants for the LEGION-MARL project.

GLOBAL_OBS_DIM: the padded observation dimension all tasks are mapped to.
MAX_AGENTS: the maximum number of agents across all tasks (used for task encoder
            mean-pooling — not for padding agent count).

All observations are zero-padded to GLOBAL_OBS_DIM at the tail.
The first 4 dims (pos, vel) are semantically aligned across all tasks natively.
"""

GLOBAL_OBS_DIM = 19   # discovery has the largest obs (19), all others are padded up to this
MAX_AGENTS = 5        # discovery has the most agents; used as documentation, not hard constraint
ACTION_DIM = 2        # uniform across all tasks (continuous 2D force)

# Per-task ground-truth metadata (used for oracle z in Phase 1 and ARI eval in Phase 2+)
TASK_CONFIGS = {
    "navigation":        {"n_agents": 4, "obs_dim": 18, "task_id": 0, "reward_scale": 1.0},
    "balance":           {"n_agents": 3, "obs_dim": 16, "task_id": 1, "reward_scale": 0.02},
    "flocking":          {"n_agents": 4, "obs_dim": 18, "task_id": 2, "reward_scale": 1.0},
    "transport":         {"n_agents": 4, "obs_dim": 11, "task_id": 3, "reward_scale": 0.1},
    "reverse_transport": {"n_agents": 4, "obs_dim": 10, "task_id": 4, "reward_scale": 0.1},
    "discovery":         {"n_agents": 5, "obs_dim": 19, "task_id": 5, "reward_scale": 1.0},
}

# Ordered task sequences for each phase
# PHASE_1_SEQUENCE = ["navigation", "balance"]
PHASE_1_SEQUENCE = ["balance","navigation"]
# PHASE_3_SEQUENCE = ["navigation", "balance", "flocking"]
PHASE_2_SEQUENCE = ["balance", "navigation", "flocking", "reverse_transport", "discovery"]
# PHASE_1_SEQUENCE = ["balance","navigation", "balance","navigation"]