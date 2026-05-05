import torch


def navigation_success_from_rollout(td):
    """
    Success if all agents reached their goal at any timestep.
    """

    done = td["next", "done"]

    return done.any().item()

def transport_success_from_rollout(td, threshold=0.1):

    box_pos = td["next", "agents", "info", "box_pos"]
    goal_pos = td["next", "agents", "info", "goal_pos"]

    dist = torch.norm(box_pos - goal_pos, dim=-1)

    return (dist < threshold).any().item()

def flocking_success_from_rollout(td, threshold=0.7):

    vel_rew = td["next", "agents", "info", "vel_rew"]

    mean_rew = vel_rew.mean()

    return mean_rew > threshold

def compute_success(task_name, td):
    
    if "navigation" in task_name:
        return navigation_success_from_rollout(td)

    if "transport" in task_name:
        return transport_success_from_rollout(td)

    if "flocking" in task_name:
        return flocking_success_from_rollout(td)