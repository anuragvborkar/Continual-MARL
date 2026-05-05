import json
from hydra import initialize, compose

from benchmarl.hydra_config import load_experiment_from_hydra
from continual.tasks import TASK_SEQUENCE
from continual.pad_transform import PadObservation

MAX_OBS_DIM = 32


def create_experiment(task, algorithm="mappo"):

    with initialize(version_base=None, config_path="conf"):

        cfg = compose(
            config_name="config_vmas",
            overrides=[
                f"algorithm={algorithm}",
                f"task={task}",
                "experiment.max_n_frames=100000",
            ],
        )

    experiment = load_experiment_from_hydra(cfg, task_name=task)

    # -------------------------------------------------------
    # Inject padding transform into the task env transforms
    # -------------------------------------------------------

    original_get_transforms = experiment.task.get_env_transforms

    def patched_get_env_transforms(env):

        transforms = original_get_transforms(env)

        # prepend padding transform
        return [PadObservation(MAX_OBS_DIM)] + transforms

    experiment.task.get_env_transforms = patched_get_env_transforms

    return experiment


def run_continual_experiment(algorithm="mappo"):

    results = {}
    previous_policy_state = None

    for task_idx, task in enumerate(TASK_SEQUENCE):

        print(f"\n===== Training on {task} =====")

        experiment = create_experiment(task, algorithm)

        if task_idx > 0:
            experiment.policy.load_state_dict(previous_policy_state, strict=False)

        experiment.run()

        previous_policy_state = experiment.policy.state_dict()

        results[f"after_task_{task_idx+1}"] = {}

        for eval_task in TASK_SEQUENCE[: task_idx + 1]:

            eval_exp = create_experiment(eval_task, algorithm)

            eval_exp.policy.load_state_dict(previous_policy_state)

            success = eval_exp.evaluate()

            results[f"after_task_{task_idx+1}"][eval_task] = success

        with open("continual_results.json", "w") as f:
            json.dump(results, f, indent=2)

        import wandb
        wandb.finish()

    return results