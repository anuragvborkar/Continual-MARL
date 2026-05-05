"""
Yaml config loader for PaddedVmasTask.

PaddedVmasTask.get_from_yaml() uses Task._load_from_yaml() which looks in
benchmarl/conf/task/<env_name>/<task_name>.yaml.

Since PaddedVmasTask.associated_class() returns PaddedVmasClass, and
PaddedVmasClass.env_name() returns "vmas", it will look in:
  benchmarl/conf/task/vmas/<task_name>.yaml

Those files already exist in BenchMARL — so get_from_yaml() works out of the box.

However, if you want to customise n_agents or other params, you can also call:
  PaddedVmasTask.NAVIGATION.get_task(config={...})
directly.

This file provides convenience functions for doing that cleanly.
"""

from legion_marl.tasks.padded_vmas import PaddedVmasTask
from legion_marl.tasks.task_config import TASK_CONFIGS


def make_padded_task(task_name: str, config_overrides: dict = None):
    """
    Create a PaddedVmasTask instance for the given task name.

    Args:
        task_name: e.g. "navigation", "balance"
        config_overrides: optional dict to override yaml defaults, e.g. {"n_agents": 4}

    Returns:
        A TaskClass instance (PaddedVmasClass) ready to pass to Experiment.
    """
    task_enum_name = task_name.upper()
    task_enum = getattr(PaddedVmasTask, task_enum_name)

    if config_overrides:
        task = task_enum.get_from_yaml()
        task.config.update(config_overrides)
        return task
    else:
        return task_enum.get_from_yaml()
