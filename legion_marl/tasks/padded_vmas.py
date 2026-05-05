"""
Padded VMAS task classes that override observation_spec and inject a
zero-padding transform so all tasks present GLOBAL_OBS_DIM-dimensional
observations to the model, regardless of their native obs size.

Usage in place of VmasTask:
    from legion_marl.tasks.padded_vmas import PaddedVmasTask
    task = PaddedVmasTask.NAVIGATION.get_from_yaml()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, Transform
from torchrl.envs.libs.vmas import VmasEnv

from benchmarl.environments.common import Task
from benchmarl.environments.vmas.common import VmasClass
from torchrl.envs.transforms import RewardScaling
from benchmarl.utils import DEVICE_TYPING

from legion_marl.tasks.task_config import GLOBAL_OBS_DIM


class ObsPaddingTransform(Transform):
    """
    Zero-pads the 'observation' key of every agent group to GLOBAL_OBS_DIM
    along the last dimension.  Applied both at env-step time and at reset.

    This transform is non-destructive: if the observation is already at
    GLOBAL_OBS_DIM it is returned unchanged.
    """

    def __init__(self, target_dim: int = GLOBAL_OBS_DIM):
        # We register no in_keys/out_keys here because the group names are
        # dynamic.  We handle everything in _call / _reset ourselves.
        super().__init__()
        self.target_dim = target_dim

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _pad_tensordict(self, tensordict: TensorDictBase) -> TensorDictBase:
        for key in list(tensordict.keys()):
            sub = tensordict.get(key)
            # Recurse into nested TensorDicts (e.g. the "agents" group)
            if hasattr(sub, "keys"):
                self._pad_tensordict(sub)
            # Pad the 'observation' leaf tensor
            if key == "observation" and isinstance(sub, torch.Tensor):
                native_dim = sub.shape[-1]
                if native_dim < self.target_dim:
                    pad_size = self.target_dim - native_dim
                    padded = torch.nn.functional.pad(sub, (0, pad_size))
                    tensordict.set(key, padded)
        return tensordict

    # ------------------------------------------------------------------
    # Transform hooks required by TorchRL
    # ------------------------------------------------------------------
    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._pad_tensordict(tensordict)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._pad_tensordict(tensordict_reset)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        """
        Update the spec so downstream components (model, algorithm) see the
        padded shape and not the native one.
        """
        for key in list(observation_spec.keys(True, True)):
            # key is a tuple like ("agents", "observation") when iterating nested
            leaf = observation_spec[key]
            if isinstance(key, tuple) and key[-1] == "observation":
                native_dim = leaf.shape[-1]
                if native_dim < self.target_dim:
                    new_shape = torch.Size(list(leaf.shape[:-1]) + [self.target_dim])
                    observation_spec[key] = Unbounded(
                        shape=new_shape,
                        device=leaf.device,
                        dtype=leaf.dtype,
                    )
            elif key == "observation":
                native_dim = leaf.shape[-1]
                if native_dim < self.target_dim:
                    new_shape = torch.Size(list(leaf.shape[:-1]) + [self.target_dim])
                    observation_spec[key] = Unbounded(
                        shape=new_shape,
                        device=leaf.device,
                        dtype=leaf.dtype,
                    )
        return observation_spec


class PaddedVmasClass(VmasClass):
    """
    VmasClass subclass that injects ObsPaddingTransform and RewardScaling
    so all observations are GLOBAL_OBS_DIM-dimensional and rewards are
    normalised to a consistent scale across tasks.
    """

    # Set by PaddedVmasTask subclasses via task_name lookup
    reward_scale: float = 1.0

    def get_env_transforms(self, env: EnvBase):
        from legion_marl.tasks.task_config import TASK_CONFIGS
        base = super().get_env_transforms(env)
        # Look up reward scale from task name
        task_name = self.__class__.__name__.lower().replace("task", "")
        # Fall back to instance attribute set by PaddedVmasTask
        scale = getattr(self, "reward_scale", 1.0)
        transforms = base + [ObsPaddingTransform(GLOBAL_OBS_DIM)]
        if scale != 1.0:
            transforms += [RewardScaling(
                loc=0.0, scale=scale,
                in_keys=[("agents", "reward")],
                out_keys=[("agents", "reward")],
            )]
        return transforms

    def observation_spec(self, env: EnvBase) -> Composite:
        # Get native spec, then apply the padding transform spec update
        spec = super().observation_spec(env)
        transform = ObsPaddingTransform(GLOBAL_OBS_DIM)
        return transform.transform_observation_spec(spec)


class PaddedVmasTask(Task):
    """
    Drop-in replacement for VmasTask with padded observations.

    All members mirror VmasTask.  Add new tasks here as needed.
    """

    BALANCE = None
    NAVIGATION = None
    FLOCKING = None
    TRANSPORT = None
    REVERSE_TRANSPORT = None
    DISCOVERY = None

    @staticmethod
    def associated_class():
        return PaddedVmasClass