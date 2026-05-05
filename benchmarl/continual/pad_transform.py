import torch.nn.functional as F
from torchrl.envs.transforms import Transform


class PadObservation(Transform):
    """
    Pads the last dimension of observations to target_dim.
    Works for VMAS multi-agent observations like:
    ('agents', 'observation')
    """

    def __init__(self, target_dim, in_key=("agents", "observation")):
        super().__init__()
        self.target_dim = target_dim
        self.in_key = in_key

    def _call(self, tensordict):

        obs = tensordict.get(self.in_key)

        pad = self.target_dim - obs.shape[-1]

        if pad > 0:
            obs = F.pad(obs, (0, pad))

        tensordict.set(self.in_key, obs)

        return tensordict

    def transform_observation_spec(self, observation_spec):

        shape = observation_spec.shape
        new_shape = (*shape[:-1], self.target_dim)

        observation_spec.shape = new_shape

        return observation_spec