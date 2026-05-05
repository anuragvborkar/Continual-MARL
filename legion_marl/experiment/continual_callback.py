"""
ContinualCallback: BenchMARL Callback for continual learning.

Phase 1 (use_oracle=True):
  - Loads shared weights into freshly built model on setup
  - Moves loss buffers to correct device (TorchRL 0.11 workaround)

Phase 2 (use_oracle=False):
  - All Phase 1 responsibilities
  - Maintains a SEPARATE optimizer for the task encoder (independent of PPO)
  - Every dpmm_update_freq optimizer steps: runs DPMM M-step
  - Every kl_update_freq optimizer steps: steps the encoder optimizer with KL loss
  - The encoder optimizer is completely decoupled from PPO's gradient tape
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from legion_marl.models.dpmm import DPMM
    from legion_marl.models.task_conditioned_mlp import TaskConditionedMlp


class ContinualCallback(Callback):
    """
    Args:
        task_name:           name of the current task
        task_id:             0-based index in the task sequence
        n_tasks:             total tasks in the sequence
        z_dim:               latent dimension
        use_oracle:          Phase 1 = True, Phase 2+ = False
        shared_model_state:  state_dict from previous task (None for first task)
        dpmm:                shared DPMM instance owned by ContinualRunner
        z_buffer_max:        max z sample chunks in rolling buffer
        kl_update_freq:      optimizer steps between KL encoder updates
        dpmm_update_freq:    optimizer steps between DPMM M-steps
        dpmm_warmup_steps:   optimizer steps before DPMM activates
        kl_warmup_steps:     optimizer steps before KL updates begin.
                             Must be > dpmm_warmup_steps so the DPMM has run
                             several M-steps before the encoder is pulled toward
                             cluster means.  Default: dpmm_warmup_steps + 500.
        kl_coef:             weight of KL loss
        encoder_lr:          learning rate for separate encoder optimizer
    """

    def __init__(
        self,
        task_name: str,
        task_id: int,
        n_tasks: int,
        z_dim: int = 32,
        use_oracle: bool = True,
        shared_model_state: Optional[Dict] = None,
        dpmm: Optional["DPMM"] = None,
        z_buffer_max: int = 10000,
        kl_update_freq: int = 50,
        dpmm_update_freq: int = 100,
        dpmm_warmup_steps: int = 2000,
        kl_warmup_steps: Optional[int] = None,
        # kl_coef is intentionally small: the PPO reward signal must be free to
        # push z apart between tasks before KL can sharpen cluster separation.
        # Too large a kl_coef with 1 cluster collapses all tasks to the same z.
        kl_coef: float = 0.01,
        encoder_lr: float = 3e-4,
    ):
        super().__init__()
        self.task_name = task_name
        self.task_id = task_id
        self.n_tasks = n_tasks
        self.z_dim = z_dim
        self.use_oracle = use_oracle
        self.shared_model_state = shared_model_state
        self.dpmm = dpmm
        self.z_buffer_max = z_buffer_max
        self.kl_update_freq = kl_update_freq
        self.dpmm_update_freq = dpmm_update_freq
        self.dpmm_warmup_steps = dpmm_warmup_steps
        # KL updates must start after the DPMM has run a few M-steps so that
        # cluster means are meaningful.  Default: dpmm_warmup_steps + 5 M-step
        # intervals, giving the DPMM time to settle before pulling the encoder.
        self.kl_warmup_steps = (
            kl_warmup_steps
            if kl_warmup_steps is not None
            else dpmm_warmup_steps + 5 * dpmm_update_freq
        )
        self.kl_coef = kl_coef
        self.encoder_lr = encoder_lr

        # Separate encoder optimizer (created in on_setup)
        self._encoder_optimizer: Optional[torch.optim.Optimizer] = None

        # Rolling z buffer for DPMM M-step
        self.z_buffer: List[torch.Tensor] = []
        self._step = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def on_setup(self):
        # Transfer weights from previous task
        if self.shared_model_state is not None:
            self._load_shared_weights()

        # TorchRL 0.11 bug: clip_epsilon buffer lands on CPU
        try:
            device = self.experiment.config.train_device
            for loss in self.experiment.losses.values():
                loss.to(device)
        except Exception as e:
            print(f"[ContinualCallback] Warning: could not move losses to device: {e}")

        # Create a separate Adam optimizer just for the task encoder
        if not self.use_oracle:
            model = self._get_policy_model()
            if model is not None:
                self._encoder_optimizer = torch.optim.Adam(
                    model.task_encoder.parameters(),
                    lr=self.encoder_lr,
                )

    # ------------------------------------------------------------------
    # Per-optimizer-step hook
    # ------------------------------------------------------------------

    def on_train_step(
        self, batch: TensorDictBase, group: str
    ) -> Optional[TensorDictBase]:
        self._step += 1

        if self.use_oracle or self.dpmm is None:
            return None

        model = self._get_policy_model()
        if model is None or model.last_z is None:
            return None

        z = model.last_z  # (N, z_dim) detached, on train device

        # --- Reconstruction loss update (fires every step, no warmup needed) ---
        # This is the primary gradient signal for the encoder. It forces z to
        # encode information about the input obs, making it task-discriminative
        # BEFORE the DPMM has any clusters. Without this, the encoder stays at
        # random init and z is identical across tasks — the DPMM never clusters.
        if self._step % self.kl_update_freq == 0 and self._encoder_optimizer is not None:
            try:
                obs = batch.get((group, "observation"))
                leading_shape = obs.shape[:-2]
                N = 1
                for d in leading_shape:
                    N *= d
                obs_flat = obs.reshape(N, model.n_agents, obs.shape[-1])
                joint_obs = obs_flat.mean(dim=1)  # (N, obs_dim)

                was_training = model.task_encoder.training
                model.task_encoder.train()
                self._encoder_optimizer.zero_grad()
                recon_loss = model.task_encoder.reconstruction_loss(joint_obs)

                if torch.isfinite(recon_loss):
                    recon_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.task_encoder.parameters(), max_norm=1.0
                    )
                    self._encoder_optimizer.step()

                    if self._step % (self.dpmm_update_freq * 5) == 0:
                        print(
                            f"[Encoder] step={self._step} | "
                            f"recon_loss={recon_loss.item():.4f} | "
                            f"n_clusters={self.dpmm.n_clusters}"
                        )
                model.task_encoder.train(was_training)
            except Exception as e:
                print(f"[ContinualCallback] Recon update failed at step {self._step}: {e}")

        # --- Accumulate z for DPMM M-step ---
        self.z_buffer.append(z)
        max_chunks = self.z_buffer_max // max(z.shape[0], 1) + 1
        if len(self.z_buffer) > max_chunks:
            self.z_buffer.pop(0)

        # Skip everything until warmup is done
        if self._step < self.dpmm_warmup_steps:
            return None

        # --- DPMM M-step ---
        if self._step % self.dpmm_update_freq == 0:
            z_all = torch.cat(self.z_buffer, dim=0).to(self.dpmm.device)
            self.dpmm.update(z_all)

        # --- KL encoder update ---
        # The encoder is pulled toward DPMM cluster centres via MSE-based KL loss.
        # IMPORTANT: we only apply KL when there are >=2 clusters. With a single
        # cluster, KL would pull ALL tasks' z toward the same centre, actively
        # preventing the encoder from learning to separate tasks. The PPO reward
        # signal is what first drives z to separate; KL then sharpens that
        # separation once a second cluster has already spawned.
        if (self._step >= self.kl_warmup_steps
                and self._step % self.kl_update_freq == 0
                and len(self.dpmm.clusters) >= 2
                and self._encoder_optimizer is not None):
            try:
                obs = batch.get((group, "observation"))
                leading_shape = obs.shape[:-2]
                N = 1
                for d in leading_shape:
                    N *= d
                obs_flat = obs.reshape(N, model.n_agents, obs.shape[-1])
                joint_obs = obs_flat.mean(dim=1)  # (N, obs_dim)

                # Restore encoder training mode; save prior mode for safety
                was_training = model.task_encoder.training
                model.task_encoder.train()

                # Fresh forward pass so gradients flow through encoder weights.
                # deterministic=True avoids stochastic reparameterisation noise
                # in the KL gradient, which can destabilise the encoder.
                self._encoder_optimizer.zero_grad()
                z_grad = model.task_encoder(joint_obs, deterministic=True)
                z_grad_dpmm = z_grad.to(self.dpmm.device)
                kl = self.dpmm.kl_loss(z_grad_dpmm)
                kl_scaled = self.kl_coef * kl.to(joint_obs.device)

                if torch.isfinite(kl_scaled) and torch.isfinite(z_grad).all():
                    kl_scaled.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.task_encoder.parameters(), max_norm=1.0
                    )
                    self._encoder_optimizer.step()

                    # Periodic diagnostic log (every 5 DPMM M-step intervals)
                    if self._step % (self.dpmm_update_freq * 5) == 0:
                        policy_norm = sum(p.norm().item() for p in model.policy_mlp.parameters())
                        encoder_norm = sum(p.norm().item() for p in model.task_encoder.parameters())
                        print(
                            f"[DPMM] step={self._step} | "
                            f"n_clusters={self.dpmm.n_clusters} | "
                            f"kl_loss={kl.item():.4f} | "
                            f"policy_norm={policy_norm:.2f} | "
                            f"encoder_norm={encoder_norm:.2f}"
                        )
                else:
                    # NaN in KL: skip the step but log once so it's visible
                    print(
                        f"[ContinualCallback] step={self._step}: KL loss is "
                        f"non-finite ({kl.item():.4f}), skipping encoder update."
                    )

                # Restore encoder mode
                model.task_encoder.train(was_training)

            except Exception as e:
                print(f"[ContinualCallback] KL update failed at step {self._step}: {e}")

        return None

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        pass

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def on_state_dict(self, state_dict: Dict[str, Any]):
        model = self._get_policy_model()
        if model is not None:
            state_dict["shared_model_state"] = model.state_dict()

    def on_load_state_dict(self, state_dict: Dict[str, Any]):
        if "shared_model_state" in state_dict:
            self.shared_model_state = state_dict["shared_model_state"]

    # ------------------------------------------------------------------
    # Weight extraction
    # ------------------------------------------------------------------

    def extract_shared_weights(self) -> Optional[Dict]:
        model = self._get_policy_model()
        return model.state_dict() if model is not None else None

    def extract_z_buffer(self) -> Optional[torch.Tensor]:
        if not self.z_buffer:
            return None
        return torch.cat(self.z_buffer, dim=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_shared_weights(self):
        model = self._get_policy_model()
        if model is None:
            print("[ContinualCallback] Warning: model not found for weight loading.")
            return
        missing, unexpected = model.load_state_dict(
            self.shared_model_state, strict=False
        )
        if missing:
            print(f"[ContinualCallback] Missing keys: {missing}")
        if unexpected:
            print(f"[ContinualCallback] Unexpected keys: {unexpected}")

    def _get_policy_model(self) -> Optional["TaskConditionedMlp"]:
        if self.experiment is None:
            return None
        try:
            from legion_marl.models.task_conditioned_mlp import TaskConditionedMlp
            for module in self.experiment.policy.modules():
                if isinstance(module, TaskConditionedMlp):
                    return module
        except Exception as e:
            print(f"[ContinualCallback] Could not find model: {e}")
        return None