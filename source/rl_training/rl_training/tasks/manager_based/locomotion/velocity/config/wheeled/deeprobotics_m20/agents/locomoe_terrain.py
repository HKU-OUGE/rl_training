# source/.../deeprobotics_m20/agents/locomoe_terrain.py
"""LocoMoE policy / PPO / config.

Re-implementation of the MoE-Loco architecture (Huang et al., "MoE-Loco:
Mixture of Experts for Multitask Locomotion", IROS 2025, arXiv:2503.08564)
as a comparison baseline for SplitMoE on the Deeprobotics M20.

Key architectural facts (paper):
  * N_exp = 6 experts (paper L499: "We select expert number N_exp as 6").
  * Single shared gate between actor and critic:
        g(h_t) -> softmax -> N-dim weights.
  * Each expert is an MLP [256, 128, 128] outputting the FULL action vector
    (paper Table VI, "Expert Head").
  * Gating MLP [128] (paper Table VI, "Gating Network").
  * Action = weighted sum  a_t = sum_i g_hat_i * f_i(h_t)   (paper eq. 3).
  * h_t is the output of a low-level recurrent module over the dual-state
    representation. Paper uses LSTM; we use GRU here to match SplitMoE so the
    head-to-head comparison only varies the MoE head.

Training protocol (per user-confirmed spec):
  * Single-stage PPO (no two-stage student/oracle, no L_recon).
  * Loss = L_surrogate + L_value (i.e. rsl_rl's vanilla PPO).
  * Variable-isolated comparison: shares the same env / observation /
    encoder/AE/VAE as SplitMoE; only the MoE head + gating differ.

Constructor accepts both calling conventions:
  1. ``LocoMoEActorCritic(obs, obs_groups, num_actions, **policy_cfg)`` -
     called by rsl_rl's OnPolicyRunner; mirrors SplitMoEActorCritic so the
     encoder (proprio + ElevationAE + ScanAE + ProprioVAE -> concat -> GRU)
     is identical to SplitMoE's training path.
  2. ``LocoMoEActorCritic(num_actor_obs=N, num_critic_obs=M, num_actions=A,
     num_experts=6, ...)`` - standalone smoke-test form. Treats the input as
     a single flat proprioception vector and skips all AE/VAE branches.
"""

from __future__ import annotations

from dataclasses import field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import unpad_trajectories

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

# Reuse the AE / VAE / MLP building blocks already validated for SplitMoE so
# both baselines share an identical perception stack.
from .moe_terrain import (
    MLP,
    ElevationAE,
    EmpiricalNormalization,
    MultiLayerScanAE,
    ProprioVAE,
    orthogonal_init,
)


# ==============================================================================
# 1. LocoMoE Actor-Critic
# ==============================================================================


class LocoMoEActorCritic(ActorCritic):
    """MoE-Loco actor-critic with a single shared gate over N=6 full-action experts."""

    is_recurrent = True

    def __init__(
        self,
        obs=None,
        obs_groups=None,
        num_actions: int | None = None,
        actor_hidden_dims=(256, 128, 128),
        critic_hidden_dims=(256, 128, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        num_experts: int = 6,
        latent_dim: int = 256,
        rnn_type: str = "gru",
        gating_hidden_dim: int = 128,
        # Encoder switches (mirror SplitMoE so the perception path is identical).
        blind_vision: bool = True,
        use_elevation_ae: bool = False,
        elevation_dim: int = 187,
        use_multilayer_scan: bool = False,
        num_scan_channels: int = 12,
        num_scan_rays: int = 21,
        feed_estimator_to_policy: bool = False,
        feed_ae_to_policy: bool = False,
        # Standalone / smoke-test calling form.
        num_actor_obs: int | None = None,
        num_critic_obs: int | None = None,
        **kwargs,
    ):
        # ---------------------------------------------------------------
        # Detect which calling convention was used.
        # ---------------------------------------------------------------
        self._standalone_mode = num_actor_obs is not None

        if self._standalone_mode:
            # Smoke-test / standalone form: skip ActorCritic.__init__ (it
            # requires the dict obs+obs_groups). Initialize nn.Module manually.
            nn.Module.__init__(self)
            assert num_actions is not None, "num_actions must be specified"
            if num_critic_obs is None:
                num_critic_obs = num_actor_obs
            self.obs_groups = obs_groups  # may be None
            self.input_keys = None
            self.critic_keys = None
            self.proprio_dim = num_actor_obs
            self.critic_proprio_dim = num_critic_obs
            self.has_estimator_group = False
            self.has_elevation_input = False
            # Action noise parameter (one shared per-action std as per spec).
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            self.noise_std_type = "scalar"
            # Skip optional normalizers and AEs in standalone mode.
            self.actor_obs_normalization = False
            self.critic_obs_normalization = False
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()
            self.estimator = None
            self.estimator_obs_normalizer = None
            self.estimator_obs_normalization = False
            self.estimator_output_dim = 0
            self.estimator_input_indices = []
            self.estimator_target_indices = []
            self.use_elevation_ae = False
            self.use_multilayer_scan = False
            self.use_cnn = False
            self.elevation_dim = elevation_dim
            self.scan_dim = 0
            self.ae_output_dim = 0
            self.blind_vision = True
            self.feed_estimator = False
            self.feed_ae = False
            self.vae_feature_dim = 0
            rnn_input_dim = num_actor_obs
            critic_rnn_input_dim = num_critic_obs
        else:
            # OnPolicyRunner form: obs is a TensorDict, obs_groups is a dict.
            base_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "estimator_output_dim",
                    "estimator_hidden_dims",
                    "estimator_input_indices",
                    "estimator_target_indices",
                    "estimator_obs_normalization",
                    "actor_obs_normalization",
                    "critic_obs_normalization",
                    "init_noise_legs",
                    "init_noise_wheels",
                    "num_leg_actions",
                    "num_wheel_experts",
                    "num_leg_experts",
                ]
            }
            super().__init__(
                obs,
                obs_groups,
                num_actions,
                actor_hidden_dims=list(actor_hidden_dims),
                critic_hidden_dims=list(critic_hidden_dims),
                activation=activation,
                init_noise_std=init_noise_std,
                **base_kwargs,
            )

            # Resolve observation dimensions (mirroring SplitMoEActorCritic).
            if isinstance(obs, dict) or hasattr(obs, "keys"):
                self.input_keys = obs_groups.get("policy", None)
                if self.input_keys:
                    num_obs = sum(obs[k].shape[-1] for k in self.input_keys)
                else:
                    num_obs = list(obs.values())[0].shape[-1]
                self.critic_keys = obs_groups.get("critic", self.input_keys)
                critic_num_obs = sum(obs[k].shape[-1] for k in self.critic_keys)
            else:
                self.input_keys = None
                self.critic_keys = None
                num_obs = obs.shape[-1]
                critic_num_obs = num_obs

            # Encoder switches.
            self.blind_vision = blind_vision
            self.use_elevation_ae = use_elevation_ae
            self.elevation_dim = elevation_dim
            self.use_multilayer_scan = use_multilayer_scan
            self.num_scan_channels = num_scan_channels
            self.num_scan_rays = num_scan_rays
            self.scan_dim = num_scan_channels * num_scan_rays
            self.use_cnn = False
            self.feed_estimator = feed_estimator_to_policy
            self.feed_ae = feed_ae_to_policy

            # Determine proprio dim (env-tail is everything beyond proprio).
            self.has_elevation_input = False
            if self.use_elevation_ae and num_obs > self.elevation_dim:
                self.proprio_dim = num_obs - self.elevation_dim
                self.has_elevation_input = True
            else:
                self.proprio_dim = num_obs

            if self.use_elevation_ae and critic_num_obs > self.elevation_dim:
                self.critic_proprio_dim = critic_num_obs - self.elevation_dim
            else:
                self.critic_proprio_dim = critic_num_obs

            # Optional ProprioVAE estimator.
            self.estimator_output_dim = kwargs.get("estimator_output_dim", 0)
            self.estimator_hidden_dims = kwargs.get("estimator_hidden_dims", [128, 64])
            self.estimator_obs_normalization = kwargs.get("estimator_obs_normalization", True)
            self.estimator_target_indices = kwargs.get("estimator_target_indices", [0, 1, 2])
            self.estimator_input_indices = kwargs.get(
                "estimator_input_indices", list(range(3, 32))
            )
            self.has_estimator_group = False
            est_input_dim = 0
            if self.estimator_output_dim > 0:
                try:
                    if obs_groups is not None and "estimator" not in obs_groups:
                        raise KeyError("estimator omitted from obs_groups")
                    est_group = obs["estimator"]
                    est_input_dim = est_group.shape[-1]
                    self.has_estimator_group = True
                except (KeyError, TypeError, AttributeError):
                    self.has_estimator_group = False
                    est_input_dim = len(self.estimator_input_indices)

                if est_input_dim > 0:
                    self.estimator = ProprioVAE(
                        input_dim=est_input_dim,
                        vel_dim=self.estimator_output_dim,
                        latent_dim=64,
                        hidden_dims=self.estimator_hidden_dims,
                    )
                    self.estimator_obs_normalizer = None
                    if self.estimator_obs_normalization:
                        self.estimator_obs_normalizer = EmpiricalNormalization(
                            shape=[est_input_dim], until_step=1.0e9
                        )
                    self.vae_feature_dim = self.estimator_output_dim + 64
                else:
                    self.estimator = None
                    self.vae_feature_dim = 0
            else:
                self.estimator = None
                self.vae_feature_dim = 0

            # Environment-feature autoencoders.
            self.ae_output_dim = 0
            if self.use_elevation_ae:
                self.elev_out_dim = 64
                self.ae_output_dim += self.elev_out_dim
                self.elevation_encoder = ElevationAE(
                    input_dim=self.elevation_dim, output_dim=self.elev_out_dim
                )
            if self.use_multilayer_scan:
                self.scan_out_dim = 64
                self.ae_output_dim += self.scan_out_dim
                self.scan_encoder = MultiLayerScanAE(
                    num_channels=self.num_scan_channels,
                    num_rays=self.num_scan_rays,
                    output_dim=self.scan_out_dim,
                )

            # Per-stream observation normalizers (proprio only; AEs handle env).
            self.actor_obs_normalization = kwargs.get("actor_obs_normalization", True)
            self.critic_obs_normalization = kwargs.get("critic_obs_normalization", True)
            if self.actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(
                    shape=[self.proprio_dim], until_step=1.0e9
                )
            if self.critic_obs_normalization:
                self.critic_obs_normalizer = EmpiricalNormalization(
                    shape=[self.critic_proprio_dim], until_step=1.0e9
                )

            # Compute RNN input dims (proprio + optional VAE + optional AE/env).
            rnn_input_dim = self.proprio_dim
            critic_rnn_input_dim = self.critic_proprio_dim
            if self.has_elevation_input:
                rnn_input_dim += (
                    self.ae_output_dim if self.feed_ae else (num_obs - self.proprio_dim)
                )
                critic_rnn_input_dim += (
                    self.ae_output_dim
                    if self.feed_ae
                    else (critic_num_obs - self.critic_proprio_dim)
                )
            elif self.feed_ae:
                rnn_input_dim += self.ae_output_dim
                critic_rnn_input_dim += self.ae_output_dim
            if self.feed_estimator and self.estimator is not None:
                rnn_input_dim += self.vae_feature_dim
                critic_rnn_input_dim += self.vae_feature_dim

            # Re-init action std as one shared per-action parameter (paper: a
            # single Gaussian over the full action vector, no leg/wheel split).
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

            print(
                f"[LocoMoE] RNN Actor Input: {rnn_input_dim} (Proprio: {self.proprio_dim}, "
                f"VAE appended: {self.vae_feature_dim if self.feed_estimator else 0}, "
                f"Env appended: "
                f"{self.ae_output_dim if self.feed_ae else (num_obs - self.proprio_dim if self.has_elevation_input else 0)}, "
                f"Blind: {self.blind_vision})"
            )

        # ---------------------------------------------------------------
        # Shared sub-modules: RNN -> shared gate -> N full-action experts.
        # ---------------------------------------------------------------
        self.latent_dim = latent_dim
        self.rnn_type = rnn_type.lower()
        self.num_experts = num_experts
        self.num_actions = num_actions

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim, hidden_size=self.latent_dim, batch_first=False
            )
            self.critic_rnn = nn.LSTM(
                input_size=critic_rnn_input_dim,
                hidden_size=self.latent_dim,
                batch_first=False,
            )
        else:
            self.rnn = nn.GRU(
                input_size=rnn_input_dim, hidden_size=self.latent_dim, batch_first=False
            )
            self.critic_rnn = nn.GRU(
                input_size=critic_rnn_input_dim,
                hidden_size=self.latent_dim,
                batch_first=False,
            )
        for rnn_net in (self.rnn, self.critic_rnn):
            for name, param in rnn_net.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

        # Paper: shared gate between actor and critic.
        # MLP [128] -> N (paper Table VI, "Gating Network"). One ELU hidden layer.
        self.gate_input_norm = nn.LayerNorm(self.latent_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.latent_dim, gating_hidden_dim),
            nn.ELU(),
            nn.Linear(gating_hidden_dim, num_experts),
        )
        orthogonal_init(self.gate[0], gain=np.sqrt(2))
        orthogonal_init(self.gate[2], gain=0.01)

        # Paper Table VI: Expert Head is MLP [256, 128, 128] (hidden dims).
        # Each actor expert emits the full action; each critic expert emits 1.
        actor_hidden = list(actor_hidden_dims)
        critic_hidden = list(critic_hidden_dims)
        self.actor_experts = nn.ModuleList(
            [
                MLP(
                    self.latent_dim,
                    num_actions,
                    hidden_dims=actor_hidden,
                    activation=activation,
                    output_gain=0.01,
                )
                for _ in range(num_experts)
            ]
        )
        self.critic_experts = nn.ModuleList(
            [
                MLP(
                    self.latent_dim,
                    1,
                    hidden_dims=critic_hidden,
                    activation=activation,
                    output_gain=1.0,
                )
                for _ in range(num_experts)
            ]
        )

        # Running RNN hidden states (set lazily on first call when we don't
        # have a ref obs tensor to size them with; if we have one, init now).
        self.active_hidden_states = None
        self.active_critic_hidden_states = None
        if not self._standalone_mode and obs is not None:
            ref_tensor = obs[list(obs.keys())[0]] if isinstance(obs, dict) else obs
            if hasattr(ref_tensor, "shape") and ref_tensor.ndim >= 1:
                batch_size = ref_tensor.shape[0]
                device = ref_tensor.device
                self.active_hidden_states = self._init_rnn_state(batch_size, device)
                self.active_critic_hidden_states = self._init_rnn_state(batch_size, device)

        # Bookkeeping for logging (mirrors SplitMoE convention).
        self.latest_weights: dict = {}

    # ------------------------------------------------------------------
    # RNN state helpers (mirrors SplitMoE's behavior; GRU 1-layer / LSTM 1-layer).
    # ------------------------------------------------------------------

    def _init_rnn_state(self, batch_size, device):
        if self.rnn_type == "lstm":
            return (
                torch.zeros(1, batch_size, self.latent_dim, device=device),
                torch.zeros(1, batch_size, self.latent_dim, device=device),
            )
        return torch.zeros(1, batch_size, self.latent_dim, device=device)

    def _prepare_hidden_state(self, hidden, device):
        if hidden is None:
            return None
        if isinstance(hidden, (tuple, list)):
            if self.rnn_type == "lstm" and len(hidden) == 2:
                if isinstance(hidden[0], (tuple, list)):
                    hidden = hidden[0]
                return tuple(h.to(device).contiguous() for h in hidden)
            if len(hidden) == 2 and self.rnn_type != "lstm":
                hidden = hidden[0]
        return hidden.to(device).contiguous()

    def _extract_raw_obs(self, obs, key_list):
        if key_list is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in key_list if k in obs]
            if not tensors:
                return list(obs.values())[0]
            return torch.cat(tensors, dim=-1)
        return obs

    # ------------------------------------------------------------------
    # Observation preprocessing — proprio normalization, optional AE/VAE
    # ------------------------------------------------------------------

    def _get_estimator_input(self, obs_dict):
        if self.has_estimator_group and (
            isinstance(obs_dict, dict) or hasattr(obs_dict, "keys")
        ):
            if "estimator" in obs_dict:
                return obs_dict["estimator"]
        if hasattr(obs_dict, "keys") or isinstance(obs_dict, dict):
            full_obs = self._extract_raw_obs(obs_dict, self.input_keys)
        else:
            full_obs = obs_dict
        return full_obs[..., self.estimator_input_indices]

    def update_normalization(self, obs):
        if self._standalone_mode:
            return
        if self.actor_obs_normalization and isinstance(
            self.actor_obs_normalizer, EmpiricalNormalization
        ):
            x_raw = self._extract_raw_obs(obs, self.input_keys)
            self.actor_obs_normalizer.update(x_raw[..., : self.proprio_dim])
        if self.critic_obs_normalization and isinstance(
            self.critic_obs_normalizer, EmpiricalNormalization
        ):
            x_raw_c = self._extract_raw_obs(obs, self.critic_keys or self.input_keys)
            self.critic_obs_normalizer.update(x_raw_c[..., : self.critic_proprio_dim])
        if (
            self.estimator is not None
            and self.estimator_obs_normalization
            and self.estimator_obs_normalizer is not None
        ):
            est_input = self._get_estimator_input(obs)
            self.estimator_obs_normalizer.update(est_input)

    def _process_obs(self, x, obs_dict=None, normalizer=None, proprio_dim=None):
        """Build RNN input = [normalized proprio | optional VAE feats | optional AE/env feats]."""
        if proprio_dim is None:
            proprio_dim = self.proprio_dim
        proprio = x[..., :proprio_dim]
        if normalizer is not None and not isinstance(normalizer, nn.Identity):
            proprio = normalizer(proprio)

        env_feat = None
        if (self.use_elevation_ae or self.use_multilayer_scan) and self.feed_ae:
            if obs_dict is not None and (isinstance(obs_dict, dict) or hasattr(obs_dict, "keys")):
                if "noisy_elevation" in obs_dict:
                    env_raw_full = obs_dict["noisy_elevation"]
                elif "policy" in obs_dict:
                    env_raw_full = obs_dict["policy"][..., proprio_dim:]
                else:
                    env_raw_full = x[..., proprio_dim:]
            else:
                env_raw_full = x[..., proprio_dim:]

            feats = []
            idx = 0
            if self.use_elevation_ae:
                env_raw_elev = env_raw_full[..., idx : idx + self.elevation_dim]
                was = self.elevation_encoder.training
                self.elevation_encoder.eval()
                latent_elev, _ = self.elevation_encoder(env_raw_elev)
                self.elevation_encoder.train(was)
                feats.append(latent_elev.detach())
                idx += self.elevation_dim
            if self.use_multilayer_scan:
                env_raw_scan = env_raw_full[..., idx : idx + self.scan_dim]
                was = self.scan_encoder.training
                self.scan_encoder.eval()
                latent_scan, _ = self.scan_encoder(env_raw_scan)
                self.scan_encoder.train(was)
                feats.append(latent_scan.detach())
            env_feat = torch.cat(feats, dim=-1)
            if self.blind_vision:
                env_feat = torch.zeros_like(env_feat)
        elif x.shape[-1] > proprio_dim and not self._standalone_mode:
            env_feat = x[..., proprio_dim:]
            if self.blind_vision:
                env_feat = torch.zeros_like(env_feat)

        vae_feat = None
        if self.estimator is not None and obs_dict is not None and self.feed_estimator:
            est_input = self._get_estimator_input(obs_dict)
            if self.estimator_obs_normalization and self.estimator_obs_normalizer is not None:
                est_input = self.estimator_obs_normalizer(est_input)
            was = self.estimator.training
            self.estimator.eval()
            vel_pred, _, mu, _, _ = self.estimator(est_input)
            self.estimator.train(was)
            vae_feat = torch.cat([vel_pred.detach(), mu.detach()], dim=-1)

        components = [proprio]
        if vae_feat is not None:
            components.append(vae_feat)
        if env_feat is not None:
            components.append(env_feat)
        return torch.cat(components, dim=-1)

    # ------------------------------------------------------------------
    # RNN forward (sequence or single step, with optional padding masks).
    # ------------------------------------------------------------------

    def _run_rnn(self, rnn_module, x_in, hidden_states, masks):
        if hidden_states is None:
            batch = x_in.shape[1] if x_in.ndim == 3 else x_in.shape[0]
            hidden_states = self._init_rnn_state(batch, x_in.device)

        if x_in.ndim == 3:
            rnn_out, next_h = rnn_module(x_in, hidden_states)
            latent = unpad_trajectories(rnn_out, masks) if masks is not None else rnn_out
        else:
            x_rnn = x_in.unsqueeze(0)
            if masks is not None:
                m = masks.view(1, -1, 1)
                if self.rnn_type == "lstm":
                    hidden_states = (hidden_states[0] * m, hidden_states[1] * m)
                else:
                    hidden_states = hidden_states * m
            rnn_out, next_h = rnn_module(x_rnn, hidden_states)
            latent = rnn_out[0]
        return latent, next_h

    # ------------------------------------------------------------------
    # MoE head: shared gate -> weighted sum of expert outputs.
    # ------------------------------------------------------------------

    def _compute_gate(self, latent):
        gate_in = self.gate_input_norm(latent)
        return F.softmax(self.gate(gate_in), dim=-1)

    def _moe_combine(self, latent, experts):
        """a_t = sum_i g_hat_i * f_i(h_t)  (paper eq. 3)."""
        weights = self._compute_gate(latent)
        out = 0
        for i, expert in enumerate(experts):
            out = out + expert(latent) * weights[..., i].unsqueeze(-1)
        if not self.training:
            with torch.no_grad():
                self.latest_weights = {
                    "unified": weights.reshape(-1, weights.shape[-1]).mean(dim=0).detach()
                }
        return out

    # ------------------------------------------------------------------
    # rsl_rl interface: act / act_inference / evaluate / forward.
    # ------------------------------------------------------------------

    def _make_rnn_input(self, obs, is_critic=False):
        if is_critic:
            key_list = self.critic_keys or self.input_keys
            normalizer = self.critic_obs_normalizer if self.critic_obs_normalization else None
            proprio_dim = self.critic_proprio_dim
        else:
            key_list = self.input_keys
            normalizer = self.actor_obs_normalizer if self.actor_obs_normalization else None
            proprio_dim = self.proprio_dim
        x_raw = self._extract_raw_obs(obs, key_list)
        return self._process_obs(x_raw, obs_dict=obs, normalizer=normalizer, proprio_dim=proprio_dim)

    def act(self, obs, masks=None, hidden_state=None):
        x_in = self._make_rnn_input(obs, is_critic=False)
        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None:
            current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_state is None:
            self.active_hidden_states = next_state
        mean = self._moe_combine(latent, self.actor_experts)
        safe_std = torch.clamp(self.std, min=1e-4)
        self.distribution = torch.distributions.Normal(mean, safe_std)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None, hidden_states=None):
        x_in = self._make_rnn_input(obs, is_critic=False)
        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None:
            current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_states is None:
            self.active_hidden_states = next_state
        return self._moe_combine(latent, self.actor_experts)

    def evaluate(self, obs, masks=None, hidden_state=None):
        x_in = self._make_rnn_input(obs, is_critic=True)
        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None:
            current_state = self._prepare_hidden_state(
                self.active_critic_hidden_states, x_in.device
            )
        latent, next_state = self._run_rnn(self.critic_rnn, x_in, current_state, masks)
        if hidden_state is None:
            self.active_critic_hidden_states = next_state
        # Critic MoE shares the same gate (paper III-B): weighted sum of scalar
        # value heads.
        return self._moe_combine(latent, self.critic_experts)

    def forward(self, obs, masks=None, hidden_states=None, save_dist=True):
        x_in = self._make_rnn_input(obs, is_critic=False)
        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None:
            current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        mean = self._moe_combine(latent, self.actor_experts)
        safe_std = torch.clamp(self.std, min=1e-4)
        if save_dist:
            self.distribution = torch.distributions.Normal(mean, safe_std)
        return mean, safe_std, next_state

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return self.active_hidden_states, self.active_critic_hidden_states

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is not None:
            if isinstance(hidden_states, tuple) and len(hidden_states) == 2:
                actor_hs, critic_hs = hidden_states
            else:
                actor_hs = critic_hs = hidden_states
            device = self.std.device
            self.active_hidden_states = self._prepare_hidden_state(actor_hs, device)
            self.active_critic_hidden_states = self._prepare_hidden_state(critic_hs, device)
        if dones is None:
            return
        if hasattr(dones, "dtype") and dones.dtype == torch.uint8:
            dones = dones.bool()

        def _reset(h, mask):
            if isinstance(h, tuple):
                return tuple(_reset(x, mask) for x in h)
            h = h.clone()
            h[:, mask, :] = 0.0
            return h

        if self.active_hidden_states is not None:
            self.active_hidden_states = _reset(self.active_hidden_states, dones)
        if self.active_critic_hidden_states is not None:
            self.active_critic_hidden_states = _reset(self.active_critic_hidden_states, dones)


# ==============================================================================
# 2. LocoMoE PPO algorithm wrapper
# ==============================================================================


class LocoMoEPPO(PPO):
    """Vanilla PPO for the LocoMoE baseline.

    Per design spec: training protocol matches SplitMoE *with* the auxiliary
    losses stripped — no symmetry loss L_sym, no load-balancing loss L_bal,
    no L_recon. Only L_surrogate + L_value remain. rsl_rl's ``PPO`` base
    implementation already provides exactly that, so no method overrides
    are needed.
    """

    pass


# ==============================================================================
# 3. Configuration dataclasses
# ==============================================================================


@configclass
class LocoMoEActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "LocoMoEActorCritic"

    # MoE-Loco paper architecture knobs (paper L499 + Table VI).
    num_experts: int = 6
    latent_dim: int = 256
    rnn_type: str = "gru"
    gating_hidden_dim: int = 128
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [256, 128, 128])
    activation: str = "elu"
    init_noise_std: float = 1.0

    # Encoder switches (matched to SplitMoEPPOCfg for variable-isolated comparison).
    blind_vision: bool = False
    use_elevation_ae: bool = True
    elevation_dim: int = 187
    use_multilayer_scan: bool = True
    num_scan_channels: int = 24
    num_scan_rays: int = 21

    estimator_output_dim: int = 3
    estimator_hidden_dims: list = field(default_factory=lambda: [128, 64])
    estimator_target_indices: list = field(default_factory=lambda: [0, 1, 2])
    estimator_input_indices: list = field(
        default_factory=lambda: list(range(3, 9)) + list(range(12, 56))
    )
    estimator_obs_normalization: bool = True

    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True

    feed_estimator_to_policy: bool = True
    feed_ae_to_policy: bool = True


@configclass
class LocoMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO Configuration for training the LocoMoE baseline (head-to-head vs SplitMoE)."""

    num_steps_per_env = 36
    max_iterations = 15000  # match ablation iter count for head-to-head comparison
    save_interval = 200
    experiment_name = "locomoe_teacher_parallel"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "estimator": ["estimator"],
        "noisy_elevation": ["noisy_elevation"],
    }

    policy = LocoMoEActorCriticCfg(
        num_experts=6,
        latent_dim=256,
        rnn_type="gru",
        gating_hidden_dim=128,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
        init_noise_std=1.0,
        # Match SplitMoEPPOCfg's perception stack so only the MoE head differs.
        blind_vision=False,
        use_elevation_ae=True,
        elevation_dim=187,
        use_multilayer_scan=True,
        num_scan_channels=24,
        num_scan_rays=21,
        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        feed_estimator_to_policy=True,
        feed_ae_to_policy=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="LocoMoEPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
