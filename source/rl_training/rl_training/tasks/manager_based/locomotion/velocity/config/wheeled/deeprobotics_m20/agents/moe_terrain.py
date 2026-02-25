# source/robot_lab/algo/moe/policy_split_moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import unpad_trajectories
from rsl_rl.algorithms import PPO
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from dataclasses import field
import numpy as np

# ==============================================================================
# 1. 基础组件
# ==============================================================================

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class EmpiricalNormalization(nn.Module):
    def __init__(self, shape, epsilon=1e-4, until_step=None):
        super().__init__()
        self.epsilon = epsilon
        self.until_step = until_step
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def update(self, x):
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean[:] = new_mean
        self.var[:] = new_var
        
        if self.count.ndim == 0:
            self.count.fill_(tot_count)
        else:
            self.count[:] = tot_count

    def forward(self, x):
        if self.training:
            if torch.is_grad_enabled():
                if self.until_step is None or self.count < self.until_step:
                    self.update(x)
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], activation="elu", output_gain=1.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act_fn = nn.ELU() if activation == "elu" else nn.ReLU()
        
        for h_dim in hidden_dims:
            layer = nn.Linear(prev_dim, h_dim)
            orthogonal_init(layer, gain=np.sqrt(2))
            layers.append(layer)
            layers.append(act_fn)
            prev_dim = h_dim
        
        out_layer = nn.Linear(prev_dim, output_dim)
        orthogonal_init(out_layer, gain=output_gain)
        layers.append(out_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# 2. CNN Encoder for Lidar (升级为双通道大视场处理)
# ==============================================================================

class DepthCNN(nn.Module):
    def __init__(self, input_channels=2, output_dim=128):
        super().__init__()
        # Input shape expected: (Batch, 2, 58, 87)
        self.net = nn.Sequential(
            # Layer 1: -> (16, 29, 44)
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            # Layer 2: -> (32, 15, 22)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # Layer 3: -> (64, 8, 11)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # Layer 4: -> (128, 4, 6)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((2, 4)) 
        self.projection = nn.Linear(1024, output_dim)
        self.act = nn.ELU()

    def forward(self, x):
        features = self.net(x)
        features = self.global_pool(features)
        features = features.flatten(1)
        return self.act(self.projection(features))

# ==============================================================================
# 3. Split MoE Policy Network
# ==============================================================================

class SplitMoEActorCritic(ActorCritic):
    is_recurrent = True

    def __init__(self, 
                 obs, 
                 obs_groups, 
                 num_actions, 
                 actor_hidden_dims=[256, 128, 128], 
                 critic_hidden_dims=[512, 256, 128], 
                 activation='elu', 
                 init_noise_std=1.0,
                 # === Split MoE Params ===
                 num_wheel_experts=2, 
                 num_leg_experts=2,
                 num_leg_actions=12,
                 latent_dim=256,
                 rnn_type="gru",
                 aux_loss_coef=0.01,
                 # === CNN Params ===
                 use_cnn=False,
                 num_cameras=2,       # 从 lidar 更改为 camera
                 camera_height=58,    # 深度图高
                 camera_width=87,     # 深度图宽
                 **kwargs):
        
        base_kwargs = {k: v for k, v in kwargs.items() if k not in [
            "estimator_output_dim", "estimator_hidden_dims", 
            "estimator_input_indices", "estimator_target_indices", 
            "estimator_obs_normalization", "init_noise_legs", "init_noise_wheels",
            "actor_obs_normalization", "critic_obs_normalization"
        ]}

        super().__init__(obs, obs_groups, num_actions, 
                         actor_hidden_dims=actor_hidden_dims, 
                         critic_hidden_dims=critic_hidden_dims, 
                         activation=activation, 
                         init_noise_std=init_noise_std, 
                         **base_kwargs)

        self.input_keys = None 
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            keys = obs_groups.get("policy", None)
            self.input_keys = keys
            num_obs = sum(obs[k].shape[-1] for k in keys)
        else:
            num_obs = obs.shape[-1]
            
        self.use_cnn = use_cnn
        self.num_cameras = num_cameras
        self.camera_height = camera_height
        self.camera_width = camera_width
        
        # 计算深度图展平后的总维度: 2 * 58 * 87 = 10092
        self.image_raw_dim = self.num_cameras * self.camera_height * self.camera_width 
        
        # 推断本体感觉维度
        self.proprio_dim = num_obs - self.image_raw_dim if self.use_cnn else num_obs
        print(f"[SplitMoE] Total Obs: {num_obs}, Image Raw: {self.image_raw_dim}, Proprio: {self.proprio_dim}")

        if self.use_cnn:
            self.cnn_output_dim = 128
            self.depth_encoder = DepthCNN(input_channels=self.num_cameras, output_dim=self.cnn_output_dim)
            rnn_input_dim = self.proprio_dim + self.cnn_output_dim
        else:
            rnn_input_dim = num_obs

        self.actor_obs_normalization = kwargs.get("actor_obs_normalization", True)
        self.critic_obs_normalization = kwargs.get("critic_obs_normalization", True)

        if self.actor_obs_normalization:
            print(f"[SplitMoE] Re-initializing Actor Normalizer for Proprio dim only: {self.proprio_dim}")
            self.actor_obs_normalizer = EmpiricalNormalization(shape=[self.proprio_dim], until_step=1.0e9)
        
        if self.critic_obs_normalization:
            print(f"[SplitMoE] Re-initializing Critic Normalizer for Proprio dim only: {self.proprio_dim}")
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[self.proprio_dim], until_step=1.0e9)

        self.latent_dim = latent_dim
        self.rnn_type = rnn_type.lower()
        self.aux_loss_coef = aux_loss_coef
        
        self.num_leg_actions = num_leg_actions
        self.num_wheel_actions = num_actions - num_leg_actions

        print(f"[SplitMoE] RNN Input Dim: {rnn_input_dim}")

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=self.latent_dim, batch_first=False)
        else:
            self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=self.latent_dim, batch_first=False)

        for name, param in self.rnn.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.constant_(param, 0)

        self.gate_input_norm = nn.LayerNorm(self.latent_dim)
        self.leg_gate = nn.Sequential(nn.Linear(self.latent_dim, 64), nn.ELU(), nn.Linear(64, num_leg_experts))
        self.wheel_gate = nn.Sequential(nn.Linear(self.latent_dim, 64), nn.ELU(), nn.Linear(64, num_wheel_experts))

        self._init_gate(self.leg_gate)
        self._init_gate(self.wheel_gate)

        self.actor_leg_experts = nn.ModuleList([
            MLP(self.latent_dim, self.num_leg_actions, hidden_dims=actor_hidden_dims, activation=activation, output_gain=0.01) 
            for _ in range(num_leg_experts)
        ])

        self.actor_wheel_experts = nn.ModuleList([
            MLP(self.latent_dim, self.num_wheel_actions, hidden_dims=actor_hidden_dims, activation=activation, output_gain=0.01) 
            for _ in range(num_wheel_experts)
        ])

        self.critic_mlp = MLP(self.latent_dim, 1, hidden_dims=critic_hidden_dims, activation=activation, output_gain=1.0)

        self.estimator_output_dim = kwargs.get("estimator_output_dim", 0)
        self.estimator_hidden_dims = kwargs.get("estimator_hidden_dims", [128, 64])
        self.estimator_obs_normalization = kwargs.get("estimator_obs_normalization", True)
        self.estimator_target_indices = kwargs.get("estimator_target_indices", [0, 1, 2])
        self.estimator_input_indices = kwargs.get("estimator_input_indices", list(range(3, 32)))
        
        if self.estimator_output_dim > 0:
            est_input_dim = 0
            self.has_estimator_group = False
            try:
                est_group = obs["estimator"]
                est_input_dim = est_group.shape[-1]
                self.has_estimator_group = True
            except (KeyError, TypeError, AttributeError):
                self.has_estimator_group = False
                est_input_dim = len(self.estimator_input_indices)

            if est_input_dim > 0:
                self.estimator = MLP(est_input_dim, self.estimator_output_dim, hidden_dims=self.estimator_hidden_dims, activation=activation)
                self.estimator_obs_normalizer = None
                if self.estimator_obs_normalization:
                    self.estimator_obs_normalizer = EmpiricalNormalization(shape=[est_input_dim], until_step=1.0e9)
        else:
            self.estimator = None

        if isinstance(obs, dict): ref_tensor = obs[list(obs.keys())[0]]
        else: ref_tensor = obs
        batch_size = ref_tensor.shape[0]
        device = ref_tensor.device
        
        self.active_hidden_states = self._init_rnn_state(batch_size, device)
        self.critic_keys = obs_groups.get("critic", None)
        self.latest_weights = {}
        self.active_aux_loss = 0.0
        self.active_estimator_loss = 0.0
        self.active_estimator_error = 0.0
        
        new_std = torch.ones(num_actions)
        self.num_wheel_experts = num_wheel_experts
        self.num_leg_experts = num_leg_experts
        noise_legs = kwargs.get("init_noise_legs", 1.0)
        noise_wheels = kwargs.get("init_noise_wheels", 0.4) 
        if num_leg_actions <= num_actions:
            new_std[:num_leg_actions] = noise_legs
            new_std[num_leg_actions:] = noise_wheels
        self.std.data.copy_(new_std.to(device))

    def _init_rnn_state(self, batch_size, device):
        if self.rnn_type == "lstm":
            return (torch.zeros(1, batch_size, self.latent_dim, device=device), 
                    torch.zeros(1, batch_size, self.latent_dim, device=device))
        else:
            return torch.zeros(1, batch_size, self.latent_dim, device=device)

    def _init_gate(self, gate_net):
        orthogonal_init(gate_net[0], gain=np.sqrt(2))
        orthogonal_init(gate_net[2], gain=0.01)

    def _extract_raw_obs(self, obs, key_list):
        if key_list is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in key_list]
            return torch.cat(tensors, dim=-1)
        return obs

    # === [Fix] 新增: 覆盖 update_normalization 以解决维度不匹配问题 ===
    def update_normalization(self, obs):
        """
        覆盖父类方法。分离 Actor 和 Critic 的归一化更新，确保 Critic 使用无噪声数据。
        """
        # 1. Update Actor Normalizer (Noised data)
        if self.actor_obs_normalization:
            x_raw_actor = self._extract_raw_obs(obs, self.input_keys)
            proprio_actor = x_raw_actor[..., :self.proprio_dim]
            self.actor_obs_normalizer.update(proprio_actor)
            
        # 2. Update Critic Normalizer (Clean data)
        if self.critic_obs_normalization:
            x_raw_critic = self._extract_raw_obs(obs, getattr(self, "critic_keys", self.input_keys))
            proprio_critic = x_raw_critic[..., :self.proprio_dim]
            self.critic_obs_normalizer.update(proprio_critic)

    def _process_obs(self, x, normalizer=None):
        """核心切片逻辑：分离本体感觉和深度图"""
        proprio = x[..., :self.proprio_dim]
        depth_raw = x[..., self.proprio_dim:]
        
        if normalizer is not None:
            proprio = normalizer(proprio)
            
        if self.use_cnn:
            original_shape = depth_raw.shape[:-1] 
            # 还原为图像维度 (Batch, Cameras, Height, Width)
            depth_image = depth_raw.view(
                *original_shape, 
                self.num_cameras, 
                self.camera_height, 
                self.camera_width
            )
            
            # 适配 RNN 产生的 5D 张量
            if depth_image.ndim == 5: 
                B, T, C, H, W = depth_image.shape
                depth_image = depth_image.view(B*T, C, H, W)
                depth_feat = self.depth_encoder(depth_image)
                depth_feat = depth_feat.view(B, T, -1)
            else:
                depth_feat = self.depth_encoder(depth_image)
            
            x_out = torch.cat([proprio, depth_feat], dim=-1)
        else:
            x_out = torch.cat([proprio, depth_raw], dim=-1)
            
        return x_out
    
    def _prepare_input(self, obs, key_list):
        x = self._extract_raw_obs(obs, key_list)
        return self._process_obs(x, normalizer=None) 

    def _run_rnn(self, rnn_module, x_in, hidden_states, masks):
        if hidden_states is None:
            B = x_in.shape[1] if x_in.ndim == 3 else x_in.shape[0]
            rnn_state = self._init_rnn_state(B, x_in.device)
        else:
            rnn_state = hidden_states

        if x_in.ndim == 3: 
            rnn_out, next_rnn_state = rnn_module(x_in, rnn_state)
            if masks is not None: latent = unpad_trajectories(rnn_out, masks)
            else: latent = rnn_out
        elif x_in.ndim == 2: 
            x_rnn = x_in.unsqueeze(0)
            if masks is not None:
                m = masks.view(1, -1, 1)
                if self.rnn_type == "lstm": rnn_state = (rnn_state[0] * m, rnn_state[1] * m)
                else: rnn_state = rnn_state * m
            rnn_out, next_rnn_state = rnn_module(x_rnn, rnn_state)
            latent = rnn_out[0]
        return latent, next_rnn_state

    def _get_estimator_input(self, obs_dict):
        if self.has_estimator_group:
            try: return obs_dict["estimator"]
            except (KeyError, TypeError): pass
        
        if hasattr(obs_dict, "keys") or isinstance(obs_dict, dict):
            try: 
                full_obs = self._extract_raw_obs(obs_dict, self.input_keys)
            except KeyError: 
                full_obs = self._extract_raw_obs(obs_dict, self.input_keys)
        else: 
            full_obs = obs_dict 
            
        return full_obs[..., self.estimator_input_indices]

    def _compute_actor_output(self, latent, obs_dict=None):
        gate_in = self.gate_input_norm(latent)
        leg_logits = self.leg_gate(gate_in)
        w_leg = F.softmax(leg_logits, dim=-1)
        wheel_logits = self.wheel_gate(gate_in)
        w_wheel = F.softmax(wheel_logits, dim=-1)

        leg_act_sum = 0
        for i, expert in enumerate(self.actor_leg_experts):
            leg_act_sum += expert(latent) * w_leg[..., i].unsqueeze(-1)
        wheel_act_sum = 0
        for i, expert in enumerate(self.actor_wheel_experts):
            wheel_act_sum += expert(latent) * w_wheel[..., i].unsqueeze(-1)
        total_action = torch.cat([leg_act_sum, wheel_act_sum], dim=-1)

        if self.training:
            # 仅仅保留 Load Balancing Loss
            self.active_aux_loss = self._calculate_load_balancing_loss(w_leg, w_wheel) * self.aux_loss_coef
            
            # -------------------------------------------------------------------
            # [删除此处原本的 if self.estimator is not None and obs_dict is not None:]
            # 以及里面的 estimated_state 和 target_state 逻辑
            # -------------------------------------------------------------------
        
        with torch.no_grad():
            def flat_mean(w): return w.reshape(-1, w.shape[-1]).mean(dim=0).detach()
            self.latest_weights = {"leg": flat_mean(w_leg), "wheel": flat_mean(w_wheel)}
        return total_action

    def _calculate_load_balancing_loss(self, w_leg, w_wheel):
        loss = 0.0
        leg_usage = w_leg.reshape(-1, w_leg.shape[-1]).mean(dim=0)
        target_leg = torch.full_like(leg_usage, 1.0 / self.num_leg_experts)
        loss += (leg_usage - target_leg).pow(2).sum()
        wheel_usage = w_wheel.reshape(-1, w_wheel.shape[-1]).mean(dim=0)
        target_wheel = torch.full_like(wheel_usage, 1.0 / self.num_wheel_experts)
        loss += (wheel_usage - target_wheel).pow(2).sum()
        return loss

    def _prepare_hidden_state(self, hidden, device):
        if hidden is None: return None
        if isinstance(hidden, (tuple, list)):
            if len(hidden) == 2 and not self.rnn_type == "lstm": hidden = hidden[0]
            elif self.rnn_type == "lstm" and len(hidden) == 2:
                if isinstance(hidden[0], (tuple, list)): hidden = hidden[0]
        def to_dev(h):
            if isinstance(h, (tuple, list)): return tuple(x.to(device).contiguous() for x in h)
            return h.to(device).contiguous()
        return to_dev(hidden)

    def act(self, obs, masks=None, hidden_state=None):
        x_raw = self._extract_raw_obs(obs, self.input_keys)
        normalizer = self.actor_obs_normalizer if self.actor_obs_normalization else None
        x_in = self._process_obs(x_raw, normalizer)
        
        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_state is None: self.active_hidden_states = next_state
        
        mean = self._compute_actor_output(latent, obs_dict=obs) 
        self.distribution = torch.distributions.Normal(mean, self.std)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None, hidden_states=None):
        x_raw = self._extract_raw_obs(obs, self.input_keys)
        normalizer = self.actor_obs_normalizer if self.actor_obs_normalization else None
        x_in = self._process_obs(x_raw, normalizer)

        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_states is None: self.active_hidden_states = next_state
        return self._compute_actor_output(latent)

    def evaluate(self, obs, masks=None, hidden_state=None):
        # x_raw = self._extract_raw_obs(obs, self.input_keys)
        x_raw = self._extract_raw_obs(obs, getattr(self, "critic_keys", self.input_keys))
        normalizer = self.critic_obs_normalizer if self.critic_obs_normalization else None
        x_in = self._process_obs(x_raw, normalizer)

        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, _ = self._run_rnn(self.rnn, x_in, current_state, masks)
        return self.critic_mlp(latent)
    
    def get_estimated_state(self, obs, masks=None, hidden_states=None):
        if self.estimator is None: raise RuntimeError("Estimator is not initialized!")
        est_input = self._get_estimator_input(obs)
        if self.estimator_obs_normalization and self.estimator_obs_normalizer is not None:
             est_input = self.estimator_obs_normalizer(est_input)
        return self.estimator(est_input)

    def forward(self, obs, masks=None, hidden_states=None, save_dist=True):
        x_raw = self._extract_raw_obs(obs, self.input_keys)
        normalizer = self.actor_obs_normalizer if self.actor_obs_normalization else None
        x_in = self._process_obs(x_raw, normalizer)

        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        action_mean = self._compute_actor_output(latent, obs_dict=obs)
        if save_dist: self.distribution = torch.distributions.Normal(action_mean, self.std)
        return action_mean, self.std, next_state

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_hidden_states(self):
        return self.active_hidden_states, self.active_hidden_states
    
    def reset(self, dones=None):
        if dones is None: return
        def reset_hidden(h, mask):
            if isinstance(h, tuple): return tuple(reset_hidden(x, mask) for x in h)
            else: 
                h[:, mask, :] = 0.0
                return h
        if self.active_hidden_states is not None:
            self.active_hidden_states = reset_hidden(self.active_hidden_states, dones)

# ==============================================================================
# 4. PPO Algorithm Wrapper & Configs
# ==============================================================================

class SplitMoEPPO(PPO):
    def update(self):
        loss_dict = super().update()
        model = getattr(self, "actor_critic", getattr(self, "policy", None))
        
        if model is not None and self.num_learning_epochs > 0:
            obs_batch = self.storage.observations.flatten(0, 1)
            self.optimizer.zero_grad()
            total_aux_loss = 0.0
            
            if model.estimator is not None:
                est_input = model._get_estimator_input(obs_batch)
                if hasattr(obs_batch, "keys") or isinstance(obs_batch, dict):
                    try: full_obs_for_target = obs_batch["policy"]
                    except KeyError: full_obs_for_target = model._extract_raw_obs(obs_batch, model.input_keys)
                else: full_obs_for_target = obs_batch
                target_state = full_obs_for_target[..., model.estimator_target_indices]
                if model.estimator_obs_normalization and model.estimator_obs_normalizer is not None:
                    est_input = model.estimator_obs_normalizer(est_input)
                estimated_state = model.estimator(est_input)
                est_loss = (estimated_state - target_state).pow(2).mean()
                total_aux_loss += est_loss
                model.active_estimator_loss = est_loss.detach()
                model.active_estimator_error = (estimated_state - target_state).abs().mean().detach()

            if isinstance(total_aux_loss, torch.Tensor):
                total_aux_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        if model is not None:
            if hasattr(model, "latest_weights") and model.latest_weights:
                w = model.latest_weights
                if "leg" in w:
                    for i, val in enumerate(w["leg"]): loss_dict[f"Gate/Leg_Expert_{i}"] = val.item()
                if "wheel" in w:
                    for i, val in enumerate(w["wheel"]): loss_dict[f"Gate/Wheel_Expert_{i}"] = val.item()
            if hasattr(model, "active_aux_loss"): loss_dict["Loss/Load_Balancing"] = model.active_aux_loss.item() if isinstance(model.active_aux_loss, torch.Tensor) else model.active_aux_loss
            if hasattr(model, "active_estimator_loss"): loss_dict["Loss/Estimator_MSE"] = model.active_estimator_loss.item() if isinstance(model.active_estimator_loss, torch.Tensor) else model.active_estimator_loss
            if hasattr(model, "active_estimator_error"): loss_dict["Loss/Estimator_Error_MAE"] = model.active_estimator_error.item() if isinstance(model.active_estimator_error, torch.Tensor) else model.active_estimator_error
            if hasattr(model, "std"):
                std_np = model.std.detach().cpu().numpy()
                n_legs = getattr(model, "num_leg_actions", 12)
                if len(std_np) >= n_legs:
                    loss_dict["Noise/Leg_Std"] = std_np[:n_legs].mean()
                    loss_dict["Noise/Wheel_Std"] = std_np[n_legs:].mean() if len(std_np) > n_legs else 0.0
        return loss_dict

@configclass
class SplitMoEActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "SplitMoEActorCritic"
    num_wheel_experts: int = 6
    num_leg_experts: int = 6
    num_leg_actions: int = 12
    init_noise_std: float = 1.0
    init_noise_legs: float = 1.0
    init_noise_wheels: float = 0.5
    latent_dim: int = 256
    rnn_type: str = "gru"
    aux_loss_coef: float = 0.01

    use_cnn: bool = True
    num_cameras: int = 2
    camera_height: int = 58
    camera_width: int = 87

    estimator_output_dim: int = 3  
    estimator_hidden_dims: list = field(default_factory=lambda: [128, 64])
    estimator_target_indices: list = field(default_factory=lambda: [0, 1, 2])
    estimator_input_indices: list = field(default_factory=lambda: list(range(3, 9)) + list(range(12, 56)))
    estimator_obs_normalization: bool = True 
    
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    
    def get_std(self):
            std_np = self.std.detach().cpu().numpy()
            if self.num_leg_actions > 0:
                leg_std = std_np[:self.num_leg_actions].mean()
            else:
                leg_std = 0.0
            if self.num_leg_actions < len(std_np):
                wheel_std = std_np[self.num_leg_actions:].mean()
            else:
                wheel_std = 0.0
            return leg_std, wheel_std

@configclass
class SplitMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 25000
    save_interval = 200
    experiment_name = "split_moe_parallel" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"]}
    
    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0, 
        init_noise_legs=0.8,
        init_noise_wheels=0.5, 
        actor_hidden_dims=[256, 128, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=3,
        num_leg_experts=6,
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,
        
        use_cnn=False,
        num_cameras=2,
        camera_height=58,
        camera_width=87,

        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,
        
        actor_obs_normalization=True, 
        critic_obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )