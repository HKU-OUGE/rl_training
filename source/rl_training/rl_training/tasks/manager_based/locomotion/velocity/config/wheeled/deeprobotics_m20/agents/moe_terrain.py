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
    RslRlDistillationRunnerCfg,       # Added for Distillation
    RslRlDistillationAlgorithmCfg     # Added for Distillation
)
from dataclasses import field
import numpy as np
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import Distillation
# ==============================================================================
# 1. Base Components
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
# 2. CMoE: ElevationAE, DepthAE & ProprioVAE 
# ==============================================================================

class ElevationAE(nn.Module):
    def __init__(self, input_dim=187, output_dim=64, grid_shape=(11, 17)):
        super().__init__()
        self.grid_shape = grid_shape
        self.input_dim = input_dim

        # ---------------------------------------------------------
        # Encoder: 2D CNN 提取空间特征
        # 输入尺寸: (Batch, 1, 11, 17)
        # ---------------------------------------------------------
        self.encoder_conv = nn.Sequential(
            # 输出: (16, 6, 9) -> 计算方式: floor((size + 2*pad - kernel)/stride) + 1
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ELU(),
            # 输出: (32, 3, 5)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
        )
        # 32 channels * 3 height * 5 width = 480
        self.encoder_fc = nn.Linear(32 * 3 * 5, output_dim)

        # ---------------------------------------------------------
        # Decoder: 2D CNN 还原地形
        # ---------------------------------------------------------
        self.decoder_fc = nn.Linear(output_dim, 32 * 3 * 5)
        
        # 使用 Nearest Upsample + 保持尺寸的 Conv2d 替代反卷积 (减少棋盘效应)
        self.decoder_conv = nn.Sequential(
            nn.Upsample(size=(6, 9), mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ELU(),
            
            nn.Upsample(size=grid_shape, mode='nearest'),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # 最后一层直接输出高度值
        )

    def forward(self, x):
        # 记录原始输入的 batch 维度形状 (兼容 [B, 187] 或 [Seq, B, 187])
        original_shape = x.shape[:-1]
        
        # 1. 展平批次维度并 Reshape 成 2D 图像格式: (B_total, C, H, W)
        x_img = x.reshape(-1, 1, *self.grid_shape)

        # 2. Encoder 前向传播
        conv_features = self.encoder_conv(x_img)
        conv_flat = conv_features.reshape(conv_features.size(0), -1)
        latent_flat = self.encoder_fc(conv_flat)

        # 3. Decoder 前向传播 (仅在训练时重建)
        recon = None
        if self.training:
            dec_in = self.decoder_fc(latent_flat)
            dec_in = dec_in.reshape(-1, 32, 3, 5)
            recon_img = self.decoder_conv(dec_in)
            
            # 将重建的图像重新展平成 187 维，并恢复原始 Batch 维度
            recon = recon_img.reshape(*original_shape, self.input_dim)
        # 4. 恢复 latent 的原始批次形状
        latent = latent_flat.reshape(*original_shape, -1)

        return latent, recon

class MultiLayerScanAE(nn.Module):
    def __init__(self, num_channels=12, num_rays=21, output_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        self.num_channels = num_channels
        self.num_rays = num_rays
        self.flat_dim = num_channels * num_rays
        self.actual_channels = num_channels // 2  
        
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(self.actual_channels * 2, 16, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
        )
        
        conv_out_len = (self.num_rays + 1) // 2
        conv_out_len = (conv_out_len + 1) // 2
        linear_in_dim = 32 * conv_out_len 
        
        self.encoder_linear = nn.Sequential(
            nn.Linear(linear_in_dim, hidden_dims[0]), nn.ELU(),
            nn.Linear(hidden_dims[0], output_dim)
        )
        self.decoder = MLP(output_dim, self.flat_dim, hidden_dims[::-1], activation="elu")

    def forward(self, x):
        original_shape = x.shape[:-1] 
        x_reshaped = x.reshape(-1, self.num_channels, self.num_rays) 
        
        fwd = x_reshaped[:, :self.actual_channels, :] 
        bwd = x_reshaped[:, self.actual_channels:, :] 
        bwd = torch.flip(bwd, dims=[-1])
        
        x_spatial = torch.cat([fwd, bwd], dim=1) 
        
        conv_features = self.encoder_conv(x_spatial).reshape(x_spatial.shape[0], -1)
        latent_flat = self.encoder_linear(conv_features)
        
        latent = latent_flat.reshape(*original_shape, -1)
        recon = self.decoder(latent) if self.training else None
        
        return latent, recon

class DepthAE(nn.Module):
    def __init__(self, input_channels=2, output_dim=128, camera_height=58, camera_width=87):
        super().__init__()
        self.camera_height = camera_height
        self.camera_width = camera_width
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((2, 4)) 
        self.fc_encode = nn.Linear(1024, output_dim)
        self.act = nn.ELU()

        self.fc_decode = nn.Linear(output_dim, 1024)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, input_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        features_pooled = self.global_pool(features).flatten(1)
        latent = self.act(self.fc_encode(features_pooled)) 
        
        recon = None
        if self.training:
            dec_in = self.act(self.fc_decode(latent))
            dec_in = dec_in.view(-1, 128, 2, 4)
            recon = self.decoder(dec_in)
            recon = F.interpolate(recon, size=(self.camera_height, self.camera_width), mode='bilinear', align_corners=False)
            
        return latent, recon

class ProprioVAE(nn.Module):
    def __init__(self, input_dim, vel_dim=3, latent_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        self.encoder_mlp = MLP(input_dim, hidden_dims[-1], hidden_dims[:-1])
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.vel_estimator = nn.Linear(hidden_dims[-1], vel_dim)
        self.decoder_mlp = MLP(latent_dim, input_dim, hidden_dims[::-1])

    def encode(self, x):
        h = self.encoder_mlp(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        vel_pred = self.vel_estimator(h) 
        return mu, logvar, vel_pred

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_mlp(z)

    def forward(self, x):
        mu, logvar, vel_pred = self.encode(x)
        logvar = torch.clamp(logvar, min=-20.0, max=10.0)
        z = self.reparameterize(mu, logvar) if self.training else mu 
        recon_x = self.decode(z) if self.training else None
        return vel_pred, recon_x, mu, logvar, z

# ==============================================================================
# 3. Split MoE Policy Network
# ==============================================================================

class SplitMoEActorCritic(ActorCritic):
    is_recurrent = True

    def __init__(self, obs, obs_groups, num_actions, actor_hidden_dims=[256, 128, 128], 
                 critic_hidden_dims=[512, 256, 128], activation='elu', init_noise_std=1.0,
                 num_wheel_experts=2, num_leg_experts=2, num_leg_actions=12,
                 latent_dim=256, rnn_type="gru", aux_loss_coef=0.01,
                 blind_vision=True, use_elevation_ae=True, elevation_dim=187,     
                 use_cnn=False, num_cameras=2, camera_height=58, camera_width=87,
                 forced_input_key=None, is_student_mode=False, 
                 feed_estimator_to_policy=False, feed_ae_to_policy=False, **kwargs):
        
        base_kwargs = {k: v for k, v in kwargs.items() if k not in [
            "estimator_output_dim", "estimator_hidden_dims", 
            "estimator_input_indices", "estimator_target_indices", 
            "estimator_obs_normalization", "init_noise_legs", "init_noise_wheels",
            "actor_obs_normalization", "critic_obs_normalization",
            "use_elevation_ae", "elevation_dim", "blind_vision", "is_student_mode",
            "feed_estimator_to_policy", "feed_ae_to_policy",
            "use_multilayer_scan", "num_scan_channels", "num_scan_rays", "teacher_is_mlp",
            "use_cnn", "num_cameras", "camera_height", "camera_width",
            "sym_loss_coef"
        ]}

        super().__init__(obs, obs_groups, num_actions, actor_hidden_dims=actor_hidden_dims, 
                         critic_hidden_dims=critic_hidden_dims, activation=activation, 
                         init_noise_std=init_noise_std, **base_kwargs)

        self.is_student_mode = is_student_mode
        self.feed_estimator = feed_estimator_to_policy
        self.feed_ae = feed_ae_to_policy

        self.input_keys = None 
        if forced_input_key:
            self.input_keys = [forced_input_key]
            try:
                num_obs = obs[forced_input_key].shape[-1]
            except (KeyError, TypeError):
                if hasattr(obs, "keys"):
                     found = False
                     obs_keys = list(obs.keys())
                     if forced_input_key in obs_keys:
                         num_obs = obs[forced_input_key].shape[-1]
                         found = True
                     if not found and obs_groups and forced_input_key in obs_groups:
                         group_keys = obs_groups[forced_input_key]
                         if group_keys:
                             valid_keys = [k for k in group_keys if k in obs_keys]
                             if valid_keys:
                                 num_obs = sum(obs[k].shape[-1] for k in valid_keys)
                                 found = True
                     if not found and len(obs_keys) > 0:
                         num_obs = list(obs.values())[0].shape[-1]
                else:
                     num_obs = obs.shape[-1]
        elif isinstance(obs, dict) or hasattr(obs, "keys"):
            keys = obs_groups.get("policy", None)
            self.input_keys = keys
            if keys: num_obs = sum(obs[k].shape[-1] for k in keys)
            else: num_obs = list(obs.values())[0].shape[-1]
        else:
            num_obs = obs.shape[-1]
            
        self.blind_vision = blind_vision
        self.use_cnn = use_cnn
        self.use_elevation_ae = use_elevation_ae
        self.elevation_dim = elevation_dim
        
        self.has_elevation_input = False
        if self.use_elevation_ae:
            if num_obs > self.elevation_dim:
                self.proprio_dim = num_obs - self.elevation_dim
                self.has_elevation_input = True
            else:
                self.proprio_dim = num_obs
                self.has_elevation_input = False
        elif self.use_cnn:
            self.image_raw_dim = num_cameras * camera_height * camera_width 
            if num_obs > self.image_raw_dim:
                self.proprio_dim = num_obs - self.image_raw_dim
                self.has_elevation_input = True
            else:
                self.proprio_dim = num_obs
                self.has_elevation_input = False
        else:
            self.proprio_dim = num_obs

        self.critic_keys = obs_groups.get("critic", None)
        if self.critic_keys and (isinstance(obs, dict) or hasattr(obs, "keys")):
            critic_num_obs = sum(obs[k].shape[-1] for k in self.critic_keys)
        else:
            critic_num_obs = num_obs 
            
        if self.use_elevation_ae and critic_num_obs > self.elevation_dim:
            self.critic_proprio_dim = critic_num_obs - self.elevation_dim
        elif self.use_cnn and critic_num_obs > self.image_raw_dim:
            self.critic_proprio_dim = critic_num_obs - self.image_raw_dim
        else:
            self.critic_proprio_dim = critic_num_obs

        self.estimator_output_dim = kwargs.get("estimator_output_dim", 0)
        self.estimator_hidden_dims = kwargs.get("estimator_hidden_dims", [128, 64])
        self.estimator_obs_normalization = kwargs.get("estimator_obs_normalization", True)
        self.estimator_target_indices = kwargs.get("estimator_target_indices", [0, 1, 2])
        self.estimator_input_indices = kwargs.get("estimator_input_indices", list(range(3, 32)))
        
        est_input_dim = 0
        if self.estimator_output_dim > 0:
            self.has_estimator_group = False
            try:
                if obs_groups is not None and "estimator" not in obs_groups:
                    raise KeyError("estimator explicitly omitted from obs_groups")
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
                    hidden_dims=self.estimator_hidden_dims
                )
                self.estimator_obs_normalizer = None
                if self.estimator_obs_normalization:
                    self.estimator_obs_normalizer = EmpiricalNormalization(shape=[est_input_dim], until_step=1.0e9)
                
                self.vae_feature_dim = self.estimator_output_dim + 64
                if self.is_student_mode:
                    for param in self.estimator.parameters():
                        param.requires_grad = False
                    self.estimator.eval()
            else:
                self.estimator = None
                self.vae_feature_dim = 0
        else:
            self.estimator = None
            self.vae_feature_dim = 0

        self.use_multilayer_scan = kwargs.get("use_multilayer_scan", True)
        self.num_scan_channels = kwargs.get("num_scan_channels", 12)
        self.num_scan_rays = kwargs.get("num_scan_rays", 21)
        self.scan_dim = self.num_scan_channels * self.num_scan_rays

        self.ae_output_dim = 0
        if self.use_elevation_ae:
            self.elev_out_dim = 64
            self.ae_output_dim += self.elev_out_dim
            self.elevation_encoder = ElevationAE(input_dim=self.elevation_dim, output_dim=self.elev_out_dim)
            if self.is_student_mode:
                for param in self.elevation_encoder.parameters(): param.requires_grad = False
                self.elevation_encoder.eval()
                
        if self.use_multilayer_scan:
            self.scan_out_dim = 64
            self.ae_output_dim += self.scan_out_dim
            self.scan_encoder = MultiLayerScanAE(
                num_channels=self.num_scan_channels, num_rays=self.num_scan_rays, output_dim=self.scan_out_dim)
            if self.is_student_mode:
                for param in self.scan_encoder.parameters(): param.requires_grad = False
                self.scan_encoder.eval()

        rnn_input_dim = self.proprio_dim
        critic_rnn_input_dim = self.critic_proprio_dim

        if getattr(self, 'has_elevation_input', True):
            rnn_input_dim += self.ae_output_dim if self.feed_ae else (num_obs - self.proprio_dim)
            critic_rnn_input_dim += self.ae_output_dim if self.feed_ae else (critic_num_obs - self.critic_proprio_dim)
        else:
            if self.feed_ae:
                rnn_input_dim += self.ae_output_dim
                critic_rnn_input_dim += self.ae_output_dim

        if self.feed_estimator:
            rnn_input_dim += self.vae_feature_dim
            critic_rnn_input_dim += self.vae_feature_dim

        print(f"[SplitMoE] RNN Actor Input: {rnn_input_dim} (Proprio: {self.proprio_dim}, "
              f"VAE appended: {self.vae_feature_dim if self.feed_estimator else 0}, "
              f"Env appended: {self.ae_output_dim if self.feed_ae else (num_obs - self.proprio_dim)}, "
              f"Blind: {self.blind_vision}, Student: {self.is_student_mode})")

        self.actor_obs_normalization = kwargs.get("actor_obs_normalization", True)
        self.critic_obs_normalization = kwargs.get("critic_obs_normalization", True)

        if self.actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(shape=[self.proprio_dim], until_step=1.0e9)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[self.critic_proprio_dim], until_step=1.0e9)

        self.latent_dim = latent_dim
        self.rnn_type = rnn_type.lower()
        self.aux_loss_coef = aux_loss_coef
        self.sym_loss_coef = kwargs.get("sym_loss_coef", 1.0)
        # sym_loss_coef<=0 时禁用对称性增强, 避免 rollout 2x 前向与更新阶段镜像 MSE 的额外开销,
        # 同时让 RNN 隐状态只用 1 层 (正常路径), 不再维护镜像 piggyback 层.
        self.sym_enabled = bool(self.sym_loss_coef > 0.0)
        self.num_leg_actions = num_leg_actions
        self.num_wheel_actions = num_actions - num_leg_actions

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=self.latent_dim, batch_first=False)
            self.critic_rnn = nn.LSTM(input_size=critic_rnn_input_dim, hidden_size=self.latent_dim, batch_first=False)
        else:
            self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=self.latent_dim, batch_first=False)
            self.critic_rnn = nn.GRU(input_size=critic_rnn_input_dim, hidden_size=self.latent_dim, batch_first=False)

        for rnn_net in [self.rnn, self.critic_rnn]:
            for name, param in rnn_net.named_parameters():
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

        if isinstance(obs, dict): ref_tensor = obs[list(obs.keys())[0]]
        else: ref_tensor = obs
        batch_size = ref_tensor.shape[0]
        device = ref_tensor.device
        
        self.active_hidden_states = self._init_rnn_state(batch_size, device)
        self.active_critic_hidden_states = self._init_rnn_state(batch_size, device)
        
        self.latest_weights = {}
        
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
        # Layer 0 = 真实状态; Layer 1 = 镜像 piggyback 状态 (仅在 sym_enabled 时使用).
        num_layers = 2 if getattr(self, "sym_enabled", True) else 1
        if self.rnn_type == "lstm":
            return (torch.zeros(num_layers, batch_size, self.latent_dim, device=device),
                    torch.zeros(num_layers, batch_size, self.latent_dim, device=device))
        else:
            return torch.zeros(num_layers, batch_size, self.latent_dim, device=device)

    def _init_gate(self, gate_net):
        orthogonal_init(gate_net[0], gain=np.sqrt(2))
        orthogonal_init(gate_net[2], gain=0.01)

    def _extract_raw_obs(self, obs, key_list):
        if key_list is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in key_list if k in obs]
            if not tensors: return list(obs.values())[0]
            return torch.cat(tensors, dim=-1)
        return obs

    def update_normalization(self, obs):
        if getattr(self, "actor_obs_normalization", False) and self.actor_obs_normalizer is not None:
            x_raw_actor = self._extract_raw_obs(obs, getattr(self, "input_keys", None))
            proprio_actor = x_raw_actor[..., :self.proprio_dim]
            self.actor_obs_normalizer.update(proprio_actor)
            
        if getattr(self, "critic_obs_normalization", False) and self.critic_obs_normalizer is not None:
            x_raw_critic = self._extract_raw_obs(obs, getattr(self, "critic_keys", self.input_keys))
            proprio_critic = x_raw_critic[..., :self.critic_proprio_dim] 
            self.critic_obs_normalizer.update(proprio_critic)

        if self.estimator is not None and getattr(self, "estimator_obs_normalization", False) and self.estimator_obs_normalizer is not None:
            est_input = self._get_estimator_input(obs)
            self.estimator_obs_normalizer.update(est_input)

    def _process_obs(self, x, obs_dict=None, normalizer=None, proprio_dim=None, compute_reconstruction=False):
        if proprio_dim is None: 
            proprio_dim = self.proprio_dim
            
        proprio = x[..., :proprio_dim]
        if normalizer is not None:
            proprio = normalizer(proprio)
        
        env_feat_for_rnn = None
        aux_outputs = {}
        if (self.use_elevation_ae or self.use_multilayer_scan) and self.feed_ae:
            if obs_dict is not None and (isinstance(obs_dict, dict) or hasattr(obs_dict, "keys")):
                if "noisy_elevation" in obs_dict:
                    env_raw_full = obs_dict["noisy_elevation"]
                elif "policy" in obs_dict:
                     env_raw_full = obs_dict["policy"][..., proprio_dim:]
                else:
                    raise KeyError("Obs dictionary does not contain 'noisy_elevation'!")
            else:
                env_raw_full = x[..., proprio_dim:]
            
            feats = []
            current_idx = 0
            
            # -----------------------------------------------------------------
            # 流 1: 高程图 ElevationAE
            # -----------------------------------------------------------------
            if self.use_elevation_ae:
                env_raw_elev = env_raw_full[..., current_idx : current_idx + self.elevation_dim]
                if compute_reconstruction and self.training:
                    latent_elev, recon_elev = self.elevation_encoder(env_raw_elev)
                    aux_outputs["elev_recon"] = recon_elev     
                    aux_outputs["elev_target"] = env_raw_elev
                else:
                    was_training = self.elevation_encoder.training
                    self.elevation_encoder.eval()
                    latent_elev, _ = self.elevation_encoder(env_raw_elev)
                    self.elevation_encoder.train(was_training)
                    
                feats.append(latent_elev.detach())
                current_idx += self.elevation_dim
                
            # -----------------------------------------------------------------
            # 流 2: 多层扫描 MultiLayerScanAE
            # -----------------------------------------------------------------
            if self.use_multilayer_scan:
                env_raw_scan = env_raw_full[..., current_idx : current_idx + self.scan_dim]
                if compute_reconstruction and self.training:
                    latent_scan, recon_scan = self.scan_encoder(env_raw_scan)
                    aux_outputs["scan_recon"] = recon_scan       
                    aux_outputs["scan_target"] = env_raw_scan
                else:
                    was_training = self.scan_encoder.training
                    self.scan_encoder.eval()
                    latent_scan, _ = self.scan_encoder(env_raw_scan)
                    self.scan_encoder.train(was_training)
                    
                feats.append(latent_scan.detach()) 
                
            env_feat_for_rnn = torch.cat(feats, dim=-1)
            
            if getattr(self, "blind_vision", False):
                env_feat_for_rnn = torch.zeros_like(env_feat_for_rnn)
                
        # -----------------------------------------------------------------
        # 流 3: 深度图 CNN DepthAE
        # -----------------------------------------------------------------
        elif self.use_cnn:
            if not getattr(self, 'has_elevation_input', True) and (obs_dict is None or (isinstance(obs_dict, dict) and "noisy_elevation" not in obs_dict)):
                if self.feed_ae:
                    env_feat_for_rnn = torch.zeros((*x.shape[:-1], self.ae_output_dim), device=x.device)
            else:
                if obs_dict is not None and "noisy_elevation" in obs_dict:
                    env_raw = obs_dict["noisy_elevation"]
                else:
                    env_raw = x[..., proprio_dim:]
                    
                original_shape = env_raw.shape[:-1] 
                depth_image = env_raw.view(*original_shape, self.num_cameras, self.camera_height, self.camera_width)
                if depth_image.ndim == 5: 
                    B, T, C, H, W = depth_image.shape
                    depth_image = depth_image.view(B*T, C, H, W)
                
                if compute_reconstruction and self.training:
                    env_feat_ae, env_recon = self.depth_encoder(depth_image)
                    aux_outputs["depth_recon"] = env_recon      
                    aux_outputs["depth_target"] = depth_image
                else:
                    was_training = self.depth_encoder.training
                    self.depth_encoder.eval()
                    env_feat_ae, _ = self.depth_encoder(depth_image)
                    self.depth_encoder.train(was_training)

                if depth_image.ndim == 5:
                    env_feat_ae = env_feat_ae.view(B, T, -1)
                
                env_feat_for_rnn = env_feat_ae.detach() if self.feed_ae else env_raw
                if self.blind_vision:
                    env_feat_for_rnn = torch.zeros_like(env_feat_for_rnn)
        else:
            if x.shape[-1] > proprio_dim:
                env_feat_for_rnn = x[..., proprio_dim:]
                if self.blind_vision:
                    env_feat_for_rnn = torch.zeros_like(env_feat_for_rnn)

        # -----------------------------------------------------------------
        # 流 4: ProprioVAE (Estimator)
        # -----------------------------------------------------------------
        vae_latent_for_rnn = None
        vel_pred_for_rnn = None
        if self.estimator is not None and obs_dict is not None:
            if self.training or self.feed_estimator:
                est_input = self._get_estimator_input(obs_dict)
                if self.estimator_obs_normalization and self.estimator_obs_normalizer is not None:
                    est_input = self.estimator_obs_normalizer(est_input)
                

                if compute_reconstruction and self.training:
                    vel_pred, recon_x, mu, logvar, z = self.estimator(est_input)
                    aux_outputs["vae_recon"] = recon_x     
                    aux_outputs["vae_mu"] = mu                  
                    aux_outputs["vae_logvar"] = logvar       
                    aux_outputs["vae_vel_pred"] = vel_pred  
                    aux_outputs["vae_input"] = est_input
                else:
                    was_training = self.estimator.training
                    self.estimator.eval()
                    vel_pred, _, mu, logvar, z = self.estimator(est_input)
                    self.estimator.train(was_training)
                
                if self.feed_estimator:
                    vae_latent_for_rnn = z.detach() 
                    vel_pred_for_rnn = vel_pred.detach() 
                
        elif self.estimator is not None and obs_dict is None:
            if self.feed_estimator:
                orig_shape = proprio.shape[:-1]
                vel_pred_for_rnn = torch.zeros((*orig_shape, self.estimator_output_dim), device=proprio.device)
                vae_latent_for_rnn = torch.zeros((*orig_shape, 64), device=proprio.device)

        components = [proprio]
        if self.feed_estimator and self.estimator is not None:
            if vel_pred_for_rnn is not None:
                components.append(vel_pred_for_rnn)
                components.append(vae_latent_for_rnn)
            
        if env_feat_for_rnn is not None:
            components.append(env_feat_for_rnn)

        x_out = torch.cat(components, dim=-1)
        if compute_reconstruction:
            return x_out, aux_outputs
        return x_out
    
    def _prepare_input(self, obs, key_list):
        x = self._extract_raw_obs(obs, key_list)
        return self._process_obs(x, obs_dict=obs, normalizer=None) 

    def _run_rnn(self, rnn_module, x_in, hidden_states, masks):
        if hidden_states is None:
            B = x_in.shape[1] if x_in.ndim == 3 else x_in.shape[0]
            rnn_state = self._init_rnn_state(B, x_in.device)
        else:
            rnn_state = hidden_states

        # 辅助切片函数 (兼容 GRU 和 LSTM)
        def _slice_h(h, start, end):
            if isinstance(h, tuple): return (h[0][start:end], h[1][start:end])
            return h[start:end]
            
        def _cat_h(h1, h2):
            if isinstance(h1, tuple): return (torch.cat([h1[0], h2[0]], dim=0), torch.cat([h1[1], h2[1]], dim=0))
            return torch.cat([h1, h2], dim=0)

        # 判断是否携带有"搭便车"的镜像状态
        is_piggybacked = (rnn_state[0].shape[0] == 2) if isinstance(rnn_state, tuple) else (rnn_state.shape[0] == 2)

        # 仅将 Layer 0 喂给 RNN
        active_h = _slice_h(rnn_state, 0, 1) if is_piggybacked else rnn_state

        if x_in.ndim == 3: 
            rnn_out, next_active_h = rnn_module(x_in, active_h)
            if masks is not None: latent = unpad_trajectories(rnn_out, masks)
            else: latent = rnn_out
        elif x_in.ndim == 2: 
            x_rnn = x_in.unsqueeze(0)
            if masks is not None:
                m = masks.view(1, -1, 1)
                if self.rnn_type == "lstm": active_h = (active_h[0] * m, active_h[1] * m)
                else: active_h = active_h * m
            rnn_out, next_active_h = rnn_module(x_rnn, active_h)
            latent = rnn_out[0]
            
        # 重新组合：最新的真实状态 (Layer 0) + 原封不动的镜像状态 (Layer 1)
        if is_piggybacked:
            next_rnn_state = _cat_h(next_active_h, _slice_h(rnn_state, 1, 2))
        else:
            next_rnn_state = next_active_h
            
        return latent, next_rnn_state

    def _get_estimator_input(self, obs_dict):
        if self.has_estimator_group and (isinstance(obs_dict, dict) or hasattr(obs_dict, "keys")):
            if "estimator" in obs_dict: return obs_dict["estimator"]
        if hasattr(obs_dict, "keys") or isinstance(obs_dict, dict):
            try: full_obs = self._extract_raw_obs(obs_dict, self.input_keys)
            except KeyError: full_obs = self._extract_raw_obs(obs_dict, self.input_keys)
        else: full_obs = obs_dict 
        return full_obs[..., self.estimator_input_indices]

    def _compute_actor_output(self, latent, obs_dict=None, return_aux_loss=False):
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

        with torch.no_grad():
            def flat_mean(w): return w.reshape(-1, w.shape[-1]).mean(dim=0).detach()
            self.latest_weights = {"leg": flat_mean(w_leg), "wheel": flat_mean(w_wheel)}
            
        # 显式返回 Loss，不挂载到 self 上
        if return_aux_loss:
            aux_loss = self._calculate_load_balancing_loss(w_leg, w_wheel) * self.aux_loss_coef
            return total_action, aux_loss

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

    def act(self, obs, masks=None, hidden_state=None, return_aux_loss=False):
        x_raw = self._extract_raw_obs(obs, self.input_keys)
        normalizer = self.actor_obs_normalizer if self.actor_obs_normalization else None
        if return_aux_loss:
            x_in, aux_outputs = self._process_obs(x_raw, obs_dict=obs, normalizer=normalizer, compute_reconstruction=True)
        else:
            x_in = self._process_obs(x_raw, obs_dict=obs, normalizer=normalizer, compute_reconstruction=False)
        
        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_state is None: self.active_hidden_states = next_state
        
        # 根据 flag 接收 aux_loss
        if return_aux_loss:
            mean, lb_loss = self._compute_actor_output(latent, obs_dict=obs, return_aux_loss=True)
            aux_outputs["lb_loss"] = lb_loss
        else:
            mean = self._compute_actor_output(latent, obs_dict=obs)
            
        safe_std = torch.clamp(self.std, min=1e-4)
        self.distribution = torch.distributions.Normal(mean, safe_std)
        
        # 显式返回
        if return_aux_loss:
            return self.distribution.sample(), aux_outputs
            
        return self.distribution.sample()

    def act_inference(self, obs, masks=None, hidden_states=None):
        x_raw = self._extract_raw_obs(obs, self.input_keys)
        normalizer = self.actor_obs_normalizer if self.actor_obs_normalization else None
        x_in = self._process_obs(x_raw, obs_dict=obs, normalizer=normalizer, compute_reconstruction=False)

        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_states is None: self.active_hidden_states = next_state
        return self._compute_actor_output(latent)

    def evaluate(self, obs, masks=None, hidden_state=None):
        x_raw = self._extract_raw_obs(obs, getattr(self, "critic_keys", self.input_keys))
        normalizer = self.critic_obs_normalizer if self.critic_obs_normalization else None
        x_in = self._process_obs(x_raw, obs_dict=obs, normalizer=normalizer, proprio_dim=self.critic_proprio_dim, compute_reconstruction=False)

        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_critic_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.critic_rnn, x_in, current_state, masks)
        if hidden_state is None: self.active_critic_hidden_states = next_state
        return self.critic_mlp(latent)
    
    def get_estimated_state(self, obs, masks=None, hidden_states=None):
        if self.estimator is None: return None
        est_input = self._get_estimator_input(obs)
        if self.estimator_obs_normalization and self.estimator_obs_normalizer is not None:
             est_input = self.estimator_obs_normalizer(est_input)
        vel_pred, _, _, _, _ = self.estimator(est_input)
        return vel_pred

    def forward(self, obs, masks=None, hidden_states=None, save_dist=True):
        x_raw = self._extract_raw_obs(obs, self.input_keys)
        normalizer = self.actor_obs_normalizer if self.actor_obs_normalization else None
        x_in = self._process_obs(x_raw, obs_dict=obs, normalizer=normalizer)

        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        action_mean = self._compute_actor_output(latent, obs_dict=obs)
        
        safe_std = torch.clamp(self.std, min=1e-4)
        if save_dist: self.distribution = torch.distributions.Normal(action_mean, safe_std)
        return action_mean, safe_std, next_state

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
                
            if hasattr(self, "_prepare_hidden_state") and hasattr(self, "std"):
                 self.active_hidden_states = self._prepare_hidden_state(actor_hs, self.std.device)
                 self.active_critic_hidden_states = self._prepare_hidden_state(critic_hs, self.std.device)
            else:
                 self.active_hidden_states = actor_hs
                 self.active_critic_hidden_states = critic_hs

        if dones is None: return

        if hasattr(dones, "dtype") and dones.dtype == torch.uint8:
            dones = dones.bool()

        def reset_hidden(h, mask):
            if isinstance(h, tuple): return tuple(reset_hidden(x, mask) for x in h)
            else: 
                h = h.clone()
                h[:, mask, :] = 0.0
                return h
                
        if self.active_hidden_states is not None:
            self.active_hidden_states = reset_hidden(self.active_hidden_states, dones)
        if self.active_critic_hidden_states is not None:
            self.active_critic_hidden_states = reset_hidden(self.active_critic_hidden_states, dones)

# ==============================================================================
# 4. 蒸馏适配器 (Student-Teacher Adapter for Distillation)
# ==============================================================================

class SplitMoEStudentTeacher(nn.Module):
    is_recurrent = True
    loaded_teacher = False

    def __init__(self, obs, obs_groups, num_actions, activation="elu", **kwargs):
        super().__init__()
        self.obs_groups = obs_groups
        if isinstance(activation, dict): activation = activation.get("value", "elu") if "value" in activation else "elu"
        teacher_is_mlp = kwargs.pop("teacher_is_mlp", False) 

        print("\n[Distillation] Initializing Student Policy...")
        student_kwargs = kwargs.copy()
        student_obs_groups = {"policy": obs_groups["policy"]}
        student_obs_groups["critic"] = obs_groups["policy"]
        student_input_key = obs_groups["policy"][0]

        self.student = SplitMoEActorCritic(
            obs, student_obs_groups, num_actions, forced_input_key=student_input_key, 
            activation=activation, is_student_mode=True, **student_kwargs
        )

        teacher_source = obs_groups["teacher"]
        if teacher_is_mlp:
            print("\n[Distillation] Initializing Teacher Policy (Standard MLP)...")
            teacher_obs_groups = {"policy": teacher_source, "critic": teacher_source}
            self.teacher = ActorCritic(
                obs=obs, obs_groups=teacher_obs_groups, num_actions=num_actions,
                actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                activation=activation, init_noise_std=1.0,
            )
        else:
            print("\n[Distillation] Initializing Teacher Policy (SplitMoE)...")
            teacher_kwargs = kwargs.copy()
            teacher_kwargs["feed_estimator_to_policy"] = False
            teacher_kwargs["feed_ae_to_policy"] = False
            teacher_obs_groups = {"policy": teacher_source, "critic": teacher_source}
            teacher_input_key = teacher_source[0]
            
            self.teacher = SplitMoEActorCritic(
                obs, teacher_obs_groups, num_actions, forced_input_key=teacher_input_key, 
                activation=activation, **teacher_kwargs
            )
            
        self.teacher.eval()
        for param in self.teacher.parameters(): param.requires_grad = False

    @property
    def action_std(self): return self.student.std

    def reset(self, dones=None, hidden_states=None):
        self.student.reset(dones, hidden_states=hidden_states)
        self.teacher.reset(dones)

    def act(self, obs, masks=None, hidden_state=None): return self.student.act(obs, masks, hidden_state)
    def act_inference(self, obs): return self.student.act_inference(obs)
    def evaluate(self, obs):
        with torch.no_grad(): return self.teacher.act_inference(obs)
    def get_hidden_states(self): return self.student.get_hidden_states()
    
    def detach_hidden_states(self, dones=None):
        if dones is not None and hasattr(dones, "dtype") and dones.dtype == torch.uint8: dones = dones.bool()
        def detach_recursive(h, mask):
            if isinstance(h, tuple): return tuple(detach_recursive(x, mask) for x in h)
            h = h.detach() 
            if mask is not None:
                h = h.clone()
                h[:, mask, :] = 0.0
            return h
        if self.student.active_hidden_states is not None:
             self.student.active_hidden_states = detach_recursive(self.student.active_hidden_states, dones)
        if hasattr(self.student, "active_critic_hidden_states") and self.student.active_critic_hidden_states is not None:
             self.student.active_critic_hidden_states = detach_recursive(self.student.active_critic_hidden_states, dones)

    def update_normalization(self, obs): self.student.update_normalization(obs)
    
    def load_state_dict(self, state_dict, strict=True):
        print(f"[Distillation] Loading state dict with {len(state_dict)} keys...")
        teacher_state_dict = {k: v for k, v in state_dict.items() if "critic" not in k}
        try:
            self.teacher.load_state_dict(teacher_state_dict, strict=False)
            print("[Distillation] Successfully loaded Teacher weights (critic ignored).")
            self.loaded_teacher = True
        except Exception as e:
            print(f"[Distillation] Warning: Direct load to teacher failed: {e}")

        if not any(k.startswith("student.") for k in state_dict.keys()):
            print("[Distillation] Initializing Student's VAE/AE from Teacher...")
            student_dict = self.student.state_dict()
            for k, v in teacher_state_dict.items():
                if k.startswith("estimator") or k.startswith("elevation_encoder") or k.startswith("depth_encoder") or k.startswith("scan_encoder"):
                    if k in student_dict and student_dict[k].shape == v.shape:
                        student_dict[k].copy_(v)
            self.student.load_state_dict(student_dict, strict=False)
            print("[Distillation] VAE/AE weights copied to Student successfully.")
        else:
            super().load_state_dict(state_dict, strict=strict)
            print("[Distillation] Resumed Student-Teacher training.")
            self.loaded_teacher = True
        return True

# ==============================================================================
# 5. PPO Algorithm Wrapper (Teacher Training / Composite Loss Calc)
# ==============================================================================

class SplitMoEPPO(PPO):
    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        
        # 自定义 Loss 的追踪
        avg_lb_loss = 0.0
        avg_sym_loss_lr = 0.0
        avg_sym_loss_fb = 0.0
        avg_elev_ae_loss = 0.0
        avg_scan_ae_loss = 0.0
        avg_depth_ae_loss = 0.0
        
        avg_vel_loss = 0.0
        avg_recon_loss = 0.0
        avg_kl_loss = 0.0
        
        model = getattr(self, "actor_critic", getattr(self, "policy", None))

        if model is None or self.num_learning_epochs == 0:
            return {}

        # 1. 获取生成器 
        if model.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 2. 预定义对称性硬编码字典 (避免在循环中重复计算)
        device = self.device
        # L-R (左右) 镜像: FL↔FR, HL↔HR.  hipx 取反, hipy/knee 不取反, wheels 不取反.
        action_swap_idx = torch.tensor([3,4,5, 0,1,2, 9,10,11, 6,7,8, 13,12, 15,14], dtype=torch.long, device=device)
        action_neg_mask = torch.tensor([-1,1,1, -1,1,1, -1,1,1, -1,1,1, 1, 1, 1, 1], dtype=torch.float32, device=device)
        # F-B (前后) 镜像: FL↔HL, FR↔HR, wheel_FL↔wheel_HL, wheel_FR↔wheel_HR.
        # hipx 不取反 (左右 side 不变, abduction convention 不变).
        # hipy/knee 取反 (URDF 前后 axis 镜像, 见 deeprobotics.py init_state: f_hipy=-0.6, h_hipy=+0.6 等).
        # wheels 取反 (rolling 方向在 body-X 反转后反向).
        action_swap_idx_fb = torch.tensor([6,7,8, 9,10,11, 0,1,2, 3,4,5, 14,15, 12,13], dtype=torch.long, device=device)
        action_neg_mask_fb = torch.tensor([1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, -1,-1,-1,-1], dtype=torch.float32, device=device)

        batch_cnt = 0

        # =====================================================================
        # 全新的单趟融合循环 (Single-Pass Merged Loop)
        # =====================================================================
        for (obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, 
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch) in generator:
            
            self.optimizer.zero_grad()
            original_batch_size = obs_batch.batch_size[0] if hasattr(obs_batch, 'batch_size') else obs_batch.shape[0]

            if getattr(self, "normalize_advantage_per_mini_batch", False):
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # -----------------------------------------------------------------
            # STEP A: PPO 原生动作重评估
            # -----------------------------------------------------------------
            _, aux_outputs = model.act(obs_batch, masks=masks_batch, hidden_state=hid_states_batch[0], return_aux_loss=True)
            actions_log_prob_batch = model.get_actions_log_prob(actions_batch)
            value_batch = model.evaluate(obs_batch, masks=masks_batch, hidden_state=hid_states_batch[1])
            
            mu_batch = model.distribution.mean
            sigma_batch = model.distribution.stddev
            entropy_batch = model.distribution.entropy().sum(dim=-1)
            if getattr(self, "desired_kl", None) is not None and getattr(self, "schedule", "") == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if getattr(self, "is_multi_gpu", False):
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if getattr(self, "gpu_global_rank", 0) == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if getattr(self, "is_multi_gpu", False):
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # PPO 主 Loss
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # -----------------------------------------------------------------
            # STEP B: 注入 MoE Load Balancing Loss
            # -----------------------------------------------------------------
            lb_loss = aux_outputs.get("lb_loss")
            if lb_loss is not None:
                loss = loss + lb_loss 
                avg_lb_loss += lb_loss.item()

            # -----------------------------------------------------------------
            # STEP C: 注入自编码器 & 变分自编码器 Loss
            # -----------------------------------------------------------------
            if not getattr(model, "is_student_mode", False):
                
                # 将 masks 展平为一维布尔掩码，过滤掉 RNN sequence 结尾的 Padding (零)
                valid_mask = masks_batch.flatten().bool()

                # 1. ProprioVAE Loss
                if "vae_recon" in aux_outputs:
                    # 使用 .flatten(0, 1)[valid_mask] 只提取有效的非零帧
                    vel_pred = aux_outputs["vae_vel_pred"].flatten(0, 1)[valid_mask]
                    recon_x = aux_outputs["vae_recon"].flatten(0, 1)[valid_mask]
                    mu = aux_outputs["vae_mu"].flatten(0, 1)[valid_mask]
                    logvar = aux_outputs["vae_logvar"].flatten(0, 1)[valid_mask]
                    est_input = aux_outputs["vae_input"].flatten(0, 1)[valid_mask]
                    
                    if hasattr(obs_batch, "keys"):
                        full_obs_for_target = obs_batch.get("critic", model._extract_raw_obs(obs_batch, getattr(model, "critic_keys", model.input_keys)))
                    else:
                        full_obs_for_target = obs_batch
                        
                    target_state = full_obs_for_target[..., model.estimator_target_indices].flatten(0, 1)[valid_mask]

                    vel_loss = (vel_pred - target_state).pow(2).mean()
                    recon_loss = (recon_x - est_input).pow(2).mean()
                    logvar = torch.clamp(logvar, min=-20.0, max=10.0)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                    
                    vae_loss = vel_loss + recon_loss + 0.01 * kl_loss
                    loss = loss + vae_loss
                    
                    avg_vel_loss += vel_loss.item()
                    avg_recon_loss += recon_loss.item()
                    avg_kl_loss += kl_loss.item()

                # 2. ElevationAE Loss
                if "elev_recon" in aux_outputs:
                    env_recon = aux_outputs["elev_recon"].flatten(0, 1)[valid_mask]
                    env_raw_elev = aux_outputs["elev_target"].flatten(0, 1)[valid_mask]
                    
                    elev_ae_loss = (env_recon - env_raw_elev).pow(2).mean()
                    loss = loss + elev_ae_loss
                    avg_elev_ae_loss += elev_ae_loss.item()

                # 3. MultiLayer Scan Loss
                if "scan_recon" in aux_outputs:
                    scan_recon = aux_outputs["scan_recon"].flatten(0, 1)[valid_mask]
                    env_raw_scan = aux_outputs["scan_target"].flatten(0, 1)[valid_mask]
                    
                    scan_ae_loss = (scan_recon - env_raw_scan).pow(2).mean()
                    loss = loss + scan_ae_loss
                    avg_scan_ae_loss += scan_ae_loss.item()
                        
                # 4. DepthAE Loss (CNN)
                if "depth_recon" in aux_outputs:
                    depth_recon = aux_outputs["depth_recon"]
                    depth_image = aux_outputs["depth_target"]
                    
                    depth_ae_loss = (depth_recon - depth_image).pow(2).mean()
                    loss = loss + depth_ae_loss
                    avg_depth_ae_loss += depth_ae_loss.item()
                        
            # -----------------------------------------------------------------
            # STEP D: 对称性正则化 Symmetry Loss (sym_enabled 关闭时整段跳过)
            # -----------------------------------------------------------------
            if not getattr(model, "sym_enabled", True):
                loss.backward()

                if getattr(self, "is_multi_gpu", False):
                    self.reduce_parameters()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_batch.mean().item()
                batch_cnt += 1
                continue

            obs_mirrored_batch = self._mirror_obs(obs_batch)

            current_mean = model.distribution.mean.detach()
            target_mirrored_actions = current_mean[..., action_swap_idx] * action_neg_mask

            actor_hid_batch = hid_states_batch[0]
            if isinstance(actor_hid_batch, tuple):
                mirrored_h_batch = (actor_hid_batch[0][1:2], actor_hid_batch[1][1:2])
            else:
                mirrored_h_batch = actor_hid_batch[1:2]
            # # =================================================================
            # # DEBUG 1: 检查隐藏状态切片是否有效 (只在 epoch 刚开始时打印)
            # # =================================================================
            # if batch_cnt == 0 and getattr(self, "gpu_global_rank", 0) == 0:
            #     print("\n" + "▼"*50)
            #     print("[DEBUG 1] 正在检查镜像隐藏状态 (Layer 1)")
            #     if isinstance(actor_hid_batch, tuple):
            #         print(f"-> 原始 hid_states_batch h shape: {actor_hid_batch[0].shape}")
            #         max_abs_h = mirrored_h_batch[0].abs().max().item()
            #         max_abs_c = mirrored_h_batch[1].abs().max().item()
            #         print(f"-> 切片后 mirrored_h_batch h 最大绝对值: {max_abs_h:.6f}")
            #         print(f"-> 切片后 mirrored_h_batch c 最大绝对值: {max_abs_c:.6f}")
            #         if max_abs_h == 0.0:
            #             print("!! 警告: 提取的隐藏状态 h 全是 0，说明 Layer 1 未在 act() 中被正确推进 !!")
            #     else:
            #         print(f"-> 原始 hid_states_batch shape: {actor_hid_batch.shape}")
            #         max_abs = mirrored_h_batch.abs().max().item()
            #         print(f"-> 切片后 mirrored_h_batch 最大绝对值: {max_abs:.6f}")
            #         if max_abs == 0.0:
            #             print("!! 警告: 提取的隐藏状态全是 0，说明 Layer 1 未在 act() 中被正确推进 !!")
            #     print("▲"*50 + "\n")
            # # =================================================================
            # # DEBUG 2: 检查 _mirror_obs 的硬编码维度映射 (已修复 TensorDict)
            # # =================================================================
            # if batch_cnt == 0 and getattr(self, "gpu_global_rank", 0) == 0:
            #     print("\n" + "▼"*50)
            #     print("[DEBUG 2] 正在检查观测张量的镜像映射")
                
            #     # 安全解包 TensorDict，获取纯粹的张量
            #     if hasattr(obs_batch, "keys"):
            #         # 兼容 PPO generator 返回的可能形状: [seq_len, batch_size, dim] 或 [batch_size, dim]
            #         p_orig = obs_batch["policy"][0, 0] if obs_batch["policy"].ndim == 3 else obs_batch["policy"][0]
            #         p_mirr = obs_mirrored_batch["policy"][0, 0] if obs_mirrored_batch["policy"].ndim == 3 else obs_mirrored_batch["policy"][0]
            #     else:
            #         p_orig = obs_batch[0, 0] if obs_batch.ndim == 3 else obs_batch[0]
            #         p_mirr = obs_mirrored_batch[0, 0] if obs_mirrored_batch.ndim == 3 else obs_mirrored_batch[0]
                
            #     print(f"-> Policy 观测总维度: {p_orig.shape[-1]}")
                
            #     # 推断 offset 
            #     has_lin_vel = p_orig.shape[-1] > 57
            #     print(f"-> 代码推断包含线速度 (has_lin_vel): {has_lin_vel}")
                
            #     if has_lin_vel:
            #         print(f"  [测试线速度 Y] 原值: {p_orig[1].item():.4f} -> 镜像值: {p_mirr[1].item():.4f} (应互为相反数)")
            #         print(f"  [测试角速度 X] 原值: {p_orig[3].item():.4f} -> 镜像值: {p_mirr[3].item():.4f} (应互为相反数)")
            #         offset = 12
            #     else:
            #         print(f"  [测试角速度 X] 原值: {p_orig[0].item():.4f} -> 镜像值: {p_mirr[0].item():.4f} (应互为相反数)")
            #         offset = 9
                
            #     if p_orig.shape[-1] > offset + 3:
            #         orig_lf_hip = p_orig[offset + 0].item()
            #         orig_rf_hip = p_orig[offset + 3].item()
            #         mirr_lf_hip = p_mirr[offset + 0].item()
            #         print(f"  [测试关节位移] 原 LF_hip (idx {offset}): {orig_lf_hip:.4f}")
            #         print(f"  [测试关节位移] 原 RF_hip (idx {offset+3}): {orig_rf_hip:.4f}")
            #         print(f"  [测试关节位移] 镜像后的 LF_hip 应该是原 RF_hip 的相反数 (-{orig_rf_hip:.4f})")
            #         print(f"  [测试关节位移] 实际镜像后 LF_hip 值为: {mirr_lf_hip:.4f}")
                    
            #         if abs(mirr_lf_hip - (-orig_rf_hip)) > 1e-4:
            #             print("!! 警告: 镜像映射数值对不上，硬编码 offset 或 action_swap_idx 写错了 !!")
            #         else:
            #             print(">> 测试通过: 镜像映射的 offset 和索引完全正确！")
            #     print("▲"*50 + "\n")
            # # =================================================================
            pred_actions = model.act_inference(
                obs_mirrored_batch, 
                masks=masks_batch, 
                hidden_states=mirrored_h_batch
            )

            sym_loss = torch.nn.functional.mse_loss(pred_actions, target_mirrored_actions)
            loss = loss + getattr(model, 'sym_loss_coef', 1.0) * sym_loss
            avg_sym_loss_lr += sym_loss.item()

            # -----------------------------------------------------------------
            # STEP D2: F-B (前后) 对称性正则化 Symmetry Loss
            # 用 layer 0 (real) hidden state 作为起点 — 等变收敛时 M_FB(h) ≈ h_FB,
            # 所以 policy(M_FB(obs), h_real) ≈ M_FB(policy(obs, h_real)) 这个约束是合理的.
            # -----------------------------------------------------------------
            obs_mirrored_fb_batch = self._mirror_obs_fb(obs_batch)
            target_mirrored_fb_actions = current_mean[..., action_swap_idx_fb] * action_neg_mask_fb

            if isinstance(actor_hid_batch, tuple):
                real_h_batch = (actor_hid_batch[0][0:1], actor_hid_batch[1][0:1])
            else:
                real_h_batch = actor_hid_batch[0:1]

            pred_fb_actions = model.act_inference(
                obs_mirrored_fb_batch,
                masks=masks_batch,
                hidden_states=real_h_batch
            )

            sym_loss_fb = torch.nn.functional.mse_loss(pred_fb_actions, target_mirrored_fb_actions)
            loss = loss + getattr(model, 'sym_loss_coef', 1.0) * sym_loss_fb
            avg_sym_loss_fb += sym_loss_fb.item()

            # -----------------------------------------------------------------
            # STEP E: 终极单趟反向传播
            # -----------------------------------------------------------------
            loss.backward()
            
            if getattr(self, "is_multi_gpu", False):
                self.reduce_parameters()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            # # =================================================================
            # # DEBUG 3: 检查专家网络的梯度健康度 (确保没有死神经元)
            # # =================================================================
            # if batch_cnt == 0 and getattr(self, "gpu_global_rank", 0) == 0:
            #     print("\n" + "▼"*50)
            #     print("[DEBUG 3] MoE 专家网络梯度流动检查")
                
            #     leg_grads = []
            #     for i, expert in enumerate(model.actor_leg_experts):
            #         grad_norm = sum(p.grad.norm().item() for p in expert.parameters() if p.grad is not None)
            #         leg_grads.append(grad_norm)
                    
            #     wheel_grads = []
            #     for i, expert in enumerate(model.actor_wheel_experts):
            #         grad_norm = sum(p.grad.norm().item() for p in expert.parameters() if p.grad is not None)
            #         wheel_grads.append(grad_norm)
                
            #     print(f"-> 腿部专家 (Leg) 各自梯度范数: {[round(g, 4) for g in leg_grads]}")
            #     print(f"-> 轮部专家 (Wheel) 各自梯度范数: {[round(g, 4) for g in wheel_grads]}")
                
            #     if any(g == 0.0 for g in leg_grads + wheel_grads):
            #         print("!! 警告: 发现梯度为 0 的死专家 (Dead Expert) !!")
            #     else:
            #         print(">> 测试通过: 所有专家都在积极学习！")
            #     print("▲"*50 + "\n")
            # # =================================================================
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            batch_cnt += 1

        # =====================================================================
        # 日志输出整理
        # =====================================================================
        if batch_cnt > 0:
            mean_value_loss /= batch_cnt
            mean_surrogate_loss /= batch_cnt
            mean_entropy /= batch_cnt
            avg_lb_loss /= batch_cnt
            avg_sym_loss_lr /= batch_cnt
            avg_sym_loss_fb /= batch_cnt
            avg_elev_ae_loss /= batch_cnt
            avg_scan_ae_loss /= batch_cnt
            avg_vel_loss /= batch_cnt
            avg_recon_loss /= batch_cnt
            avg_kl_loss /= batch_cnt
            
        self.storage.clear()
        
        # 构建基础的 loss_dict
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }

        # 填入附加 Loss
        if not getattr(model, "is_student_mode", False):
            loss_dict["Loss/Load_Balancing"] = avg_lb_loss
            loss_dict["Loss/Actor_Symmetry_Reg_LR"] = avg_sym_loss_lr
            loss_dict["Loss/Actor_Symmetry_Reg_FB"] = avg_sym_loss_fb
            loss_dict["Loss/Actor_Symmetry_Reg"] = avg_sym_loss_lr + avg_sym_loss_fb
            if model.estimator is not None:
                loss_dict["Loss/VAE_Vel_MSE"] = avg_vel_loss
                loss_dict["Loss/VAE_Recon_MSE"] = avg_recon_loss
                loss_dict["Loss/VAE_KL"] = avg_kl_loss
                
            if getattr(model, "use_elevation_ae", False):
                loss_dict["Loss/Elevation_AE_Recon"] = avg_elev_ae_loss
            if getattr(model, "use_multilayer_scan", False):
                loss_dict["Loss/Scan_AE_Recon"] = avg_scan_ae_loss
            if getattr(model, "use_cnn", False):
                loss_dict["Loss/Depth_AE_Recon"] = avg_depth_ae_loss
                
        # 记录 MoE 选通门行为
        if model is not None:
            if hasattr(model, "latest_weights") and model.latest_weights:
                w = model.latest_weights
                if "leg" in w:
                    for i, val in enumerate(w["leg"]): loss_dict[f"Gate/Leg_Expert_{i}"] = val.item()
                if "wheel" in w:
                    for i, val in enumerate(w["wheel"]): loss_dict[f"Gate/Wheel_Expert_{i}"] = val.item()
            if hasattr(model, "std"):
                std_np = model.std.detach().cpu().numpy()
                n_legs = getattr(model, "num_leg_actions", 12)
                if len(std_np) >= n_legs:
                    loss_dict["Noise/Leg_Std"] = std_np[:n_legs].mean()
                    loss_dict["Noise/Wheel_Std"] = std_np[n_legs:].mean() if len(std_np) > n_legs else 0.0

        return loss_dict
    
    def _mirror_obs(self, obs):
        device = self.device
        model = getattr(self, "actor_critic", getattr(self, "policy", None))
        
        action_swap_idx = torch.tensor([3,4,5, 0,1,2, 9,10,11, 6,7,8, 13,12, 15,14], dtype=torch.long, device=device)
        action_neg_mask = torch.tensor([-1,1,1, -1,1,1, -1,1,1, -1,1,1, 1, 1, 1, 1], dtype=torch.float32, device=device)
        
        obs_swap_idx = torch.arange(57, dtype=torch.long, device=device)
        obs_neg_mask = torch.ones(57, dtype=torch.float32, device=device)
        obs_neg_mask[0], obs_neg_mask[2], obs_neg_mask[4], obs_neg_mask[7], obs_neg_mask[8] = -1.0, -1.0, -1.0, -1.0, -1.0
        for offset in [9, 25, 41]: 
            obs_swap_idx[offset:offset+16] = action_swap_idx + offset
            obs_neg_mask[offset:offset+16] = action_neg_mask

        # 使用深拷贝或新建字典，对 Tensor 显式 .clone()
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            obs_mirrored = {}
            for k, v in obs.items():
                obs_mirrored[k] = v.clone()
        else:
            obs_mirrored = obs.clone()
        
        if isinstance(obs_mirrored, dict):
            if "policy" in obs_mirrored:
                # 关键修复：不在原地修改，而是新建张量并赋值
                p = obs_mirrored["policy"]
                obs_mirrored["policy"] = (p[..., obs_swap_idx] * obs_neg_mask).clone()
                
            if "estimator" in obs_mirrored:
                e = obs_mirrored["estimator"]
                history_len = e.shape[-1] // 57
                est_neg_mask = obs_neg_mask.repeat(history_len)
                est_swap_idx = torch.zeros(e.shape[-1], dtype=torch.long, device=device)
                for h_i in range(history_len):
                    est_swap_idx[h_i*57 : (h_i+1)*57] = obs_swap_idx + h_i*57
                obs_mirrored["estimator"] = (e[..., est_swap_idx] * est_neg_mask).clone()
                
            if "noisy_elevation" in obs_mirrored:
                env_raw = obs_mirrored["noisy_elevation"].clone() # 不污染原图
                if getattr(model, "use_elevation_ae", False):
                    elev = env_raw[..., :model.elevation_dim]
                    elev_mirrored = elev.view(*elev.shape[:-1], 11, 17).flip(dims=[-2]).view(*elev.shape[:-1], model.elevation_dim)
                    env_raw[..., :model.elevation_dim] = elev_mirrored
                if getattr(model, "use_multilayer_scan", False):
                    scan_start = model.elevation_dim if getattr(model, "use_elevation_ae", False) else 0
                    scan = env_raw[..., scan_start : scan_start + model.scan_dim]
                    scan_mirrored = scan.view(*scan.shape[:-1], model.num_scan_channels, model.num_scan_rays).flip(dims=[-1]).view(*scan.shape[:-1], model.scan_dim)
                    env_raw[..., scan_start : scan_start + model.scan_dim] = scan_mirrored
                obs_mirrored["noisy_elevation"] = env_raw
        else:
            obs_mirrored = (obs_mirrored[..., obs_swap_idx] * obs_neg_mask).clone()

        return obs_mirrored

    def _mirror_obs_fb(self, obs):
        """前后 (F-B) 镜像: X 轴反射, FL↔HL / FR↔HR / 前 lidar ↔ 后 lidar.

        Layout (与 _mirror_obs 保持一致):
            policy 57-d   = [w_x, w_y, w_z, g_x, g_y, g_z, vx, vy, wz, joint_pos(16), joint_vel(16), last_act(16)]
            estimator     = policy obs × history
            noisy_elev    = elevation(187, 11×17 grid) ⊕ scan(252, 12 ch × 21 ray)
                            scan channels = [fwd_l0..l5, bwd_l0..l5]

        F-B 约束:
            prefix flips : w_y, w_z, g_x, vx_cmd, wz_cmd
            joint blocks : action_swap_idx_fb + action_neg_mask_fb (FL↔HL 等)
            elevation    : flip(dim=-1)  (17 列 = X 轴 = 前后)
            scan         : 前 6 channels ↔ 后 6 channels (整段互换), ray 不翻
        """
        device = self.device
        model = getattr(self, "actor_critic", getattr(self, "policy", None))

        action_swap_idx = torch.tensor([6,7,8, 9,10,11, 0,1,2, 3,4,5, 14,15, 12,13], dtype=torch.long, device=device)
        action_neg_mask = torch.tensor([1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, -1,-1,-1,-1], dtype=torch.float32, device=device)

        obs_swap_idx = torch.arange(57, dtype=torch.long, device=device)
        obs_neg_mask = torch.ones(57, dtype=torch.float32, device=device)
        obs_neg_mask[1], obs_neg_mask[2], obs_neg_mask[3], obs_neg_mask[6], obs_neg_mask[8] = -1.0, -1.0, -1.0, -1.0, -1.0
        for offset in [9, 25, 41]:
            obs_swap_idx[offset:offset+16] = action_swap_idx + offset
            obs_neg_mask[offset:offset+16] = action_neg_mask

        if isinstance(obs, dict) or hasattr(obs, "keys"):
            obs_mirrored = {}
            for k, v in obs.items():
                obs_mirrored[k] = v.clone()
        else:
            obs_mirrored = obs.clone()

        if isinstance(obs_mirrored, dict):
            if "policy" in obs_mirrored:
                p = obs_mirrored["policy"]
                obs_mirrored["policy"] = (p[..., obs_swap_idx] * obs_neg_mask).clone()

            if "estimator" in obs_mirrored:
                e = obs_mirrored["estimator"]
                history_len = e.shape[-1] // 57
                est_neg_mask = obs_neg_mask.repeat(history_len)
                est_swap_idx = torch.zeros(e.shape[-1], dtype=torch.long, device=device)
                for h_i in range(history_len):
                    est_swap_idx[h_i*57 : (h_i+1)*57] = obs_swap_idx + h_i*57
                obs_mirrored["estimator"] = (e[..., est_swap_idx] * est_neg_mask).clone()

            if "noisy_elevation" in obs_mirrored:
                env_raw = obs_mirrored["noisy_elevation"].clone()
                if getattr(model, "use_elevation_ae", False):
                    elev = env_raw[..., :model.elevation_dim]
                    elev_mirrored = elev.view(*elev.shape[:-1], 11, 17).flip(dims=[-1]).reshape(*elev.shape[:-1], model.elevation_dim)
                    env_raw[..., :model.elevation_dim] = elev_mirrored
                if getattr(model, "use_multilayer_scan", False):
                    scan_start = model.elevation_dim if getattr(model, "use_elevation_ae", False) else 0
                    scan = env_raw[..., scan_start : scan_start + model.scan_dim]
                    n_half = model.num_scan_channels // 2
                    chan_swap = torch.cat([
                        torch.arange(n_half, model.num_scan_channels, device=device),
                        torch.arange(0, n_half, device=device),
                    ]).long()
                    scan_view = scan.view(*scan.shape[:-1], model.num_scan_channels, model.num_scan_rays)
                    scan_mirrored = scan_view[..., chan_swap, :].reshape(*scan.shape[:-1], model.scan_dim)
                    env_raw[..., scan_start : scan_start + model.scan_dim] = scan_mirrored
                obs_mirrored["noisy_elevation"] = env_raw
        else:
            obs_mirrored = (obs_mirrored[..., obs_swap_idx] * obs_neg_mask).clone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        model = getattr(self, "actor_critic", getattr(self, "policy", None))

        # =====================================================================
        # 快速路径: sym 未启用时, 单 B 前向, 不做任何镜像 piggyback.
        # =====================================================================
        if not getattr(model, "sym_enabled", True):
            if model.is_recurrent:
                self.transition.hidden_states = model.get_hidden_states()

            was_training = model.training
            model.eval()
            with torch.no_grad():
                action = model.act(obs).detach()
                value = model.evaluate(obs).detach()
            model.train(was_training)

            self.transition.actions = action
            self.transition.values = value
            self.transition.actions_log_prob = model.get_actions_log_prob(action).detach()
            self.transition.action_mean = model.action_mean.detach()
            self.transition.action_sigma = model.action_std.detach()
            self.transition.observations = obs
            return self.transition.actions

        # 辅助函数 (兼容 GRU 和 LSTM hidden state)
        def _slice_layer(h, start, end):
            if isinstance(h, tuple): return (h[0][start:end], h[1][start:end])
            return h[start:end]

        def _cat_layer(h1, h2):
            if isinstance(h1, tuple): return (torch.cat([h1[0], h2[0]], dim=0), torch.cat([h1[1], h2[1]], dim=0))
            return torch.cat([h1, h2], dim=0)

        def _cat_batch(h1, h2):
            if isinstance(h1, tuple): return (torch.cat([h1[0], h2[0]], dim=1), torch.cat([h1[1], h2[1]], dim=1))
            return torch.cat([h1, h2], dim=1)

        def _split_batch(h, B):
            if isinstance(h, tuple): return (h[0][:, :B], h[1][:, :B]), (h[0][:, B:], h[1][:, B:])
            return h[:, :B], h[:, B:]

        # =====================================================================
        # 1. 保存隐状态到 Transition (供 Rollout Storage 使用)
        # =====================================================================
        if model.is_recurrent:
            self.transition.hidden_states = model.get_hidden_states()

        dev = obs["policy"].device if isinstance(obs, dict) else obs.device
        B = obs["policy"].shape[0] if isinstance(obs, dict) else obs.shape[0]

        # =====================================================================
        # 2. 准备镜像观测，沿 batch 维拼接 [real, mirror]
        # =====================================================================
        obs_mirrored = self._mirror_obs(obs)
        batched_obs = {}
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            for k in obs.keys():
                batched_obs[k] = torch.cat([obs[k], obs_mirrored[k]], dim=0)
        else:
            batched_obs = torch.cat([obs, obs_mirrored], dim=0)

        # =====================================================================
        # 3. 拆解 piggybacked 隐状态 → 沿 batch 维合并用于单次 forward
        #    piggybacked: (2, B, H) → h_real=(1,B,H), h_mirror=(1,B,H)
        #    batched:     (1, 2B, H)
        # =====================================================================
        current_h = model._prepare_hidden_state(model.active_hidden_states, dev)
        h_real = _slice_layer(current_h, 0, 1)
        h_mirror = _slice_layer(current_h, 1, 2)
        h_batched = _cat_batch(h_real, h_mirror)

        current_c = model._prepare_hidden_state(model.active_critic_hidden_states, dev)
        has_critic = current_c is not None
        if has_critic:
            c_real = _slice_layer(current_c, 0, 1)
            c_mirror = _slice_layer(current_c, 1, 2)
            c_batched = _cat_batch(c_real, c_mirror)

        was_training = model.training
        model.eval()

        with torch.no_grad():
            # =============================================================
            # 4. 单次 batched Actor 前向 (real + mirror 合并)
            # =============================================================
            x_raw_a = model._extract_raw_obs(batched_obs, model.input_keys)
            normalizer_a = model.actor_obs_normalizer if model.actor_obs_normalization else None
            x_in_a = model._process_obs(x_raw_a, obs_dict=batched_obs, normalizer=normalizer_a)

            rnn_out_a, next_h_batched = model.rnn(x_in_a.unsqueeze(0), h_batched)
            latent_all = rnn_out_a[0]

            latent_real = latent_all[:B]
            next_h_real, next_h_mirror = _split_batch(next_h_batched, B)
            model.active_hidden_states = _cat_layer(next_h_real, next_h_mirror)

            mean = model._compute_actor_output(latent_real, obs_dict=obs)
            safe_std = torch.clamp(model.std, min=1e-4)
            model.distribution = torch.distributions.Normal(mean, safe_std)

            # =============================================================
            # 5. 单次 batched Critic 前向 (real + mirror 合并)
            # =============================================================
            critic_keys = getattr(model, "critic_keys", model.input_keys)
            x_raw_c = model._extract_raw_obs(batched_obs, critic_keys)
            normalizer_c = model.critic_obs_normalizer if model.critic_obs_normalization else None
            x_in_c = model._process_obs(
                x_raw_c, obs_dict=batched_obs,
                normalizer=normalizer_c, proprio_dim=model.critic_proprio_dim
            )

            if has_critic:
                rnn_out_c, next_c_batched = model.critic_rnn(x_in_c.unsqueeze(0), c_batched)
                critic_latent_real = rnn_out_c[0][:B]
                next_c_real, next_c_mirror = _split_batch(next_c_batched, B)
                model.active_critic_hidden_states = _cat_layer(next_c_real, next_c_mirror)
            else:
                rnn_out_c, _ = model.critic_rnn(x_in_c.unsqueeze(0), None)
                critic_latent_real = rnn_out_c[0][:B]

            value = model.critic_mlp(critic_latent_real)

        model.train(was_training)

        # =====================================================================
        # 6. 保存 Transition 数据
        # =====================================================================
        self.transition.actions = model.distribution.sample().detach()
        self.transition.values = value.detach()
        self.transition.actions_log_prob = model.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = model.action_mean.detach()
        self.transition.action_sigma = model.action_std.detach()
        self.transition.observations = obs

        return self.transition.actions
# ==============================================================================
# 自定义算法：带对称性增强的蒸馏 (Sequence-Level Augmentation 双趟遍历)
# (完全保留原始逻辑与硬编码)
# ==============================================================================

class SymmetricMoEDistillation(Distillation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.device
        self.action_swap_idx = torch.tensor([3,4,5, 0,1,2, 9,10,11, 6,7,8, 13,12, 15,14], dtype=torch.long, device=device)
        self.action_neg_mask = torch.tensor([-1,1,1, -1,1,1, -1,1,1, -1,1,1, 1, 1, 1, 1], dtype=torch.float32, device=device)

    def _mirror_obs(self, obs):
        obs_mirrored = obs.clone()
        if "policy" in obs_mirrored.keys():
            p = obs_mirrored["policy"].clone()
            if p.shape[-1] > 57:
                # 60-dim layout: [base_lin_vel(3), base_ang_vel(3), gravity(3), vel_cmd(3), joints...]
                p[..., 1] *= -1.0  # base_lin_vel: y
                p[..., 3] *= -1.0  # base_ang_vel: roll (x)
                p[..., 5] *= -1.0  # base_ang_vel: yaw (z)
                p[..., 7] *= -1.0  # projected_gravity: y
                p[..., 10] *= -1.0 # velocity_commands: vy
                p[..., 11] *= -1.0 # velocity_commands: wz
                for offset in [12, 28, 44]:
                    p[..., offset:offset+16] = p[..., offset:offset+16][..., self.action_swap_idx] * self.action_neg_mask
            else:
                # 57-dim layout: [base_ang_vel(3), gravity(3), vel_cmd(3), joints...]
                p[..., 0] *= -1.0  # base_ang_vel: roll (x)
                p[..., 2] *= -1.0  # base_ang_vel: yaw (z)
                p[..., 4] *= -1.0  # projected_gravity: y
                p[..., 7] *= -1.0  # velocity_commands: vy
                p[..., 8] *= -1.0  # velocity_commands: wz
                for offset in [9, 25, 41]:
                    p[..., offset:offset+16] = p[..., offset:offset+16][..., self.action_swap_idx] * self.action_neg_mask
            obs_mirrored["policy"] = p

        for group_name in ["blind_student_policy", "student_policy"]:
            if group_name in obs_mirrored.keys():
                b = obs_mirrored[group_name].clone()
                b[..., 0] *= -1.0  # base_ang_vel: roll (x)
                b[..., 2] *= -1.0  # base_ang_vel: yaw (z)
                b[..., 4] *= -1.0  # projected_gravity: y
                b[..., 7] *= -1.0  # velocity_commands: vy
                b[..., 8] *= -1.0  # velocity_commands: wz
                for offset in [9, 25, 41]:
                    b[..., offset:offset+16] = b[..., offset:offset+16][..., self.action_swap_idx] * self.action_neg_mask
                obs_mirrored[group_name] = b

        if "estimator" in obs_mirrored.keys():
            e = obs_mirrored["estimator"].clone()
            obs_neg_mask = torch.ones(57, dtype=torch.float32, device=self.device)
            obs_neg_mask[0], obs_neg_mask[2], obs_neg_mask[4], obs_neg_mask[7], obs_neg_mask[8] = -1.0, -1.0, -1.0, -1.0, -1.0
            obs_swap_idx = torch.arange(57, dtype=torch.long, device=self.device)
            for offset in [9, 25, 41]:
                obs_swap_idx[offset:offset+16] = self.action_swap_idx + offset
                obs_neg_mask[offset:offset+16] = self.action_neg_mask
                
            history_len = e.shape[-1] // 57
            est_neg_mask = obs_neg_mask.repeat(history_len)
            est_swap_idx = torch.zeros(e.shape[-1], dtype=torch.long, device=self.device)
            for h_i in range(history_len):
                est_swap_idx[h_i*57 : (h_i+1)*57] = obs_swap_idx + h_i*57
            obs_mirrored["estimator"] = e[..., est_swap_idx] * est_neg_mask

        if "noisy_elevation" in obs_mirrored.keys():
            env_raw = obs_mirrored["noisy_elevation"].clone()
            if self.policy.use_elevation_ae:
                elev = env_raw[..., :self.policy.elevation_dim]
                elev_mirrored = elev.view(*elev.shape[:-1], 11, 17).flip(dims=[-2]).view(*elev.shape[:-1], self.policy.elevation_dim)
                env_raw[..., :self.policy.elevation_dim] = elev_mirrored
            if getattr(self.policy, "use_multilayer_scan", False):
                scan_start = self.policy.elevation_dim if self.policy.use_elevation_ae else 0
                scan = env_raw[..., scan_start : scan_start + self.policy.scan_dim]
                scan_mirrored = scan.view(*scan.shape[:-1], self.policy.num_scan_channels, self.policy.num_scan_rays).flip(dims=[-1]).view(*scan.shape[:-1], self.policy.scan_dim)
                env_raw[..., scan_start : scan_start + self.policy.scan_dim] = scan_mirrored
            obs_mirrored["noisy_elevation"] = env_raw
        return obs_mirrored

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0
        force_reset_dones = torch.ones(self.storage.num_envs, dtype=torch.bool, device=self.device)

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(dones=force_reset_dones)
            self.policy.detach_hidden_states()
            
            for obs, _, privileged_actions, dones in self.storage.generator():
                obs_mirrored = self._mirror_obs(obs)
                target_actions_mirrored = privileged_actions[..., self.action_swap_idx] * self.action_neg_mask
                
                actions = self.policy.act_inference(obs_mirrored)
                behavior_loss = self.loss_fn(actions, target_actions_mirrored)
                
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            
            for obs, _, privileged_actions, dones in self.storage.generator():
                actions = self.policy.act_inference(obs)
                behavior_loss = self.loss_fn(actions, privileged_actions)
                
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {"Loss/Distillation_Symmetric": mean_behavior_loss}

# ==============================================================================
# 6. Configurations
# ==============================================================================

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
    sym_loss_coef: float = 0.0

    blind_vision: bool = False       
    use_elevation_ae: bool = True   
    elevation_dim: int = 187      
    use_multilayer_scan: bool = False
    num_scan_channels: int = 12 
    num_scan_rays: int = 21   
    use_cnn: bool = False           
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

    feed_estimator_to_policy: bool = True 
    feed_ae_to_policy: bool = True
    teacher_is_mlp: bool = False

    def get_std(self):
        std_np = self.std.detach().cpu().numpy()
        leg_std = std_np[:self.num_leg_actions].mean() if self.num_leg_actions > 0 else 0.0
        wheel_std = std_np[self.num_leg_actions:].mean() if self.num_leg_actions < len(std_np) else 0.0
        return leg_std, wheel_std

@configclass
class SplitMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO Configuration for training the Teacher."""
    num_steps_per_env = 36
    max_iterations = 6000
    save_interval = 200
    experiment_name = "split_moe_teacher_parallel" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}
    
    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0,
        init_noise_legs=0.2,
        init_noise_wheels=1.5,
        actor_hidden_dims=[256, 128, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=3,
        num_leg_experts=6,
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,
        
        blind_vision=False, # 盲视平地训练
        use_elevation_ae=True, 
        elevation_dim=187,
        use_cnn=False, 
        
        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2], 
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        use_multilayer_scan=True,
        num_scan_channels=12,  # 6前 + 6后
        num_scan_rays=21,     # 每个通道的射线数

        actor_obs_normalization=True, 
        critic_obs_normalization=True,

        # 接收 AE/VAE
        feed_estimator_to_policy=True, 
        feed_ae_to_policy=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
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

@configclass
class SplitCMPMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO Configuration for training the Teacher."""
    num_steps_per_env = 36
    max_iterations = 25000
    save_interval = 200
    experiment_name = "split_moe_teacher_parallel_cmp" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}
    
    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0, 
        init_noise_legs=1.0,
        init_noise_wheels=0.8, 
        actor_hidden_dims=[256, 128, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=3,
        num_leg_experts=6,
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,
        
        blind_vision=True, 
        use_elevation_ae=True,
        elevation_dim=187,
        use_cnn=False, 
        
        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,
        
        actor_obs_normalization=True, 
        critic_obs_normalization=True,

        feed_estimator_to_policy=False, 
        feed_ae_to_policy=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# ------------------------------------------------------------------------------
# Distillation / Behavior Cloning Configurations (Student)
# ------------------------------------------------------------------------------

@configclass
class SplitMoEDistillationCfg(RslRlDistillationRunnerCfg):
    """
    Configuration for Student-Teacher Distillation (Behavior Cloning).
    """
    num_steps_per_env = 36
    max_iterations = 10000
    save_interval = 200
    experiment_name = "split_moe_distill"
    empirical_normalization = False

    obs_groups = {"policy": ["blind_student_policy"], "teacher": ["policy"], "noisy_elevation": ["noisy_elevation"]} 

    policy = SplitMoEActorCriticCfg(
        class_name="SplitMoEStudentTeacher", 
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
        actor_obs_normalization=True, 
        critic_obs_normalization=True,
        
        blind_vision=False,
        
        use_elevation_ae=True, 
        elevation_dim=187,
        use_cnn=False,
        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        # Student 需要开启这两个开关，以接收 Teacher 预训练好的 VAE/AE 特征
        feed_estimator_to_policy=True, 
        feed_ae_to_policy=True,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1.0e-4,
    )

@configclass
class SplitMoESenseDistillationCfg(RslRlDistillationRunnerCfg):
    """
    Configuration for Student-Teacher Distillation using 'student_policy' group.
    """
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "split_moe_distill_sense"
    empirical_normalization = False

    obs_groups = {"policy": ["student_policy"], "teacher": ["policy"], "noisy_elevation": ["noisy_elevation"]} 

    policy = SplitMoEActorCriticCfg(
        class_name="SplitMoEStudentTeacher", 
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
        actor_obs_normalization=True, 
        critic_obs_normalization=True,
        
        blind_vision=False,
        use_elevation_ae=True, 
        elevation_dim=187,
        use_cnn=False,
        
        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        # Student 接入已训练特征
        feed_estimator_to_policy=True, 
        feed_ae_to_policy=True,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1.0e-4,
    )

@configclass
class MlpToMoeDistillationCfg(RslRlDistillationRunnerCfg):
    """用于第一阶段：从平地 MLP 蒸馏到 MoE Teacher (带 Estimator)"""
    num_steps_per_env = 24
    max_iterations = 5000  # BC 蒸馏不需要太多 epoch
    save_interval = 200
    experiment_name = "mlp_to_moe_distill"
    empirical_normalization = False

    # 【修改点 1】: 将 estimator 观测组加入字典，以便传入网络
    obs_groups = {
        # "policy": ["policy"], 
        "policy": ["blind_student_policy"], 
        "teacher": ["pretraincfg"],
        # "noisy_elevation": ["noisy_elevation"]
        # "estimator": ["estimator"]
    }

    policy = SplitMoEActorCriticCfg(
        class_name="SplitMoEStudentTeacher", 
        teacher_is_mlp=True, # 开启 MLP Teacher
        
        init_noise_std=1.0, init_noise_legs=0.8, init_noise_wheels=0.5, 
        actor_hidden_dims=[256, 128, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=3, num_leg_experts=6, num_leg_actions=12,
        latent_dim=256, rnn_type="gru", aux_loss_coef=0.01,
        
        # 视觉相关关闭（保持盲视）
        blind_vision=True, 
        use_elevation_ae=False, 
        feed_ae_to_policy=False, 
        elevation_dim=187,
        
        # === Estimator 配置 ===
        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,
        feed_estimator_to_policy=False,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="SymmetricMoEDistillation", 
        num_learning_epochs=5, 
        learning_rate=1.0e-4
    )

@configclass
class BlindMoECfg(RslRlOnPolicyRunnerCfg):
    """全盲 MoE 训练配置，仅包含 57 维本体感受信息。"""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "blind_moe_training"
    empirical_normalization = False

    # 仅包含本体信息输入，对应 57 维观测
    obs_groups = {
        "policy": ["blind_student_policy"], 
    }

    policy = SplitMoEActorCriticCfg(
        class_name="SplitMoEActorCritic", 
        teacher_is_mlp=True,
        
        # 网络维度配置
        init_noise_std=1.0, 
        init_noise_legs=0.8, 
        init_noise_wheels=0.5, 
        actor_hidden_dims=[256, 128, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        
        # MoE 专家设置
        num_wheel_experts=3, 
        num_leg_experts=6, 
        num_leg_actions=12,
        latent_dim=256, 
        rnn_type="gru", 
        aux_loss_coef=0.01,
        
        # 全盲核心配置
        blind_vision=True,           # 显式开启全盲模式
        use_elevation_ae=False,      # 关闭海拔自编码器
        feed_ae_to_policy=False,     # 不将 AE 特征传入 Policy
        elevation_dim=187,           # 默认维度
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
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


@configclass
class EleMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO Configuration for training the Teacher."""
    num_steps_per_env = 36
    max_iterations = 6000
    save_interval = 100
    experiment_name = "ele_moe_teacher_parallel" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}
    
    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0, 
        init_noise_legs=0.6,
        init_noise_wheels=1.5, 
        actor_hidden_dims=[256, 128, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=3,
        num_leg_experts=4,
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,
        
        blind_vision=False, # 盲视平地训练
        use_elevation_ae=True, 
        elevation_dim=187,
        use_cnn=False, 
        
        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2], 
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        use_multilayer_scan=True,
        num_scan_channels=12,  # 6前 + 6后
        num_scan_rays=21,     # 每个通道的射线数

        actor_obs_normalization=True, 
        critic_obs_normalization=True,

        # 接收 AE/VAE
        feed_estimator_to_policy=True, 
        feed_ae_to_policy=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
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

@configclass
class ScanMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO Configuration for training the Teacher."""
    num_steps_per_env = 36
    max_iterations = 6000
    save_interval = 100
    experiment_name = "scan_moe_teacher_parallel" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}
    
    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0,
        init_noise_legs=0.6,
        init_noise_wheels=1.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=3,
        num_leg_experts=4,
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,
        # 启用对称性增强 (LR + FB sym loss). 0.5 是 sweep 测试得到的最佳折中:
        # sym 残差 ~0.005-0.007, task tracking 几乎不受影响, 梯度稳定 < clip.
        # 0.0 → sym MSE 反而上升 (训练让 policy 越学越不对称, 必须有约束).
        sym_loss_coef=0.5,

        blind_vision=False, # 盲视平地训练
        use_elevation_ae=True,
        elevation_dim=187,
        use_cnn=False,

        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        use_multilayer_scan=True,
        num_scan_channels=12,  # 6前 + 6后
        num_scan_rays=21,     # 每个通道的射线数

        actor_obs_normalization=True,
        critic_obs_normalization=True,

        # 接收 AE/VAE
        feed_estimator_to_policy=True,
        feed_ae_to_policy=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
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

# ==============================================================================
# Teacher-Specific Configurations for M20 (T1-T5)
# ==============================================================================

@configclass
class BaseMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """T1 盲视基础专家配置 - 最简单的多专家结构"""
    num_steps_per_env = 36
    max_iterations = 5000
    save_interval = 100
    experiment_name = "base_moe_teacher_parallel" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}
    
    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0, 
        init_noise_legs=0.5,
        init_noise_wheels=1.5, 
        actor_hidden_dims=[256, 128, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=2,  # 轮专家数：2（最低难度）
        num_leg_experts=3,    # 腿专家数：3（最低难度）
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,
        
        blind_vision=True,  # 盲视模式
        use_elevation_ae=False,  # 不用高程
        elevation_dim=187,
        use_cnn=False, 
        
        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2], 
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        use_multilayer_scan=False,

        actor_obs_normalization=True, 
        critic_obs_normalization=True,

        feed_estimator_to_policy=True, 
        feed_ae_to_policy=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
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

@configclass
class PlacementMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """T4 精准落足专家配置 - 中等难度的多专家结构"""
    num_steps_per_env = 36
    max_iterations = 6000
    save_interval = 100
    experiment_name = "placement_moe_teacher_parallel"
    empirical_normalization = False

    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}

    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0,
        init_noise_legs=0.6,
        init_noise_wheels=1.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=2,  # 轮专家数：2（中等难度）
        num_leg_experts=4,    # 腿专家数：4（中等难度）
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,

        blind_vision=False,
        use_elevation_ae=True,  # 使用高程估计器
        elevation_dim=187,
        use_cnn=False,

        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        use_multilayer_scan=False,

        actor_obs_normalization=True,
        critic_obs_normalization=True,

        feed_estimator_to_policy=True,
        feed_ae_to_policy=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
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

@configclass
class PlatformMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """T6 高台攀爬专家配置 - Scan AE + Elevation AE + Estimator 全开"""
    num_steps_per_env = 36
    max_iterations = 6000
    save_interval = 100
    experiment_name = "platform_moe_teacher_parallel"
    empirical_normalization = False

    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}

    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0,
        init_noise_legs=0.6,
        init_noise_wheels=1.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=3,
        num_leg_experts=4,
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,
        # 启用对称性增强 (LR + FB sym loss). 0.5 是 sweep 测试得到的最佳折中:
        # sym 残差 ~0.005-0.007, task tracking 几乎不受影响, 梯度稳定 < clip.
        # 0.0 → sym MSE 反而上升 (训练让 policy 越学越不对称, 必须有约束).
        sym_loss_coef=0.5,

        blind_vision=False,
        use_elevation_ae=True,
        elevation_dim=187,
        use_cnn=False,

        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        use_multilayer_scan=True,
        num_scan_channels=12,
        num_scan_rays=21,

        actor_obs_normalization=True,
        critic_obs_normalization=True,

        feed_estimator_to_policy=True,
        feed_ae_to_policy=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
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

@configclass
class GapMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """T7 跨越沟壑专家配置 - Scan AE + Elevation AE + Estimator 全开"""
    num_steps_per_env = 36
    max_iterations = 6000
    save_interval = 100
    experiment_name = "gap_moe_teacher_parallel"
    empirical_normalization = False

    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}

    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0,
        init_noise_legs=0.6,
        init_noise_wheels=1.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=3,
        num_leg_experts=4,
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,

        blind_vision=False,
        use_elevation_ae=True,
        elevation_dim=187,
        use_cnn=False,

        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        use_multilayer_scan=True,
        num_scan_channels=12,
        num_scan_rays=21,

        actor_obs_normalization=True,
        critic_obs_normalization=True,

        feed_estimator_to_policy=True,
        feed_ae_to_policy=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
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

@configclass
class RailMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    """T8 跨栏跳跃专家配置 - Scan AE + Elevation AE + Estimator 全开"""
    num_steps_per_env = 36
    max_iterations = 6000
    save_interval = 100
    experiment_name = "rail_moe_teacher_parallel"
    empirical_normalization = False

    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}

    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0,
        init_noise_legs=0.6,
        init_noise_wheels=1.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=3,
        num_leg_experts=4,
        num_leg_actions=12,
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,

        blind_vision=False,
        use_elevation_ae=True,
        elevation_dim=187,
        use_cnn=False,

        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2],
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        use_multilayer_scan=True,
        num_scan_channels=12,
        num_scan_rays=21,

        actor_obs_normalization=True,
        critic_obs_normalization=True,

        feed_estimator_to_policy=True,
        feed_ae_to_policy=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
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