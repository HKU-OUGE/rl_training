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
# 2. CMoE: ElevationAE, DepthAE & ProprioVAE 
# ==============================================================================

class ElevationAE(nn.Module):
    """用于 1D 高程图/地形扫描的自编码器 (对应论文 Elevation Map AE)"""
    def __init__(self, input_dim=187, output_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        # Encoder: 提取地形隐变量 z_t^E
        self.encoder = MLP(input_dim, output_dim, hidden_dims, activation="elu")
        # Decoder: 用于重建高程图
        self.decoder = MLP(output_dim, input_dim, hidden_dims[::-1], activation="elu")

    def forward(self, x):
        latent = self.encoder(x) # z_t^E
        recon = self.decoder(latent) if self.training else None
        return latent, recon
class MultiLayerScanAE(nn.Module):
    """用于处理前后多层 2D 伪激光扫描的 1D-CNN"""
    def __init__(self, num_channels=6, num_rays=61, output_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        self.num_channels = num_channels
        self.num_rays = num_rays
        self.flat_dim = num_channels * num_rays
        
        # Encoder (1D-CNN)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
        )
        
        conv_out_len = (num_rays + 1) // 2
        conv_out_len = (conv_out_len + 1) // 2
        linear_in_dim = 32 * conv_out_len # 对于 61 来说，长度最后是 16，所以是 32*16=512
        
        self.encoder_linear = nn.Sequential(
            nn.Linear(linear_in_dim, hidden_dims[0]), nn.ELU(),
            nn.Linear(hidden_dims[0], output_dim)
        )
        
        # Decoder (MLP)
        self.decoder = MLP(output_dim, self.flat_dim, hidden_dims[::-1], activation="elu")

    def forward(self, x):
        # 记录原始的前置维度 (例如: [Seq_len, Batch] 或仅 [Batch])
        original_shape = x.shape[:-1] 
        
        # [修改点 1]: 将 view 换成 reshape
        x_reshaped = x.reshape(-1, self.num_channels, self.num_rays)
        
        # 1D-CNN 提取特征
        conv_features = self.encoder_conv(x_reshaped).reshape(x_reshaped.shape[0], -1)
        latent_flat = self.encoder_linear(conv_features)
        
        # [修改点 2]: 将 view 换成 reshape
        latent = latent_flat.reshape(*original_shape, -1)
        
        recon = self.decoder(latent) if self.training else None
        return latent, recon
class DepthAE(nn.Module):
    """用于 2D 高程图/深度图的自编码器 (如果使用相机 CNN)"""
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
    """用于历史本体感受状态蒸馏的变分自编码器 (beta-VAE)"""
    def __init__(self, input_dim, vel_dim=3, latent_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        # Encoder: 输入历史观测 o_t^H
        self.encoder_mlp = MLP(input_dim, hidden_dims[-1], hidden_dims[:-1])
        
        # VAE 的均值和方差
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # 速度估计器 (输出 v_tilde_t)
        self.vel_estimator = nn.Linear(hidden_dims[-1], vel_dim)

        # Decoder: 输入隐变量 z_t^H，重建观测
        self.decoder_mlp = MLP(latent_dim, input_dim, hidden_dims[::-1])

    def encode(self, x):
        h = self.encoder_mlp(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        vel_pred = self.vel_estimator(h) # 速度预测
        return mu, logvar, vel_pred

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_mlp(z)

    def forward(self, x):
        mu, logvar, vel_pred = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu # z_t^H
        recon_x = self.decode(z) if self.training else None
        return vel_pred, recon_x, mu, logvar, z

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
                 # === AE Params ===
                 blind_vision=True,     
                 use_elevation_ae=True, 
                 elevation_dim=187,     
                 use_cnn=False,
                 num_cameras=2,       
                 camera_height=58,    
                 camera_width=87,
                 # === Distillation / Overrides ===
                 forced_input_key=None,
                 is_student_mode=False, 
                 feed_estimator_to_policy=False, # 新增：Teacher设为False防止策略崩塌
                 feed_ae_to_policy=False,        # 新增：Teacher使用Raw数据，不依赖AE
                 **kwargs):
        
        base_kwargs = {k: v for k, v in kwargs.items() if k not in [
            "estimator_output_dim", "estimator_hidden_dims", 
            "estimator_input_indices", "estimator_target_indices", 
            "estimator_obs_normalization", "init_noise_legs", "init_noise_wheels",
            "actor_obs_normalization", "critic_obs_normalization",
            "use_elevation_ae", "elevation_dim", "blind_vision", "is_student_mode",
            "feed_estimator_to_policy", "feed_ae_to_policy",
            "use_multilayer_scan", "num_scan_channels", "num_scan_rays", "teacher_is_mlp",
            "use_cnn", "num_cameras", "camera_height", "camera_width"
        ]}

        super().__init__(obs, obs_groups, num_actions, 
                         actor_hidden_dims=actor_hidden_dims, 
                         critic_hidden_dims=critic_hidden_dims, 
                         activation=activation, 
                         init_noise_std=init_noise_std, 
                         **base_kwargs)

        self.is_student_mode = is_student_mode
        self.feed_estimator = feed_estimator_to_policy
        self.feed_ae = feed_ae_to_policy

        # -------------------------------------------------------------
        # 1. 解析基础输入维度
        # -------------------------------------------------------------
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
        
        # 核心：计算真正的本体感觉维度 (增强了鲁棒性判断)
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

        # Critic 维度推断
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

        # -------------------------------------------------------------
        # 2. 初始化 Estimator (ProprioVAE) 并获取其额外输出维度
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 3. 初始化地形感知 AE 并动态计算 RNN input_size
        # -------------------------------------------------------------
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

        # =============================================================
        # 核心修改：根据 Router 开关解耦输入维度
        # =============================================================
        rnn_input_dim = self.proprio_dim
        critic_rnn_input_dim = self.critic_proprio_dim

        if getattr(self, 'has_elevation_input', True):
            rnn_input_dim += self.ae_output_dim if self.feed_ae else (num_obs - self.proprio_dim)
            critic_rnn_input_dim += self.ae_output_dim if self.feed_ae else (critic_num_obs - self.critic_proprio_dim)
        else:
            if self.feed_ae:
                # 盲视学生若开启了 feed_ae，会拼接 0 向量，所以需要加上该维度
                rnn_input_dim += self.ae_output_dim
                critic_rnn_input_dim += self.ae_output_dim

        if self.feed_estimator:
            rnn_input_dim += self.vae_feature_dim
            critic_rnn_input_dim += self.vae_feature_dim

        print(f"[SplitMoE] RNN Actor Input: {rnn_input_dim} (Proprio: {self.proprio_dim}, "
              f"VAE appended: {self.vae_feature_dim if self.feed_estimator else 0}, "
              f"Env appended: {self.ae_output_dim if self.feed_ae else (num_obs - self.proprio_dim)}, "
              f"Blind: {self.blind_vision}, Student: {self.is_student_mode})")

        # Normalization Setup
        self.actor_obs_normalization = kwargs.get("actor_obs_normalization", True)
        self.critic_obs_normalization = kwargs.get("critic_obs_normalization", True)

        if self.actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(shape=[self.proprio_dim], until_step=1.0e9)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[self.critic_proprio_dim], until_step=1.0e9)

        self.latent_dim = latent_dim
        self.rnn_type = rnn_type.lower()
        self.aux_loss_coef = aux_loss_coef
        self.num_leg_actions = num_leg_actions
        self.num_wheel_actions = num_actions - num_leg_actions

        # -------------------------------------------------------------
        # 4. Initialize Actor & Critic RNNs
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 5. MoE Gates and Experts
        # -------------------------------------------------------------
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

        # RNN State Init
        if isinstance(obs, dict): ref_tensor = obs[list(obs.keys())[0]]
        else: ref_tensor = obs
        batch_size = ref_tensor.shape[0]
        device = ref_tensor.device
        
        self.active_hidden_states = self._init_rnn_state(batch_size, device)
        self.active_critic_hidden_states = self._init_rnn_state(batch_size, device)
        
        self.latest_weights = {}
        self.active_aux_loss = 0.0
        
        # Noise Init
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
            tensors = [obs[k] for k in key_list if k in obs]
            if not tensors: return list(obs.values())[0] # Fallback
            return torch.cat(tensors, dim=-1)
        return obs

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            x_raw_actor = self._extract_raw_obs(obs, self.input_keys)
            proprio_actor = x_raw_actor[..., :self.proprio_dim]
            self.actor_obs_normalizer.update(proprio_actor)
            
        if self.critic_obs_normalization:
            x_raw_critic = self._extract_raw_obs(obs, getattr(self, "critic_keys", self.input_keys))
            proprio_critic = x_raw_critic[..., :self.critic_proprio_dim] 
            self.critic_obs_normalizer.update(proprio_critic)

    def _process_obs(self, x, obs_dict=None, normalizer=None, proprio_dim=None):
        if proprio_dim is None: 
            proprio_dim = self.proprio_dim
            
        proprio = x[..., :proprio_dim]
        
        if normalizer is not None:
            proprio = normalizer(proprio)
        
        # -------------------------------------------------------------
        # 1. 外部地形感知 AE 特征提取 (根据解耦开关判断是否喂给 RNN)
        # -------------------------------------------------------------
        env_feat_for_rnn = None
        if (self.use_elevation_ae or self.use_multilayer_scan) and self.feed_ae:
            # 【修复点】：强制安全地从环境观测字典中提取 noisy_elevation
            if obs_dict is not None and (isinstance(obs_dict, dict) or hasattr(obs_dict, "keys")):
                if "noisy_elevation" in obs_dict:
                    env_raw_full = obs_dict["noisy_elevation"]
                elif "policy" in obs_dict:
                     # 极端回退：如果没配置 noisy_elevation，试图从 policy 截取（假设全部拼在一起）
                     env_raw_full = obs_dict["policy"][..., proprio_dim:]
                else:
                    raise KeyError("Obs dictionary does not contain 'noisy_elevation'!")
            else:
                # 如果传入的不是字典而是纯 Tensor，才使用切片
                env_raw_full = x[..., proprio_dim:]
            
            # 校验一下我们拿到的数据够不够长
            expected_min_dim = 0
            if self.use_elevation_ae: expected_min_dim += self.elevation_dim
            if self.use_multilayer_scan: expected_min_dim += self.scan_dim
            
            if env_raw_full.shape[-1] < expected_min_dim:
                 raise RuntimeError(f"Extracted environment feature dimension ({env_raw_full.shape[-1]}) "
                                    f"is smaller than required ({expected_min_dim}). Check your ObsGroup config!")
            feats = []
            current_idx = 0
            
            # 流 1: 高程图 (看脚下坑洼)
            if self.use_elevation_ae:
                env_raw_elev = env_raw_full[..., current_idx : current_idx + self.elevation_dim]
                latent_elev, _ = self.elevation_encoder(env_raw_elev)
                feats.append(latent_elev)
                current_idx += self.elevation_dim
                
            # 流 2: 多层扫描 (看前后悬垂物)
            if self.use_multilayer_scan:
                env_raw_scan = env_raw_full[..., current_idx : current_idx + self.scan_dim]
                latent_scan, _ = self.scan_encoder(env_raw_scan)
                feats.append(latent_scan)
                
            env_feat_for_rnn = torch.cat(feats, dim=-1)
            
            if getattr(self, "blind_vision", False):
                env_feat_for_rnn = torch.zeros_like(env_feat_for_rnn)
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
                    env_feat_ae, env_recon = self.depth_encoder(depth_image)
                    env_feat_ae = env_feat_ae.view(B, T, -1)
                else:
                    env_feat_ae, env_recon = self.depth_encoder(depth_image)
                
                env_feat_for_rnn = env_feat_ae if self.feed_ae else env_raw
                if self.blind_vision:
                    env_feat_for_rnn = torch.zeros_like(env_feat_for_rnn)
        else:
            if x.shape[-1] > proprio_dim:
                env_feat_for_rnn = x[..., proprio_dim:]
                if self.blind_vision:
                    env_feat_for_rnn = torch.zeros_like(env_feat_for_rnn)

        # -------------------------------------------------------------
        # 2. 历史上下文 VAE 特征提取 (根据解耦开关判断是否喂给 RNN)
        # -------------------------------------------------------------
        vae_latent_for_rnn = None
        vel_pred_for_rnn = None
        if self.estimator is not None and obs_dict is not None:
            # 【新增判断】仅在训练时计算 Loss，或策略确实需要 VAE 特征时，才进行前向传播
            if self.training or self.feed_estimator:
                est_input = self._get_estimator_input(obs_dict)
                if self.estimator_obs_normalization and self.estimator_obs_normalizer is not None:
                    est_input = self.estimator_obs_normalizer(est_input)
                
                vel_pred, recon_x, mu, logvar, z = self.estimator(est_input)
                
                if self.training:
                    self.active_vae_recon = recon_x
                    self.active_vae_mu = mu
                    self.active_vae_logvar = logvar
                    self.active_vae_vel_pred = vel_pred
                    self.active_vae_input = est_input 
                
                if self.feed_estimator:
                    vae_latent_for_rnn = z
                    vel_pred_for_rnn = vel_pred
                
        elif self.estimator is not None and obs_dict is None:
            if self.feed_estimator:
                orig_shape = proprio.shape[:-1]
                vel_pred_for_rnn = torch.zeros((*orig_shape, self.estimator_output_dim), device=proprio.device)
                vae_latent_for_rnn = torch.zeros((*orig_shape, 64), device=proprio.device)

        # -------------------------------------------------------------
        # 3. 拼接最终给到 RNN 的观测向量
        # -------------------------------------------------------------
        components = [proprio]
        if self.feed_estimator and self.estimator is not None:
            if vel_pred_for_rnn is not None:
                components.append(vel_pred_for_rnn)
                components.append(vae_latent_for_rnn)
            
        if env_feat_for_rnn is not None:
            components.append(env_feat_for_rnn)

        x_out = torch.cat(components, dim=-1)
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
        if self.has_estimator_group and (isinstance(obs_dict, dict) or hasattr(obs_dict, "keys")):
            if "estimator" in obs_dict:
                return obs_dict["estimator"]
        
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
            # PPO update calls act() passing batch, meaning we can cache aux loss natively
            self.active_aux_loss = self._calculate_load_balancing_loss(w_leg, w_wheel) * self.aux_loss_coef
        
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
        x_in = self._process_obs(x_raw, obs_dict=obs, normalizer=normalizer)
        
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
        x_in = self._process_obs(x_raw, obs_dict=obs, normalizer=normalizer)

        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_states is None: self.active_hidden_states = next_state
        return self._compute_actor_output(latent)

    def evaluate(self, obs, masks=None, hidden_state=None):
        x_raw = self._extract_raw_obs(obs, getattr(self, "critic_keys", self.input_keys))
        normalizer = self.critic_obs_normalizer if self.critic_obs_normalization else None
        x_in = self._process_obs(x_raw, obs_dict=obs, normalizer=normalizer, proprio_dim=self.critic_proprio_dim)

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
        if save_dist: self.distribution = torch.distributions.Normal(action_mean, self.std)
        return action_mean, self.std, next_state

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
    """
    Adapter class to make SplitMoEActorCritic compatible with rsl_rl.runners.DistillationRunner.
    """
    is_recurrent = True
    loaded_teacher = False

    def __init__(self, obs, obs_groups, num_actions, activation="elu", **kwargs):
        super().__init__()
        self.obs_groups = obs_groups
        
        if isinstance(activation, dict):
            activation = activation.get("value", "elu") if "value" in activation else "elu"
        # 提取动态开关 (默认 False)
        teacher_is_mlp = kwargs.pop("teacher_is_mlp", False) 
        # =========================================================
        # 初始化 Student
        # =========================================================
        print("\n[Distillation] Initializing Student Policy...")
        student_kwargs = kwargs.copy()
        
        student_obs_groups = {"policy": obs_groups["policy"]}
        student_obs_groups["critic"] = obs_groups["policy"]
        student_input_key = obs_groups["policy"][0]

        self.student = SplitMoEActorCritic(
            obs, 
            student_obs_groups, 
            num_actions, 
            forced_input_key=student_input_key, 
            activation=activation,
            is_student_mode=True, 
            **student_kwargs
        )

        # =========================================================
        # 2. 动态初始化 Teacher
        # =========================================================
        teacher_source = obs_groups["teacher"]
        
        if teacher_is_mlp:
            print("\n[Distillation] Initializing Teacher Policy (Standard MLP)...")
            
            # 为 Teacher 构造满足 ActorCritic 接口要求的 obs_groups 字典
            # 因为是平地策略，actor 和 critic 都使用相同的 teacher_source (即 policy 组)
            teacher_obs_groups = {
                "policy": teacher_source,
                "critic": teacher_source
            }

            # 初始化标准 MLP，完全遵循 RSL-RL 最新 API 规范，维度对齐 agent.yaml
            self.teacher = ActorCritic(
                obs=obs,
                obs_groups=teacher_obs_groups,
                num_actions=num_actions,
                actor_hidden_dims=[512, 256, 128], 
                critic_hidden_dims=[512, 256, 128],
                activation=activation, 
                init_noise_std=1.0,
            )
        else:
            print("\n[Distillation] Initializing Teacher Policy (SplitMoE)...")
            teacher_kwargs = kwargs.copy()
            teacher_kwargs["feed_estimator_to_policy"] = False
            teacher_kwargs["feed_ae_to_policy"] = False
            teacher_obs_groups = {"policy": teacher_source, "critic": teacher_source}
            teacher_input_key = teacher_source[0]
            
            self.teacher = SplitMoEActorCritic(
                obs, teacher_obs_groups, num_actions, 
                forced_input_key=teacher_input_key, activation=activation, 
                **teacher_kwargs
            )
            
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    @property
    def action_std(self):
        return self.student.std

    def reset(self, dones=None, hidden_states=None):
        self.student.reset(dones, hidden_states=hidden_states)
        self.teacher.reset(dones)

    def act(self, obs, masks=None, hidden_state=None):
        return self.student.act(obs, masks, hidden_state)

    def act_inference(self, obs):
        return self.student.act_inference(obs)

    def evaluate(self, obs):
        with torch.no_grad():
            return self.teacher.act_inference(obs)
    
    def get_hidden_states(self):
        return self.student.get_hidden_states()
    
    def detach_hidden_states(self, dones=None):
        if dones is not None and hasattr(dones, "dtype") and dones.dtype == torch.uint8:
            dones = dones.bool()

        def detach_recursive(h, mask):
            if isinstance(h, tuple): 
                return tuple(detach_recursive(x, mask) for x in h)
            
            # 断开计算图，防止梯度回传过长导致 OOM
            h = h.detach() 
            
            # 如果环境结束了 (done)，则清空隐藏状态
            if mask is not None:
                h = h.clone() # 必须 clone，否则原地修改会报错
                h[:, mask, :] = 0.0
            return h
            
        # 1. 处理 Actor 的隐藏状态
        if self.student.active_hidden_states is not None:
             self.student.active_hidden_states = detach_recursive(self.student.active_hidden_states, dones)
             
        # 2. 处理 Critic 的隐藏状态
        if hasattr(self.student, "active_critic_hidden_states") and self.student.active_critic_hidden_states is not None:
             self.student.active_critic_hidden_states = detach_recursive(self.student.active_critic_hidden_states, dones)

    def update_normalization(self, obs):
        self.student.update_normalization(obs)
    
    def load_state_dict(self, state_dict, strict=True):
        print(f"[Distillation] Loading state dict with {len(state_dict)} keys...")
        
        teacher_state_dict = {k: v for k, v in state_dict.items() if "critic" not in k}
        
        try:
            self.teacher.load_state_dict(teacher_state_dict, strict=False)
            print("[Distillation] Successfully loaded Teacher weights (critic ignored).")
            self.loaded_teacher = True
        except Exception as e:
            print(f"[Distillation] Warning: Direct load to teacher failed: {e}")

        # =========================================================
        # 【关键改动】将 Teacher 预训练好的 VAE 和 AE 权重拷贝给 Student
        # =========================================================
        if not any(k.startswith("student.") for k in state_dict.keys()):
            print("[Distillation] Initializing Student's VAE/AE from Teacher...")
            student_dict = self.student.state_dict()
            for k, v in teacher_state_dict.items():
                if k.startswith("estimator") or k.startswith("elevation_encoder") or k.startswith("depth_encoder"):
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
    def update(self):
        loss_dict = super().update()
        model = getattr(self, "actor_critic", getattr(self, "policy", None))
        
        if model is not None and self.num_learning_epochs > 0:
            obs_batch = self.storage.observations.flatten(0, 1)
            self.optimizer.zero_grad()
            total_aux_loss = 0.0
            
            # -------------------------------------------------------------
            # 1. 核心计算: ProprioVAE Loss
            # -------------------------------------------------------------
            if model.estimator is not None and not model.is_student_mode:
                est_input = model._get_estimator_input(obs_batch)
                if hasattr(obs_batch, "keys") or isinstance(obs_batch, dict):
                    try: full_obs_for_target = obs_batch["critic"]
                    except KeyError: full_obs_for_target = model._extract_raw_obs(obs_batch, getattr(model, "critic_keys", model.input_keys))
                else: 
                    full_obs_for_target = obs_batch
                
                target_state = full_obs_for_target[..., model.estimator_target_indices]
                
                if getattr(model, "estimator_obs_normalization", False) and model.estimator_obs_normalizer is not None:
                    est_input = model.estimator_obs_normalizer(est_input)
                
                vel_pred, recon_x, mu, logvar, z = model.estimator(est_input)

                vel_loss = (vel_pred - target_state).pow(2).mean()
                recon_loss = (recon_x - est_input).pow(2).mean()
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

                beta = 0.01 
                vae_loss = vel_loss + recon_loss + beta * kl_loss
                total_aux_loss += vae_loss
                
                if not hasattr(model, 'loss_dict_cache'):
                    model.loss_dict_cache = {}
                model.loss_dict_cache["Loss/VAE_Vel_MSE"] = vel_loss.item()
                model.loss_dict_cache["Loss/VAE_Recon_MSE"] = recon_loss.item()
                model.loss_dict_cache["Loss/VAE_KL"] = kl_loss.item()

            # -------------------------------------------------------------
            # 2. 核心计算: ElevationAE 重构 Loss
            # -------------------------------------------------------------
            if model.use_elevation_ae and not getattr(model, "blind_vision", False) and not model.is_student_mode:
                # ====== 优先从独立的 noisy_elevation 组获取训练数据 ======
                if hasattr(obs_batch, "keys") and "noisy_elevation" in obs_batch:
                    env_raw_full = obs_batch["noisy_elevation"]
                else:
                    x_raw = model._extract_raw_obs(obs_batch, model.input_keys)
                    env_raw_full = x_raw[..., model.proprio_dim:]
                
                # 【修改点 1】：必须对 553 维的数据进行切片，只取前 187 维的高程图！
                env_raw_elev = env_raw_full[..., :model.elevation_dim]
                
                # 传入切片后的 187 维数据
                _, env_recon = model.elevation_encoder(env_raw_elev)
                
                if env_recon is not None:
                    # 【修改点 2】：计算 Loss 时也要用切片后的目标数据 env_raw_elev
                    elev_ae_loss = (env_recon - env_raw_elev).pow(2).mean()
                    total_aux_loss += elev_ae_loss
                    if not hasattr(model, 'loss_dict_cache'):
                        model.loss_dict_cache = {}
                    model.loss_dict_cache["Loss/Elevation_AE_Recon"] = elev_ae_loss.item()
            # 3. 核心计算: MultiLayer Scan 重构 Loss
            if getattr(model, "use_multilayer_scan", False) and not getattr(model, "blind_vision", False) and not model.is_student_mode:
                if hasattr(obs_batch, "keys") and "noisy_elevation" in obs_batch:
                    env_raw_full = obs_batch["noisy_elevation"]
                else:
                    x_raw = model._extract_raw_obs(obs_batch, model.input_keys)
                    env_raw_full = x_raw[..., model.proprio_dim:]
                
                scan_start_idx = model.elevation_dim if model.use_elevation_ae else 0
                env_raw_scan = env_raw_full[..., scan_start_idx : scan_start_idx + model.scan_dim]
                
                _, scan_recon = model.scan_encoder(env_raw_scan)
                
                if scan_recon is not None:
                    scan_ae_loss = (scan_recon - env_raw_scan).pow(2).mean()
                    total_aux_loss += scan_ae_loss
                    if not hasattr(model, 'loss_dict_cache'): model.loss_dict_cache = {}
                    model.loss_dict_cache["Loss/Scan_AE_Recon"] = scan_ae_loss.item()
            # -------------------------------------------------------------
            # 3. 核心计算: DepthAE 重构 Loss
            # -------------------------------------------------------------
            if model.use_cnn and not getattr(model, "blind_vision", False) and not model.is_student_mode:
                if hasattr(obs_batch, "keys") and "noisy_elevation" in obs_batch:
                    env_raw = obs_batch["noisy_elevation"]
                else:
                    x_raw = model._extract_raw_obs(obs_batch, model.input_keys)
                    env_raw = x_raw[..., model.proprio_dim:]
                
                original_shape = env_raw.shape[:-1]
                depth_image = env_raw.view(*original_shape, model.num_cameras, model.camera_height, model.camera_width)
                if depth_image.ndim == 5:
                    B, T, C, H, W = depth_image.shape
                    depth_image = depth_image.view(B*T, C, H, W)
                    
                _, depth_recon = model.depth_encoder(depth_image)
                if depth_recon is not None:
                    depth_ae_loss = (depth_recon - depth_image).pow(2).mean() 
                    total_aux_loss += depth_ae_loss
                    if not hasattr(model, 'loss_dict_cache'):
                        model.loss_dict_cache = {}
                    model.loss_dict_cache["Loss/Depth_AE_Recon"] = depth_ae_loss.item()

            # --- 反向传播 ---
            if isinstance(total_aux_loss, torch.Tensor) and total_aux_loss.requires_grad:
                total_aux_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        # --- 日志记录 ---
        if model is not None:
            if hasattr(model, "latest_weights") and model.latest_weights:
                w = model.latest_weights
                if "leg" in w:
                    for i, val in enumerate(w["leg"]): loss_dict[f"Gate/Leg_Expert_{i}"] = val.item()
                if "wheel" in w:
                    for i, val in enumerate(w["wheel"]): loss_dict[f"Gate/Wheel_Expert_{i}"] = val.item()
            if hasattr(model, "active_aux_loss"): 
                loss_dict["Loss/Load_Balancing"] = model.active_aux_loss.item() if isinstance(model.active_aux_loss, torch.Tensor) else model.active_aux_loss
            
            if hasattr(model, "loss_dict_cache"):
                loss_dict.update(model.loss_dict_cache)

            if hasattr(model, "std"):
                std_np = model.std.detach().cpu().numpy()
                n_legs = getattr(model, "num_leg_actions", 12)
                if len(std_np) >= n_legs:
                    loss_dict["Noise/Leg_Std"] = std_np[:n_legs].mean()
                    loss_dict["Noise/Wheel_Std"] = std_np[n_legs:].mean() if len(std_np) > n_legs else 0.0
        # 👇 ========================== 新增：序列级 Actor 对称性正则化 ========================== 👇

        if model is not None and not getattr(model, "is_student_mode", False) and self.num_learning_epochs > 0:
            device = self.device
            # 1. 动作映射 (保持不变)
            action_swap_idx = torch.tensor([3,4,5, 0,1,2, 9,10,11, 6,7,8, 13,12, 15,14], dtype=torch.long, device=device)
            action_neg_mask = torch.tensor([-1,1,1, -1,1,1, -1,1,1, -1,1,1, 1, 1, 1, 1], dtype=torch.float32, device=device)
            
            obs_all = self.storage.observations
            obs_mirrored_all = obs_all.clone()
            
            #3. 仅对 policy 组进行精确镜像 【关键修改区：适配 57 维】
            # 因为去除了 base_lin_vel，整体维度前移 3 维
            obs_neg_mask = torch.ones(57, dtype=torch.float32, device=device)
            obs_neg_mask[0] = -1.0  # ang_vel roll (原为 3)
            obs_neg_mask[2] = -1.0  # ang_vel yaw (原为 5)
            obs_neg_mask[4] = -1.0  # gravity y (原为 7)
            obs_neg_mask[7] = -1.0  # cmd vy (原为 10)
            obs_neg_mask[8] = -1.0  # cmd wz (原为 11)
            
            obs_swap_idx = torch.arange(57, dtype=torch.long, device=device)
            # 对应的 joint_pos, joint_vel, actions 的起始索引也前移了 3
            for offset in [9, 25, 41]: 
                obs_swap_idx[offset:offset+16] = action_swap_idx + offset
                obs_neg_mask[offset:offset+16] = action_neg_mask
            # # 3. 仅对 policy 组进行精确镜像
            # obs_neg_mask = torch.ones(60, dtype=torch.float32, device=device)
            # obs_neg_mask[1] = -1.0  # lin_vel y
            # obs_neg_mask[3] = -1.0  # ang_vel roll
            # obs_neg_mask[5] = -1.0  # ang_vel yaw
            # obs_neg_mask[7] = -1.0  # gravity y
            # obs_neg_mask[10] = -1.0 # cmd vy
            # obs_neg_mask[11] = -1.0 # cmd wz
            
            # obs_swap_idx = torch.arange(60, dtype=torch.long, device=device)
            # for offset in [12, 28, 44]:
            #     obs_swap_idx[offset:offset+16] = action_swap_idx + offset
            #     obs_neg_mask[offset:offset+16] = action_neg_mask

            if "policy" in obs_mirrored_all.keys():
                p = obs_mirrored_all["policy"]
                obs_mirrored_all["policy"] = p[..., obs_swap_idx] * obs_neg_mask
            
            force_reset_dones = torch.ones(self.storage.num_envs, dtype=torch.bool, device=device)

            # 4. PASS 1 (无梯度): 真实观测下的动作基准
            model.reset(dones=force_reset_dones)
            target_actions = []
            with torch.no_grad():
                for t in range(self.storage.num_transitions_per_env):
                    # 传入完整的 TensorDict slice，所有的 VAE/AE 特征都能正确读取！
                    act = model.act_inference(obs_all[t])
                    target_actions.append(act)
            target_actions = torch.stack(target_actions)
            target_mirrored_actions = target_actions[..., action_swap_idx] * action_neg_mask

            # 5. PASS 2 (开启梯度): 镜像观测下，强迫输出镜像动作
            model.reset(dones=force_reset_dones)
            pred_actions = []
            for t in range(self.storage.num_transitions_per_env):
                act = model.act_inference(obs_mirrored_all[t])
                pred_actions.append(act)
            pred_actions = torch.stack(pred_actions)
            
            # 6. 计算 Loss 并反向传播
            sym_loss = torch.nn.functional.mse_loss(pred_actions, target_mirrored_actions.detach())
            sym_loss_weight = 1.0 
            
            self.optimizer.zero_grad()
            (sym_loss * sym_loss_weight).backward()
            
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            model.reset(dones=force_reset_dones)
            loss_dict["Loss/Actor_Symmetry_Reg"] = sym_loss.item()

        # 👆 ====================================================================================== 👆
        
        return loss_dict

# ==============================================================================
# 自定义算法：带对称性增强的蒸馏 (Sequence-Level Augmentation 双趟遍历)
# ==============================================================================
from rsl_rl.algorithms import Distillation

class SymmetricMoEDistillation(Distillation):
    """
    带有序列级左右对称性增强的蒸馏算法。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.device
        
        # 1. 动作维度的镜像映射 (16维)
        # 交换左右腿和左右轮
        self.action_swap_idx = torch.tensor(
            [3,4,5, 0,1,2, 9,10,11, 6,7,8, 13,12, 15,14], dtype=torch.long, device=device
        )
        # 仅侧摆关节 (hipx) 在左右镜像时符号相反
        self.action_neg_mask = torch.tensor(
            [-1,1,1, -1,1,1, -1,1,1, -1,1,1, 1, 1, 1, 1], dtype=torch.float32, device=device
        )

    def _mirror_obs(self, obs):
        """精准镜像 TensorDict 内部的各个观测组，而不破坏 Batch 维度"""
        obs_mirrored = obs.clone()
        
        # 1. 镜像 policy 组 (60维，包含真实线速度)
        if "policy" in obs_mirrored.keys():
            p = obs_mirrored["policy"].clone()
            p[..., 1] *= -1.0  # base_lin_vel: y
            p[..., 3] *= -1.0  # base_ang_vel: roll (x)
            p[..., 5] *= -1.0  # base_ang_vel: yaw (z)
            p[..., 7] *= -1.0  # projected_gravity: y
            p[..., 10] *= -1.0 # velocity_commands: vy
            p[..., 11] *= -1.0 # velocity_commands: wz
            # 处理关节位置、关节速度、历史动作 (起始索引 12, 28, 44)
            for offset in [12, 28, 44]:
                p[..., offset:offset+16] = p[..., offset:offset+16][..., self.action_swap_idx] * self.action_neg_mask
            obs_mirrored["policy"] = p

        # 2. 镜像 blind_student_policy / student_policy 组 (57维，无真实线速度)
        for group_name in ["blind_student_policy", "student_policy"]:
            if group_name in obs_mirrored.keys():
                b = obs_mirrored[group_name].clone()
                b[..., 0] *= -1.0  # base_ang_vel: roll (x)
                b[..., 2] *= -1.0  # base_ang_vel: yaw (z)
                b[..., 4] *= -1.0  # projected_gravity: y
                b[..., 7] *= -1.0  # velocity_commands: vy
                b[..., 8] *= -1.0  # velocity_commands: wz
                # 处理关节位置、关节速度、历史动作 (起始索引 9, 25, 41)
                for offset in [9, 25, 41]:
                    b[..., offset:offset+16] = b[..., offset:offset+16][..., self.action_swap_idx] * self.action_neg_mask
                obs_mirrored[group_name] = b
                
        # noisy_elevation 高程图在平地时对称，暂不反转；若未来在崎岖地形上也想镜像，需翻转 grid
        return obs_mirrored

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        # 创建一个全 True 的 done 掩码，用于强制清空 RNN 状态
        force_reset_dones = torch.ones(self.storage.num_envs, dtype=torch.bool, device=self.device)

        for epoch in range(self.num_learning_epochs):
            
            # ==============================================================
            # PASS 1: 镜像轨迹 (Sequence-Level Mirrored Pass)
            # ==============================================================
            self.policy.reset(dones=force_reset_dones)
            self.policy.detach_hidden_states()
            
            for obs, _, privileged_actions, dones in self.storage.generator():
                # 安全地获取镜像后的 TensorDict
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

            # ==============================================================
            # PASS 2: 标准轨迹 (Sequence-Level Standard Pass)
            # ==============================================================
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

    # === AE / Vision Params ===
    blind_vision: bool = False       
    use_elevation_ae: bool = True   
    elevation_dim: int = 187      
    use_multilayer_scan: bool = False
    num_scan_channels: int = 12  # 6前 + 6后
    num_scan_rays: int = 21     # 每个通道的射线数
    use_cnn: bool = False           
    num_cameras: int = 2
    camera_height: int = 58
    camera_width: int = 87

    # === Estimator Params ===
    estimator_output_dim: int = 3  
    estimator_hidden_dims: list = field(default_factory=lambda: [128, 64])
    estimator_target_indices: list = field(default_factory=lambda: [0, 1, 2])
    estimator_input_indices: list = field(default_factory=lambda: list(range(3, 9)) + list(range(12, 56)))
    estimator_obs_normalization: bool = True 
    
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])

    # === 解耦开关 ===
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
    max_iterations = 25000
    save_interval = 200
    experiment_name = "split_moe_teacher_parallel" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}
    
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
    max_iterations = 25000
    save_interval = 200
    experiment_name = "ele_moe_teacher_parallel" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}
    
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
        
        blind_vision=False, # 盲视平地训练
        use_elevation_ae=True, 
        elevation_dim=187,
        use_cnn=False, 
        
        estimator_output_dim=3,
        estimator_hidden_dims=[128, 64],
        estimator_target_indices=[0, 1, 2], 
        estimator_input_indices=list(range(3, 9)) + list(range(12, 56)),
        estimator_obs_normalization=True,

        use_multilayer_scan=False,
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
class SplitMoEServerPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO Configuration for training the Teacher."""
    num_steps_per_env = 36
    max_iterations = 25000
    save_interval = 200
    experiment_name = "split_moe_teacher_parallel" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"], "noisy_elevation": ["noisy_elevation"]}
    
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
        num_mini_batches=32,
        learning_rate=1.0e-3, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )