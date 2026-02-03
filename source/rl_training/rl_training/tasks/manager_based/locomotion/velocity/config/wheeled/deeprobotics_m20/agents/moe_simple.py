# source/robot_lab/algo/moe/policy_paper_match.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import ActorCritic
# === 关键导入：unpad_trajectories ===
from rsl_rl.utils import unpad_trajectories
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
import numpy as np
from dataclasses import field
# === 新增导入 ===
from rsl_rl.algorithms import PPO
# === 辅助函数：正交初始化 ===
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], activation="elu", output_gain=1.0, use_layer_norm=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act_fn = nn.ELU() if activation == "elu" else nn.ReLU()
        
        for h_dim in hidden_dims:
            layer = nn.Linear(prev_dim, h_dim)
            # === Sanity Check: 暂时注释掉正交初始化，使用默认初始化以对齐 Baseline ===
            orthogonal_init(layer, gain=np.sqrt(2)) 
            layers.append(layer)
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn)
            prev_dim = h_dim
        
        out_layer = nn.Linear(prev_dim, output_dim)
        # === Sanity Check: 暂时注释掉正交初始化 ===
        orthogonal_init(out_layer, gain=output_gain) 
        layers.append(out_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PaperMoEGate(nn.Module):
    """
    匹配 MoE-Loco 论文的 Gating Network
    结构: MLP [128] -> Softmax
    """
    def __init__(self, input_dim, num_experts, hidden_dim=128):
        super().__init__()
        
        # LayerNorm 稳定输入分布
        self.layer_norm = nn.LayerNorm(input_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # === Gating 仍保持正交初始化，因为它需要特定的起始状态（均匀） ===
        orthogonal_init(self.gate_mlp[0], gain=np.sqrt(2))
        orthogonal_init(self.gate_mlp[2], gain=0.01)

    def forward(self, x):
        x = self.layer_norm(x)
        logits = self.gate_mlp(x) 
        weights = F.softmax(logits, dim=-1)
        return weights

class RobustMoEActorCritic(ActorCritic):
    def __init__(self, 
                 obs, 
                 obs_groups, 
                 num_actions, 
                 actor_hidden_dims=[256, 128, 128], 
                 critic_hidden_dims=[512, 256, 128], 
                 activation='elu', 
                 init_noise_std=1.0,
                 num_experts=6,
                 latent_dim=256,
                 rnn_type="gru",
                 separate_gating=True,
                 use_moe_critic=False, 
                 **kwargs):
        
        super().__init__(obs, obs_groups, num_actions, 
                         actor_hidden_dims=actor_hidden_dims, 
                         critic_hidden_dims=critic_hidden_dims, 
                         activation=activation, 
                         init_noise_std=init_noise_std, 
                         **kwargs)

        # Input setup
        self.input_keys = None 
        is_dict_like = isinstance(obs, dict) or hasattr(obs, "keys")
        if is_dict_like:
            keys = obs_groups.get("policy", None)
            if keys is None and "actor" in obs_groups: keys = obs_groups["actor"]
            if keys is None: keys = [next(iter(obs.keys()))]
            self.input_keys = keys
            num_obs = sum(obs[k].shape[-1] for k in keys)
        else:
            num_obs = obs.shape[-1]

        self.latent_dim = latent_dim
        self.rnn_type = rnn_type.lower()
        self.separate_gating = separate_gating
        self.use_moe_critic = use_moe_critic

        # === RNN Backbone Setup ===
        if self.rnn_type == "lstm":
            self.hidden_state_dim = self.latent_dim 
            self.rnn = nn.LSTM(input_size=num_obs, hidden_size=self.latent_dim, batch_first=False)
        else:
            self.hidden_state_dim = self.latent_dim
            self.rnn = nn.GRU(input_size=num_obs, hidden_size=self.latent_dim, batch_first=False)

        # === Sanity Check: 对齐 RNN 初始化 ===
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                if self.rnn_type == "lstm":
                    n = self.latent_dim
                    param.data[n:2*n].fill_(1.0)
        
        self._valid_batch_size = None
        # self.backbone_ln = nn.LayerNorm(self.latent_dim) # Removed
        
        self._obs_latent_cache = None
        # === 新增：初始化权重记录容器 ===
        self.latest_gate_weights = None
        # === Actor MoE ===
        self.actor_gate = PaperMoEGate(input_dim=self.latent_dim, num_experts=num_experts)
        # 注意：MLP 内部现在使用的是默认初始化
        self.actor_experts = nn.ModuleList([
            MLP(self.latent_dim, num_actions, hidden_dims=actor_hidden_dims, activation=activation, output_gain=0.01) 
            for _ in range(num_experts)
        ])
        
        # === Critic Setup (Hybrid) ===
        if self.use_moe_critic:
            if self.separate_gating:
                self.critic_gate = PaperMoEGate(input_dim=self.latent_dim, num_experts=num_experts)
            else:
                self.critic_gate = self.actor_gate
            
            self.critic_experts = nn.ModuleList([
                MLP(self.latent_dim, 1, hidden_dims=critic_hidden_dims, activation=activation, output_gain=1.0) 
                for _ in range(num_experts)
            ])
        else:
            self.critic_mlp = MLP(self.latent_dim, 1, hidden_dims=critic_hidden_dims, activation=activation, output_gain=1.0)

        if is_dict_like: ref_tensor = obs[self.input_keys[0]]
        else: ref_tensor = obs
        batch_size = ref_tensor.shape[0]
        
        multiplier = 2 if self.rnn_type == "lstm" else 1
        self.active_hidden_states = torch.zeros(1, batch_size, multiplier * self.hidden_state_dim, device=ref_tensor.device)

    @property
    def is_recurrent(self):
        return True

    def _prepare_input(self, obs):
        if self.input_keys is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in self.input_keys]
            return torch.cat(tensors, dim=-1)
        return obs
    
    def _run_rnn(self, x_in, hidden_states, masks, valid_batch_size):
        if self.rnn_type == "lstm":
            h_0 = hidden_states[..., :self.latent_dim].contiguous()
            c_0 = hidden_states[..., self.latent_dim:].contiguous()
            rnn_state = (h_0, c_0)
        else:
            h_0 = hidden_states[..., :self.latent_dim].contiguous()
            rnn_state = h_0

        if valid_batch_size is not None:
            if x_in.ndim == 3 and x_in.shape[1] > valid_batch_size: 
                x_in = x_in[:, :valid_batch_size, :]
            elif x_in.ndim == 2 and x_in.shape[0] > valid_batch_size:
                x_in = x_in[:valid_batch_size, :]
            
            if self.rnn_type == "lstm":
                rnn_state = (h_0[:, :valid_batch_size, :], c_0[:, :valid_batch_size, :])
            else:
                rnn_state = h_0[:, :valid_batch_size, :]
                
            if masks is not None and masks.shape[1] > valid_batch_size:
                 masks = masks[:, :valid_batch_size]

        latent = None
        next_rnn_state = None

        if x_in.ndim == 3: # Sequence (Batch Mode)
            # Input: [seq_len, batch, dim]
            rnn_out, next_rnn_state = self.rnn(x_in, rnn_state)
            
            # === 关键修复：使用 unpad_trajectories 还原数据布局 ===
            # rsl_rl 的 PPO 训练数据是 flattened (time-major, masked)，
            # 而不是 padded trajectories。必须调用 unpad 才能对齐。
            if masks is not None:
                latent = unpad_trajectories(rnn_out, masks)
            else:
                latent = rnn_out
                
        elif x_in.ndim == 2: # Step (Inference Mode)
            x_rnn = x_in.unsqueeze(0)
            if masks is not None:
                mask_reshaped = masks.reshape(1, -1, 1)
                if self.rnn_type == "lstm":
                    rnn_state = (rnn_state[0] * mask_reshaped, rnn_state[1] * mask_reshaped)
                else:
                    rnn_state = rnn_state * mask_reshaped
            
            rnn_out, next_rnn_state = self.rnn(x_rnn, rnn_state)
            latent = rnn_out[0]
        
        if self.rnn_type == "lstm":
            next_hidden = torch.cat(next_rnn_state, dim=-1)
        else:
            next_hidden = next_rnn_state

        return latent, next_hidden

    def _compute_actor_output(self, latent):
        if len(self.actor_experts) == 1:
            actions_sum = self.actor_experts[0](latent)
        else:
            weights = self.actor_gate(latent)
            # === 新增：记录当前 Batch 的平均权重用于日志 ===
            # weights shape: (seq_len, batch, num_experts) or (batch, num_experts)
            with torch.no_grad():
                # 展平所有维度 (Batch/Time)，计算每个专家的平均激活值
                flat_weights = weights.reshape(-1, weights.shape[-1])
                self.latest_gate_weights = flat_weights.mean(dim=0).detach()
            # ==========================================
            actions_sum = 0
            for i, expert in enumerate(self.actor_experts):
                w_i = weights[..., i].unsqueeze(-1) 
                actions_sum += expert(latent) * w_i
        return actions_sum

    def forward(self, obs, masks=None, hidden_states=None, save_dist=True, valid_batch_size=None):
        x_in = self._prepare_input(obs)
        if hidden_states is None:
            multiplier = 2 if self.rnn_type == "lstm" else 1
            hidden_states = torch.zeros(1, x_in.shape[0], multiplier*self.latent_dim, device=x_in.device)
            
        latent, next_hidden = self._run_rnn(x_in, hidden_states, masks, valid_batch_size)
        actions_sum = self._compute_actor_output(latent)
            
        if save_dist:
            self.distribution = torch.distributions.Normal(actions_sum, self.std)
        
        return actions_sum, self.std, next_hidden

    def evaluate(self, obs, masks=None, hidden_states=None, hidden_state=None):
        obs = self._prepare_input(obs)
        if self.critic_obs_normalization:
            obs = self.critic_obs_normalizer(obs)

        latent = None
        # Cache Hit Check
        if hidden_states is None and hidden_state is None and self._obs_latent_cache is not None:
            cached_obs, cached_latent = self._obs_latent_cache
            if obs is cached_obs:
                latent = cached_latent
        
        if latent is None:
            if hidden_states is None and hidden_state is not None: 
                hidden_states = hidden_state
            if (masks is not None or hidden_states is not None) and self._valid_batch_size is not None:
                valid_batch_size = self._valid_batch_size
            else:
                valid_batch_size = None
            if hidden_states is None: 
                hidden_states = self.active_hidden_states

            latent, next_hidden = self._run_rnn(obs, hidden_states, masks, valid_batch_size)
        
        if self.use_moe_critic:
            if len(self.critic_experts) == 1:
                value = self.critic_experts[0](latent)
            else:
                weights = self.critic_gate(latent)
                value = 0
                for i, expert in enumerate(self.critic_experts):
                    w_i = weights[..., i].unsqueeze(-1)
                    value += expert(latent) * w_i
        else:
            value = self.critic_mlp(latent)
             
        return value
    
    def get_actions_log_prob(self, actions):
        # === 修复: 移除不正确的 reshaping 逻辑 ===
        # 既然现在 _run_rnn 使用 unpad_trajectories 正确对齐了数据，
        # 这里的 distribution 形状应该已经和 actions 形状一致了。
        # 直接调用 super 或标准逻辑即可。
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act(self, obs, masks=None, hidden_state=None):
        self._valid_batch_size = None
        obs = self._prepare_input(obs)
        if self.actor_obs_normalization: 
            obs = self.actor_obs_normalizer(obs)
        
        current_state = hidden_state[0] if (hidden_state is not None and isinstance(hidden_state, tuple)) else (hidden_state if hidden_state is not None else self.active_hidden_states)
        
        # 1. 运行 RNN
        latent, next_hidden = self._run_rnn(obs, current_state, masks, None)
        
        # 2. 缓存
        if hidden_state is None:
            self._obs_latent_cache = (obs, latent)
            self.active_hidden_states = next_hidden.detach()
        
        # 3. 计算 Actor
        actions_sum = self._compute_actor_output(latent)
        
        self.distribution = torch.distributions.Normal(actions_sum, self.std)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None, hidden_states=None):
        self._valid_batch_size = None
        obs = self._prepare_input(obs)
        if self.actor_obs_normalization: 
            obs = self.actor_obs_normalizer(obs)
            
        if hidden_states is None: hidden_states = self.active_hidden_states
        latent, next_hidden = self._run_rnn(obs, hidden_states, masks, None)
        self.active_hidden_states = next_hidden
        
        return self._compute_actor_output(latent)
    
    def get_hidden_states(self):
        return self.active_hidden_states, self.active_hidden_states

# === 新增：自定义 PPO 算法类 ===
class MoEPPO(PPO):
    """继承标准 PPO，增加 Gate 权重日志记录功能"""
    def update(self):
        # 调用父类 update 执行标准 PPO 训练
        loss_dict = super().update()
        
        # 从 Policy 中读取最新的 Gate 权重并添加到日志字典中
        if hasattr(self.policy, "latest_gate_weights") and self.policy.latest_gate_weights is not None:
            weights = self.policy.latest_gate_weights
            for i, w in enumerate(weights):
                # 这将在 Tensorboard/WandB 中显示为 Loss/Gate_Expert_0 等
                loss_dict[f"Gate/Expert_{i}_weight"] = w.item()
                
        return loss_dict
    
# === Config ===
@configclass
class RobustMoEActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "RobustMoEActorCritic"
    num_experts: int = 6
    latent_dim: int = 256
    rnn_type: str = "gru" 
    
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    
    # === 新增参数 ===
    separate_gating: bool = True
    use_moe_critic: bool = False # 默认关闭，防止梯度爆炸
    
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 128])
    # Critic 维度对齐 Baseline (512, 256, 128)
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])

@configclass
class SiriusRobustMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 10000 
    save_interval = 500
    experiment_name = "sirius_paper_moe_hybrid_lower_noise" # 修改实验名
    empirical_normalization = False 
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    
    policy = RobustMoEActorCriticCfg(
        init_noise_std=0.8, # === 低初始噪声 0.5 ===
        activation="elu",
        num_experts=6,   # === 核心：6 专家 ===
        latent_dim=256,
        rnn_type="gru",
        separate_gating=True,
        use_moe_critic=False, 
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[512, 256, 128], 
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="MoEPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3, # === 稍微降低 LR 以稳定 MoE 训练 ===
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class SiriusMoESanityCheckCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 5000 
    save_interval = 200
    experiment_name = "sirius_moe_sanity_check_1exp" # 明确实验名
    empirical_normalization = False 
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    
    policy = RobustMoEActorCriticCfg(
        init_noise_std=0.8, # === 对齐 Baseline ===
        activation="elu",
        
        # === 核心：单专家 ===
        num_experts=1,
        
        latent_dim=256,
        rnn_type="gru",
        separate_gating=True,
        use_moe_critic=False, 
        
        # === 对齐 Baseline ===
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128], 
        
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="MoEPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        
        # === 对齐 Baseline ===
        learning_rate=1.0e-3, 
        
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )