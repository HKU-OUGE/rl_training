# source/robot_lab/algo/moe/moe_split_cross.py

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
        self.mean[:] = new_mean
        self.var[:] = M2 / tot_count
        if self.count.ndim == 0:
            self.count.fill_(tot_count)
        else:
            self.count[:] = tot_count

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
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
    def forward(self, x): return self.net(x)

# ==============================================================================
# 2. Cross-Observation Dual-RNN MoE Policy
# ==============================================================================

class CrossMoEActorCritic(ActorCritic):
    is_recurrent = True

    def __init__(self, obs, obs_groups, num_actions, 
                 actor_hidden_dims=[256, 128, 128], 
                 critic_hidden_dims=[512, 256, 128], 
                 activation='elu', 
                 init_noise_std=1.0,
                 # === 双支路 MoE 参数 ===
                 num_wheel_experts=4, 
                 num_leg_experts=4, 
                 num_leg_actions=12,
                 latent_dim_leg=256, 
                 latent_dim_wheel=64,
                 rnn_type="gru",
                 aux_loss_coef=0.01,
                 **kwargs):
        
        # 排除自定义参数，防止父类冲突
        base_kwargs = {k: v for k, v in kwargs.items() if k not in [
            "estimator_output_dim", "estimator_hidden_dims", 
            "estimator_input_indices", "estimator_target_indices", 
            "estimator_obs_normalization", "init_noise_legs", "init_noise_wheels"
        ]}

        # [Fix 1] 使用关键字参数调用 super，修复 TypeError
        super().__init__(
            obs, 
            obs_groups, 
            num_actions, 
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **base_kwargs
        )

        # [Fix 2] 兼容 TensorDict 的输入判断
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            self.input_keys = obs_groups.get("policy", ["policy"])
            self.num_obs = sum(obs[k].shape[-1] for k in self.input_keys)
        else:
            self.input_keys = None
            self.num_obs = obs.shape[-1]

        self.num_leg_actions = num_leg_actions
        self.num_wheel_actions = num_actions - num_leg_actions
        self.latent_dim_leg = latent_dim_leg
        self.latent_dim_wheel = latent_dim_wheel
        self.rnn_type = rnn_type.lower()
        self.aux_loss_coef = aux_loss_coef

        # 2. 独立双 RNN 支路初始化
        RNN_CLASS = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn_leg = RNN_CLASS(self.num_obs, latent_dim_leg, batch_first=False)
        self.rnn_wheel = RNN_CLASS(self.num_obs, latent_dim_wheel, batch_first=False)
        
        for rnn in [self.rnn_leg, self.rnn_wheel]:
            for name, p in rnn.named_parameters():
                if 'weight' in name: nn.init.orthogonal_(p)
                elif 'bias' in name: nn.init.constant_(p, 0)

        # 3. 门控与专家 (Cross-Observation)
        # 输入维度 = 本支路 Latent + 对方支路 Latent (Detach)
        self.leg_gate = nn.Sequential(nn.Linear(latent_dim_leg + latent_dim_wheel, 64), nn.ELU(), nn.Linear(64, num_leg_experts))
        self.wheel_gate = nn.Sequential(nn.Linear(latent_dim_wheel + latent_dim_leg, 64), nn.ELU(), nn.Linear(64, num_wheel_experts))
        
        self.actor_leg_experts = nn.ModuleList([
            MLP(latent_dim_leg + latent_dim_wheel, self.num_leg_actions, actor_hidden_dims, activation, 0.01) 
            for _ in range(num_leg_experts)
        ])
        self.actor_wheel_experts = nn.ModuleList([
            MLP(latent_dim_wheel + latent_dim_leg, self.num_wheel_actions, actor_hidden_dims, activation, 0.01) 
            for _ in range(num_wheel_experts)
        ])

        # 4. 评论家 (融合全局特征)
        self.critic_mlp = MLP(latent_dim_leg + latent_dim_wheel, 1, critic_hidden_dims, activation)

        # 5. 状态估计器
        self._setup_estimator(obs, kwargs, activation)

        # 6. 状态初始化
        self.active_h_leg = None
        self.active_h_wheel = None
        
        # 7. 噪声与指标
        self._setup_noise(num_actions, kwargs)
        self.latest_weights = {}
        self.active_aux_loss = 0.0
        self.active_estimator_loss = 0.0
        self.active_estimator_error = 0.0

    def _setup_estimator(self, obs, kwargs, activation):
        self.estimator_output_dim = kwargs.get("estimator_output_dim", 0)
        self.estimator_input_indices = kwargs.get("estimator_input_indices", list(range(3, 32)))
        self.estimator_target_indices = kwargs.get("estimator_target_indices", [0, 1, 2])
        
        # [Fix] 显式保存 normalization 开关，防止 update 时访问报错
        self.estimator_obs_normalization = kwargs.get("estimator_obs_normalization", True)

        if self.estimator_output_dim > 0:
            # 尝试从 TensorDict 中获取 estimator 组
            try:
                est_input_dim = obs["estimator"].shape[-1]
                self.has_estimator_group = True
            except:
                est_input_dim = len(self.estimator_input_indices)
                self.has_estimator_group = False
            self.estimator = MLP(est_input_dim, self.estimator_output_dim, kwargs.get("estimator_hidden_dims", [128, 64]), activation)
            
            # 使用保存的 self.estimator_obs_normalization
            self.estimator_obs_normalizer = EmpiricalNormalization([est_input_dim]) if self.estimator_obs_normalization else None
        else: 
            self.estimator = None
            self.estimator_obs_normalizer = None

    def _setup_noise(self, num_actions, kwargs):
        new_std = torch.ones(num_actions)
        noise_legs = kwargs.get("init_noise_legs", 1.0)
        noise_wheels = kwargs.get("init_noise_wheels", 0.5)
        new_std[:self.num_leg_actions] = noise_legs
        new_std[self.num_leg_actions:] = noise_wheels
        self.std.data.copy_(new_std)

    def _prepare_input(self, obs, key_list):
        # [Fix 3] 兼容 TensorDict 的数据提取
        if key_list is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            return torch.cat([obs[k] for k in key_list], dim=-1)
        return obs
        
    def _get_estimator_input(self, obs_dict):
        # Helper function for Estimator input
        if self.has_estimator_group:
            try: return obs_dict["estimator"]
            except: pass
        
        if hasattr(obs_dict, "keys") or isinstance(obs_dict, dict):
            try: full_obs = obs_dict["policy"]
            except KeyError: full_obs = self._prepare_input(obs_dict, self.input_keys)
        else:
            full_obs = obs_dict
        return full_obs[..., self.estimator_input_indices]

    def _run_dual_rnn(self, x_in, masks, h_leg, h_wheel):
        is_inference = x_in.ndim == 2
        x_rnn = x_in.unsqueeze(0) if is_inference else x_in
        
        # 处理 masks 与 RNN 状态重置 (Inference)
        if is_inference and masks is not None:
            def apply_mask(h, m):
                if isinstance(h, tuple): return tuple(x * m for x in h)
                return h * m
            m = masks.view(1, -1, 1)
            if h_leg is not None: h_leg = apply_mask(h_leg, m)
            if h_wheel is not None: h_wheel = apply_mask(h_wheel, m)

        l_leg, next_h_leg = self.rnn_leg(x_rnn, h_leg)
        l_whl, next_h_whl = self.rnn_wheel(x_rnn, h_wheel)

        if not is_inference and masks is not None:
            l_leg = unpad_trajectories(l_leg, masks)
            l_whl = unpad_trajectories(l_whl, masks)
        elif is_inference:
            l_leg, l_whl = l_leg[0], l_whl[0]

        return l_leg, l_whl, next_h_leg, next_h_whl

    def _compute_actor_output(self, l_leg, l_whl, obs_dict=None):
        l_leg_d, l_whl_d = l_leg.detach(), l_whl.detach()

        # 腿部：主要靠腿部 Latent，参考轮部 (Detach)
        leg_ctx = torch.cat([l_leg, l_whl_d], dim=-1)
        w_leg = F.softmax(self.leg_gate(leg_ctx), dim=-1)
        leg_act = sum(exp(leg_ctx) * w_leg[..., i:i+1] for i, exp in enumerate(self.actor_leg_experts))

        # 轮部：主要靠轮部 Latent，参考腿部 (Detach)
        whl_ctx = torch.cat([l_whl, l_leg_d], dim=-1)
        w_whl = F.softmax(self.wheel_gate(whl_ctx), dim=-1)
        whl_act = sum(exp(whl_ctx) * w_whl[..., i:i+1] for i, exp in enumerate(self.actor_wheel_experts))

        if self.training:
            self.active_aux_loss = self._calc_load_balance(w_leg, w_whl) * self.aux_loss_coef
            if self.estimator is not None and obs_dict is not None:
                self._update_estimator_loss(obs_dict)
        
        with torch.no_grad():
            self.latest_weights = {
                "leg": w_leg.reshape(-1, w_leg.shape[-1]).mean(0),
                "wheel": w_whl.reshape(-1, w_whl.shape[-1]).mean(0)
            }
        
        return torch.cat([leg_act, whl_act], dim=-1)

    def _calc_load_balance(self, w_l, w_w):
        def ent(w):
            usage = w.reshape(-1, w.shape[-1]).mean(0)
            return (usage - 1.0/w.shape[-1]).pow(2).sum()
        return ent(w_l) + ent(w_w)

    def _update_estimator_loss(self, obs_dict):
        est_input = self._get_estimator_input(obs_dict)
        
        if hasattr(obs_dict, "keys") or isinstance(obs_dict, dict):
            try: full_obs = obs_dict["policy"]
            except KeyError: full_obs = self._prepare_input(obs_dict, self.input_keys)
        else: full_obs = obs_dict

        target = full_obs[..., self.estimator_target_indices]
        
        if self.estimator_obs_normalizer: est_input = self.estimator_obs_normalizer(est_input)
        pred = self.estimator(est_input)
        self.active_estimator_loss = (pred - target).pow(2).mean()
        self.active_estimator_error = (pred - target).abs().mean().detach()

    def act(self, obs, masks=None, hidden_states=None, hidden_state=None):
        # 1. 兼容多种 hidden_state 传参方式
        h_in = hidden_states if hidden_states is not None else hidden_state
        
        # 2. 准备 RNN 初始状态
        h_leg, h_wheel = (None, None)
        if h_in is not None:
             # 如果外部传入了状态（比如在 PPO 更新循环中），直接使用
             h_leg, h_wheel = h_in
        else:
             # 如果外部没传（比如在环境交互 Rollout 中），使用内部维护的状态
             h_leg = self.active_h_leg
             h_wheel = self.active_h_wheel

        # 3. 准备输入数据
        x_in = self._prepare_input(obs, self.input_keys)
        if self.actor_obs_normalization: x_in = self.actor_obs_normalizer(x_in)
        
        # 4. 运行双 RNN
        l_leg, l_whl, next_h_leg, next_h_wheel = self._run_dual_rnn(x_in, masks, h_leg, h_wheel)
        
        # 5. 更新内部状态 (关键修复)
        # 只有在 masks 为 None 时（表示正在进行时序推进，而非批量梯度计算）才更新状态
        if masks is None:
            self.active_h_leg = next_h_leg
            self.active_h_wheel = next_h_wheel
            
        # 6. 计算动作分布
        mean = self._compute_actor_output(l_leg, l_whl, obs_dict=obs)
        self.distribution = torch.distributions.Normal(mean, self.std)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None, hidden_states=None):
        return self.act(obs, masks, hidden_states)

    def evaluate(self, obs, masks=None, hidden_states=None, hidden_state=None):
        h_in = hidden_states if hidden_states is not None else hidden_state
        h_leg, h_wheel = (None, None)
        if h_in is not None:
             h_leg, h_wheel = h_in
             
        x_in = self._prepare_input(obs, self.input_keys)
        if self.actor_obs_normalization: x_in = self.actor_obs_normalizer(x_in)
        
        # 对于 Critic，不需要更新 hidden state，只用于前向计算
        l_leg, l_whl, _, _ = self._run_dual_rnn(x_in, masks, h_leg, h_wheel)
        return self.critic_mlp(torch.cat([l_leg, l_whl], dim=-1))
        
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        if dones is not None:
            def reset_h(h):
                if h is None: return None
                if isinstance(h, tuple): return tuple(reset_h(x) for x in h)
                h[:, dones, :] = 0.0
                return h
            self.active_h_leg = reset_h(self.active_h_leg)
            self.active_h_wheel = reset_h(self.active_h_wheel)

    def get_hidden_states(self):
        # [Fix] 这里的返回结构非常关键！
        
        # 如果内部状态尚未初始化 (Step 0)，直接返回 (None, None)
        # 这样 RolloutStorage 会跳过保存，缓冲区默认保持为 0，符合 RNN 初始状态为 0 的逻辑。
        if self.active_h_leg is None or self.active_h_wheel is None:
            return None, None
            
        # 如果已有状态，必须打包成 Tuple 伪装成 LSTM 格式，
        # 以便 rsl_rl 保存双支路 (Leg, Wheel) 的状态。
        combined_state = (self.active_h_leg, self.active_h_wheel)
        return combined_state, combined_state

# ==============================================================================
# 3. PPO Algorithm (修复 Aux Loss 和 解包问题)
# ==============================================================================

# source/robot_lab/algo/moe/moe_split_cross.py

class SplitMoEPPO(PPO):
    def update(self):
        # === 0. 获取模型实例 ===
        model = getattr(self, "policy", getattr(self, "actor_critic", None))
        
        # === 1. 初始化统计数据 ===
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_aux_loss = 0  
        mean_est_loss = 0  
        
        # === 2. 获取生成器 ===
        if model.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        # === 3. 完整的 PPO 训练循环 ===
        for sample in generator:
            # Recurrent Generator 返回 10 个变量
            if model.is_recurrent:
                (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hidden_states_batch,
                    masks_batch,
                ) = sample
                critic_obs_batch = obs_batch 
            else:
                # 非 Recurrent 兼容
                if len(sample) == 11:
                    (
                        obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, 
                        returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, 
                        hidden_states_batch, masks_batch
                    ) = sample
                else:
                    (
                        obs_batch, actions_batch, target_values_batch, advantages_batch, 
                        returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, 
                        hidden_states_batch, masks_batch
                    ) = sample
                    critic_obs_batch = obs_batch

            # --- A. 前向传播 ---
            curr_actor_hid = hidden_states_batch[0] if hidden_states_batch else None
            curr_critic_hid = hidden_states_batch[1] if hidden_states_batch else None
            
            model.act(obs_batch, masks=masks_batch, hidden_state=curr_actor_hid)
            actions_log_prob_batch = model.get_actions_log_prob(actions_batch)
            value_batch = model.evaluate(obs_batch, masks=masks_batch, hidden_state=curr_critic_hid)
            
            # --- B. 获取 MoE 负载均衡 Loss (仅作为参考，不参与 Backward) ---
            aux_loss = model.active_aux_loss
            
            # --- C. 计算 PPO 标准 Loss ---
            mu_batch = model.distribution.mean
            sigma_batch = model.distribution.stddev
            entropy_batch = model.distribution.entropy().sum(dim=-1)

            # Adaptive LR
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                        
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                            
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                        
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate Loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value Loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (value_batch - returns_batch).pow(2).mean()

            # --- D. 计算 Estimator Loss ---
            est_loss = 0.0
            if model.estimator is not None:
                est_input = model._prepare_input(obs_batch, model.input_keys) 
                if hasattr(model, "_get_estimator_input"):
                     est_input = model._get_estimator_input(obs_batch)
                
                if hasattr(obs_batch, "keys") or isinstance(obs_batch, dict):
                     try: full_obs = obs_batch["policy"]
                     except KeyError: full_obs = model._prepare_input(obs_batch, model.input_keys)
                else: full_obs = obs_batch
                target = full_obs[..., model.estimator_target_indices]

                # 修复 update 时可能访问不到 normalization 属性的问题
                if getattr(model, "estimator_obs_normalization", False) and model.estimator_obs_normalizer:
                    est_input = model.estimator_obs_normalizer(est_input)
                
                pred = model.estimator(est_input)
                est_loss = (pred - target).pow(2).mean()
                model.active_estimator_loss = est_loss.detach()

            # === E. 总 Loss 聚合与反向传播 ===
            # [Mod] 移除了 aux_loss，它现在只用于记录，不产生梯度
            loss = surrogate_loss + \
                   self.value_loss_coef * value_loss - \
                   self.entropy_coef * entropy_batch.mean() + \
                   est_loss 

            self.optimizer.zero_grad()
            loss.backward()

            # 监控双 GRU 的梯度流
            rnn_leg_grad = 0.0
            for p in model.rnn_leg.parameters():
                if p.grad is not None: rnn_leg_grad += p.grad.data.norm(2).item() ** 2
            rnn_leg_grad = rnn_leg_grad ** 0.5

            rnn_wheel_grad = 0.0
            for p in model.rnn_wheel.parameters():
                if p.grad is not None: rnn_wheel_grad += p.grad.data.norm(2).item() ** 2
            rnn_wheel_grad = rnn_wheel_grad ** 0.5

            nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # 统计
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            mean_est_loss += est_loss.item() if isinstance(est_loss, torch.Tensor) else est_loss
            
            last_rnn_leg_grad = rnn_leg_grad
            last_rnn_wheel_grad = rnn_wheel_grad

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_aux_loss /= num_updates
        mean_est_loss /= num_updates
        
        self.storage.clear()

        # === 3. Return Logs ===
        loss_dict = {
            "Loss/value_function": mean_value_loss,
            "Loss/surrogate": mean_surrogate_loss,
            "Loss/learning_rate": self.learning_rate,
            "Loss/Load_Balancing": mean_aux_loss, # 依然记录，方便观察 Gate 行为
            "Loss/Estimator_MSE": mean_est_loss,
            "Grad/RNN_Leg": last_rnn_leg_grad,
            "Grad/RNN_Wheel": last_rnn_wheel_grad,
        }
        
        # 轮腿独立噪声日志
        if hasattr(model, "std"):
            std_np = model.std.detach().cpu().numpy()
            n_legs = getattr(model, "num_leg_actions", 12)
            if len(std_np) >= n_legs:
                loss_dict["Noise/Leg_Std"] = std_np[:n_legs].mean()
                if len(std_np) > n_legs:
                    loss_dict["Noise/Wheel_Std"] = std_np[n_legs:].mean()
                else:
                    loss_dict["Noise/Wheel_Std"] = 0.0
        
        if model.latest_weights:
            for k, weights in model.latest_weights.items():
                for i, v in enumerate(weights):
                    loss_dict[f"Gate/{k.capitalize()}_Exp_{i}"] = v.item()

        return loss_dict
# ==============================================================================
# 4. Configs (高度可配置版)
# ==============================================================================

@configclass
class CrossMoEActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "CrossMoEActorCritic"
    # --- MoE 核心配置 ---
    num_wheel_experts: int = 6
    num_leg_experts: int = 6
    num_leg_actions: int = 12
    # --- 差异化双支路配置 ---
    latent_dim_leg: int = 256
    latent_dim_wheel: int = 64
    rnn_type: str = "gru" # 可选 "gru" 或 "lstm"
    # --- 网络深度配置 ---
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    # --- 噪声与损失配置 ---
    init_noise_std: float = 1.0
    init_noise_legs: float = 1.0
    init_noise_wheels: float = 0.5
    aux_loss_coef: float = 0.01
    # --- 估计器配置 ---
    estimator_output_dim: int = 3
    estimator_hidden_dims: list = field(default_factory=lambda: [128, 64])
    estimator_target_indices: list = field(default_factory=lambda: [0, 1, 2])
    estimator_input_indices: list = field(default_factory=lambda: list(range(3, 9)) + list(range(12, 56)))
    estimator_obs_normalization: bool = True
    # --- 归一化开关 ---
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True

    def get_std(self):
        if not hasattr(self, 'std'): return 0.0, 0.0
        
        std_np = self.std.detach().cpu().numpy()
        leg_std = std_np[:self.num_leg_actions].mean() if self.num_leg_actions > 0 else 0.0
        wheel_std = std_np[self.num_leg_actions:].mean() if len(std_np) > self.num_leg_actions else 0.0
        return leg_std, wheel_std

@configclass
class CrossMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 5000
    save_interval = 200
    experiment_name = "cross_moe_v1"
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"]}
    
    policy = CrossMoEActorCriticCfg()
    
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