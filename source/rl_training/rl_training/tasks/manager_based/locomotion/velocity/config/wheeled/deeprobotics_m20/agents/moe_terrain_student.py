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
    RslRlDistillationRunnerCfg,
    RslRlDistillationAlgorithmCfg
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

# [Fix] Modified EmpiricalNormalization for robustness (Supports 3D inputs)
class EmpiricalNormalization(nn.Module):
    def __init__(self, shape, epsilon=1e-4, until_step=None):
        super().__init__()
        self.epsilon = epsilon
        self.until_step = until_step
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def update(self, x):
        # [Fix] Handle 3D inputs (Batch, Time, Dim) from Recurrent Policy
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        # Welford's algorithm
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
# 2. Split MoE Policy Network (Teacher/PPO Version)
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
                 # Option to force a specific input key (useful for StudentTeacher wrapper)
                 forced_input_key=None,
                 **kwargs):
        
        base_kwargs = {k: v for k, v in kwargs.items() if k not in [
            "estimator_output_dim", "estimator_hidden_dims", 
            "estimator_input_indices", "estimator_target_indices", 
            "estimator_obs_normalization", "init_noise_legs", "init_noise_wheels"
        ]}

        super().__init__(obs, obs_groups, num_actions, 
                         actor_hidden_dims=actor_hidden_dims, 
                         critic_hidden_dims=critic_hidden_dims, 
                         activation=activation, 
                         init_noise_std=init_noise_std, 
                         **base_kwargs)

        self.input_keys = None 
        
        # Determine Input Keys and Dimension
        if forced_input_key:
            # Explicit override for StudentTeacher wrapper
            self.input_keys = [forced_input_key]
            
            # [Fix] Logic to handle both dict and TensorDict (which fails isinstance(dict))
            # First try direct access
            try:
                num_obs = obs[forced_input_key].shape[-1]
            except (KeyError, TypeError):
                # Fallback logic
                if hasattr(obs, "keys"):
                     found = False
                     # 1. Check if forced_input_key is in keys (for TensorDict)
                     obs_keys = list(obs.keys())
                     if forced_input_key in obs_keys:
                         num_obs = obs[forced_input_key].shape[-1]
                         found = True
                     
                     # 2. Try to resolve via obs_groups if not found directly
                     if not found and obs_groups and forced_input_key in obs_groups:
                         group_keys = obs_groups[forced_input_key]
                         if group_keys:
                             valid_keys = [k for k in group_keys if k in obs_keys]
                             if valid_keys:
                                 num_obs = sum(obs[k].shape[-1] for k in valid_keys)
                                 found = True
                     
                     # 3. Last resort fallback
                     if not found and len(obs_keys) > 0:
                         # Log warning to help debug dimension mismatches
                         print(f"[SplitMoE] Warning: Forced key '{forced_input_key}' not found in obs {obs_keys}. "
                               f"Fallback to '{obs_keys[0]}'.")
                         num_obs = list(obs.values())[0].shape[-1]
                else:
                     num_obs = obs.shape[-1]

        elif isinstance(obs, dict) or hasattr(obs, "keys"):
            # Standard PPO flow
            keys = obs_groups.get("policy", None)
            self.input_keys = keys
            if keys:
                num_obs = sum(obs[k].shape[-1] for k in keys)
            else:
                num_obs = list(obs.values())[0].shape[-1]
        else:
            num_obs = obs.shape[-1]

        self.latent_dim = latent_dim
        self.rnn_type = rnn_type.lower()
        self.aux_loss_coef = aux_loss_coef
        
        self.num_leg_actions = num_leg_actions
        self.num_wheel_actions = num_actions - num_leg_actions
        
        if self.num_wheel_actions < 0:
            raise ValueError(f"num_leg_actions ({num_leg_actions}) cannot be larger than total actions ({num_actions})")

        print(f"[SplitMoE] Init: Input Dim={num_obs}, Legs={self.num_leg_actions}, Wheels={self.num_wheel_actions}")

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=num_obs, hidden_size=self.latent_dim, batch_first=False)
        else:
            self.rnn = nn.GRU(input_size=num_obs, hidden_size=self.latent_dim, batch_first=False)

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

        # === Estimator Setup ===
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

            self.estimator = MLP(est_input_dim, self.estimator_output_dim, hidden_dims=self.estimator_hidden_dims, activation=activation)
            self.estimator_obs_normalizer = None
            if self.estimator_obs_normalization:
                self.estimator_obs_normalizer = EmpiricalNormalization(shape=[est_input_dim], until_step=1.0e9)
        else:
            self.estimator = None

        # RNN State Init
        if isinstance(obs, dict): ref_tensor = obs[list(obs.keys())[0]]
        else: ref_tensor = obs
        batch_size = ref_tensor.shape[0]
        device = ref_tensor.device
        self.active_hidden_states = self._init_rnn_state(batch_size, device)
        
        self.latest_weights = {}
        self.active_aux_loss = 0.0
        self.active_estimator_loss = 0.0
        self.active_estimator_error = 0.0
        
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

    def _prepare_input(self, obs, key_list):
        if key_list is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in key_list if k in obs] # Check key existence
            if not tensors:
                 # Fallback if key_list keys are missing (e.g. forced_input_key points to 'policy' but obs keys are 'student_policy')
                 # This happens if obs_groups mapping wasn't fully resolved before passing here
                 return list(obs.values())[0]
            return torch.cat(tensors, dim=-1)
        return obs
    
    def _run_rnn(self, rnn_module, x_in, hidden_states, masks):
        if hidden_states is None:
            B = x_in.shape[1] if x_in.ndim == 3 else x_in.shape[0]
            rnn_state = self._init_rnn_state(B, x_in.device)
        else:
            rnn_state = hidden_states

        if x_in.ndim == 3: # Training
            rnn_out, next_rnn_state = rnn_module(x_in, rnn_state)
            if masks is not None: latent = unpad_trajectories(rnn_out, masks)
            else: latent = rnn_out
        elif x_in.ndim == 2: # Inference
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
            except: pass
        if hasattr(obs_dict, "keys") or isinstance(obs_dict, dict):
            try: full_obs = obs_dict["policy"]
            except: full_obs = self._prepare_input(obs_dict, self.input_keys)
        else: full_obs = obs_dict 
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

        # Aux losses ... (Same as before)
        if self.training:
            self.active_aux_loss = self._calculate_load_balancing_loss(w_leg, w_wheel) * self.aux_loss_coef
            if self.estimator is not None and obs_dict is not None:
                est_input = self._get_estimator_input(obs_dict)
                if est_input.shape[-1] == self.estimator.net[0].in_features:
                    # Minimal check for target
                    target_state = None
                    if hasattr(obs_dict, "keys"):
                        if "critic" in obs_dict: 
                            target_src = obs_dict["critic"]
                        else: 
                            target_src = self._prepare_input(obs_dict, self.input_keys)
                        
                        if target_src.shape[-1] > max(self.estimator_target_indices):
                            target_state = target_src[..., self.estimator_target_indices]
                    
                    if target_state is not None:
                        if self.estimator_obs_normalization and self.estimator_obs_normalizer:
                            est_input = self.estimator_obs_normalizer(est_input)
                        estimated = self.estimator(est_input)
                        self.active_estimator_loss = (estimated - target_state).pow(2).mean()
                        self.active_estimator_error = (estimated - target_state).abs().mean().detach()

        return total_action

    def _calculate_load_balancing_loss(self, w_leg, w_wheel):
        # ... same as before
        leg_u = w_leg.reshape(-1, w_leg.shape[-1]).mean(dim=0)
        wheel_u = w_wheel.reshape(-1, w_wheel.shape[-1]).mean(dim=0)
        loss = (leg_u - 1.0/self.num_leg_experts).pow(2).sum() + \
               (wheel_u - 1.0/self.num_wheel_experts).pow(2).sum()
        return loss

    def _prepare_hidden_state(self, hidden, device):
        if hidden is None: return None
        
        # [Fix] Helper to safely move to device and handle None
        def safe_to_device(h):
            if h is None: return None
            return h.to(device).contiguous()

        # Handle GRU specific unpacking (if given as tuple) or LSTM
        if isinstance(hidden, (tuple, list)):
            if self.rnn_type == "gru" and len(hidden) == 2:
                 # GRU hidden state is sometimes passed as (h, c) tuple by generic code where c is None
                 hidden = hidden[0]
            elif self.rnn_type == "lstm" and len(hidden) == 2:
                # Handle potential double wrapping
                if isinstance(hidden[0], (tuple, list)): hidden = hidden[0]

        # Recursive/Iterative processing
        if isinstance(hidden, (tuple, list)):
             return tuple(safe_to_device(x) for x in hidden)
        
        return safe_to_device(hidden)

    def act(self, obs, masks=None, hidden_state=None):
        x_in = self._prepare_input(obs, self.input_keys)
        if self.actor_obs_normalization: x_in = self.actor_obs_normalizer(x_in)
        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_state is None: self.active_hidden_states = next_state
        
        mean = self._compute_actor_output(latent, obs_dict=obs) 
        self.distribution = torch.distributions.Normal(mean, self.std)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None, hidden_states=None):
        x_in = self._prepare_input(obs, self.input_keys)
        if self.actor_obs_normalization: x_in = self.actor_obs_normalizer(x_in)
        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_states is None: self.active_hidden_states = next_state
        return self._compute_actor_output(latent)

    def evaluate(self, obs, masks=None, hidden_state=None):
        # Default PPO behavior: Returns VALUE
        if isinstance(obs, dict) and "critic" in obs: x_in = obs["critic"]
        else: x_in = self._prepare_input(obs, self.input_keys)
        if self.critic_obs_normalization: x_in = self.critic_obs_normalizer(x_in)
        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        latent, _ = self._run_rnn(self.rnn, x_in, current_state, masks)
        return self.critic_mlp(latent)
    
    def get_estimated_state(self, obs, masks=None, hidden_states=None):
        if self.estimator is None: return None # Fail silently
        est_input = self._get_estimator_input(obs)
        if self.estimator_obs_normalization and self.estimator_obs_normalizer:
             est_input = self.estimator_obs_normalizer(est_input)
        return self.estimator(est_input)

    def forward(self, obs, masks=None, hidden_states=None, save_dist=True):
        x_in = self._prepare_input(obs, self.input_keys)
        if self.actor_obs_normalization: x_in = self.actor_obs_normalizer(x_in)
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
    
    def reset(self, dones=None, hidden_states=None):
        # [Fix] Allow setting hidden_states explicitly (required by Distillation)
        if hidden_states is not None:
            # Check if we need to device transfer or formatting
            if hasattr(self, "_prepare_hidden_state") and hasattr(self, "std"):
                 self.active_hidden_states = self._prepare_hidden_state(hidden_states, self.std.device)
            else:
                 self.active_hidden_states = hidden_states

        if dones is None: return

        # [Fix] Ensure mask is boolean (rsl_rl uses ByteTensor which fails in newer PyTorch)
        # Note: torch.byte is deprecated/removed in some versions, use torch.uint8
        if hasattr(dones, "dtype") and dones.dtype == torch.uint8:
            dones = dones.bool()

        def reset_hidden(h, mask):
            if isinstance(h, tuple): return tuple(reset_hidden(x, mask) for x in h)
            else: 
                # [Fix] Avoid in-place update on inference tensor by cloning first
                h = h.clone()
                h[:, mask, :] = 0.0
                return h
        if self.active_hidden_states is not None:
            self.active_hidden_states = reset_hidden(self.active_hidden_states, dones)

# ==============================================================================
# 3. [NEW] Student-Teacher Adapter for rsl_rl Distillation (Pure BC)
# ==============================================================================

class SplitMoEStudentTeacher(nn.Module):
    """
    Adapter class to make SplitMoEActorCritic compatible with rsl_rl.runners.DistillationRunner.
    
    In DistillationRunner:
    - 'act(obs)' calls Student -> Returns Student Actions
    - 'evaluate(obs)' calls Teacher -> Returns Teacher Actions (as targets)
    
    It does NOT use a Critic.
    """
    is_recurrent = True
    loaded_teacher = False

    def __init__(self, obs, obs_groups, num_actions, activation="elu", **kwargs):
        super().__init__()
        self.obs_groups = obs_groups
        
        # 1. Initialize Student (MoE) - Uses "policy" obs group (Student Obs)
        print("\n[Distillation] Initializing Student Policy (SplitMoE)...")
        
        # [Fix] Sanitize activation if it comes as a dict (config artifact?)
        if isinstance(activation, dict):
            print(f"[Warning] 'activation' passed as dict: {activation}. Using 'elu' as default.")
            activation = activation.get("value", "elu") if "value" in activation else "elu"
            
        # We must filter kwargs to ensure clean init for sub-modules
        student_kwargs = kwargs.copy()
        student_kwargs["estimator_output_dim"] = 0 # No estimator for student in BC
        
        # [Fix] Construct valid obs_groups for Student (must have 'critic' for ActorCritic init)
        # We map 'critic' to the same input as 'policy' since we don't train value function in BC
        student_obs_groups = {"policy": obs_groups["policy"]}
        student_obs_groups["critic"] = obs_groups["policy"]
        # Use the actual key name from the list (e.g., 'blind_student_policy')
        student_input_key = obs_groups["policy"][0]

        self.student = SplitMoEActorCritic(
            obs, 
            student_obs_groups, 
            num_actions, 
            forced_input_key=student_input_key, 
            activation=activation, # Pass sanitized activation
            **student_kwargs
        )

        # 2. Initialize Teacher (MoE) - Uses "teacher" obs group (Privileged Obs)
        print("\n[Distillation] Initializing Teacher Policy (SplitMoE)...")
        teacher_kwargs = kwargs.copy()
        
        # [Fix] Construct valid obs_groups for Teacher
        # Map 'teacher' group to 'policy' and 'critic' for the teacher instance
        teacher_source = obs_groups["teacher"]
        teacher_obs_groups = {"policy": teacher_source, "critic": teacher_source}
        teacher_input_key = teacher_source[0]
        
        self.teacher = SplitMoEActorCritic(
            obs, 
            teacher_obs_groups, 
            num_actions, 
            forced_input_key=teacher_input_key, 
            activation=activation, # Pass sanitized activation
            **teacher_kwargs
        )
        
        # Ensure Teacher is in Eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    @property
    def action_std(self):
        return self.student.std

    def reset(self, dones=None, hidden_states=None):
        # [Fix] Pass hidden_states to student (as get_hidden_states returns student states)
        self.student.reset(dones, hidden_states=hidden_states)
        self.teacher.reset(dones)

    def act(self, obs, masks=None, hidden_state=None):
        # Called by RolloutStorage to collect student data
        return self.student.act(obs, masks, hidden_state)

    def act_inference(self, obs):
        # Called during update to compute gradient
        return self.student.act_inference(obs)

    def evaluate(self, obs):
        # Called by DistillationRunner to get targets (Teacher Actions)
        # We use 'act_inference' of the teacher because we want deterministic actions (means)
        with torch.no_grad():
            return self.teacher.act_inference(obs)
    
    def get_hidden_states(self):
        return self.student.get_hidden_states()
    
    def detach_hidden_states(self, dones=None):
        # Helper for recurrent logic in DistillationRunner
        pass
        # Note: SplitMoEActorCritic doesn't have detach_hidden_states exposed like this usually,
        # but RSL-RL Distillation might call it. 
        # However, looking at the error log, the crash was at reset(), so detach might be fine or ignored if not called.
        # If needed, we can implement it:
        if self.student.active_hidden_states is not None:
             def detach_recursive(h):
                 if isinstance(h, tuple): return tuple(detach_recursive(x) for x in h)
                 return h.detach()
             self.student.active_hidden_states = detach_recursive(self.student.active_hidden_states)

    def update_normalization(self, obs):
        self.student.update_normalization(obs)
        # Teacher norm is frozen usually, but if we need to update running stats:
        # self.teacher.update_normalization(obs) 
    
    def load_state_dict(self, state_dict, strict=True):
        # Custom loading logic to handle "teacher only" load vs "resume student" load
        
        # 1. Check if loading Teacher (common starting point)
        # RSL-RL convention: keys might be prefixed or not.
        # If we load a standard PPO checkpoint, keys are like "actor.xxx", "rnn.xxx"
        # We want to load these into self.teacher
        
        print(f"[Distillation] Loading state dict with {len(state_dict)} keys...")
        
        # Attempt to load into Teacher
        # Filter keys: PPO checkpoints usually have model_state_dict with top-level keys
        # If the checkpoint matches SplitMoEActorCritic structure exactly
        try:
            self.teacher.load_state_dict(state_dict, strict=False)
            print("[Distillation] Successfully loaded Teacher weights.")
            self.loaded_teacher = True
        except Exception as e:
            print(f"[Distillation] Warning: Direct load to teacher failed: {e}")

        # If resuming distillation, state_dict might have "student." and "teacher." prefixes
        if any(k.startswith("student.") for k in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            print("[Distillation] Resumed Student-Teacher training.")
            self.loaded_teacher = True
        
        # If we just loaded teacher, we might want to copy weights to student as a starting point?
        # Typically BC starts student from random or copy. 
        # If strict BC, random is fine.
        
        return True

# ==============================================================================
# 4. PPO Algorithm Wrapper (Kept for reference or RL-FineTuning)
# ==============================================================================

class SplitMoEPPO(PPO):
    def update(self):
        loss_dict = super().update()
        model = getattr(self, "actor_critic", getattr(self, "policy", None))
        if model is not None and self.num_learning_epochs > 0:
            obs_batch = self.storage.observations.flatten(0, 1)
            total_aux_loss = 0.0
            
            if model.estimator is not None:
                est_input = model._get_estimator_input(obs_batch)
                if est_input.shape[-1] == model.estimator.net[0].in_features:
                    target = obs_batch.get("critic", model._prepare_input(obs_batch, model.input_keys)) if isinstance(obs_batch, dict) else obs_batch
                    if target.shape[-1] > max(model.estimator_target_indices):
                        t_state = target[..., model.estimator_target_indices]
                        if model.estimator_obs_normalization and model.estimator_obs_normalizer:
                            est_input = model.estimator_obs_normalizer(est_input)
                        est = model.estimator(est_input)
                        l = (est - t_state).pow(2).mean()
                        total_aux_loss += l
                        model.active_estimator_loss = l.detach()
                        model.active_estimator_error = (est - t_state).abs().mean().detach()

            if isinstance(total_aux_loss, torch.Tensor) and total_aux_loss.requires_grad:
                self.optimizer.zero_grad()
                total_aux_loss.backward()
                if self.max_grad_norm: nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        if model:
            if hasattr(model, "latest_weights") and model.latest_weights:
                w = model.latest_weights
                if "leg" in w:
                    for i, val in enumerate(w["leg"]): loss_dict[f"Gate/Leg_Expert_{i}"] = val.item()
                if "wheel" in w:
                    for i, val in enumerate(w["wheel"]): loss_dict[f"Gate/Wheel_Expert_{i}"] = val.item()
            if hasattr(model, "active_estimator_loss"): loss_dict["Loss/Estimator_MSE"] = model.active_estimator_loss
            if hasattr(model, "active_estimator_error"): loss_dict["Loss/Estimator_Error_MAE"] = model.active_estimator_error
            
        return loss_dict

# ==============================================================================
# 5. Configs
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

    # === Estimator Settings ===
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
    num_steps_per_env = 64
    max_iterations = 20000
    save_interval = 200
    experiment_name = "split_moe_parallel" ▒▒▒▒
  Vy: Est= 0.047 | GT= 0.023 | Err= 0.024 

    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "estimator": ["estimator"]}
    
    policy = SplitMoEActorCriticCfg(
        # ... standard params ...
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



# ==============================================================================
# 6. [NEW] Distillation Config (Pure BC)
# ==============================================================================
from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg, RslRlDistillationAlgorithmCfg

@configclass
class SplitMoEDistillationCfg(RslRlDistillationRunnerCfg):
    """
    Configuration for Student-Teacher Distillation (Behavior Cloning).
    Uses 'SplitMoEStudentTeacher' to wrap two SplitMoE instances.
    """
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "split_moe_distill"
    empirical_normalization = False

    # Key mapping: 
    # 'policy' -> Student inputs (student_policy)
    # 'teacher' -> Teacher inputs (policy/critic/privileged from teacher env)
    # Ensure 'teacher' group is defined in your EnvCfg! (e.g. creating a group that includes priv info)
    # Assuming EnvCfg has "student_policy" and "policy" (which is teacher's full obs)
    obs_groups = {"policy": ["blind_student_policy"], "teacher": ["policy"]} 

    # Wrapper class that holds both student and teacher
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
        # Student params (Estimator disabled for student in BC)
        estimator_output_dim=0,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1.0e-4,
    )

@configclass
class SplitMoESenseDistillationCfg(RslRlDistillationRunnerCfg):
    """
    Configuration for Student-Teacher Distillation (Behavior Cloning).
    Uses 'SplitMoEStudentTeacher' to wrap two SplitMoE instances.
    """
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "split_moe_distill_sense"
    empirical_normalization = False

    # Key mapping: 
    # 'policy' -> Student inputs (student_policy)
    # 'teacher' -> Teacher inputs (policy/critic/privileged from teacher env)
    # Ensure 'teacher' group is defined in your EnvCfg! (e.g. creating a group that includes priv info)
    # Assuming EnvCfg has "student_policy" and "policy" (which is teacher's full obs)
    obs_groups = {"policy": ["student_policy"], "teacher": ["policy"]} 

    # Wrapper class that holds both student and teacher
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
        # Student params (Estimator disabled for student in BC)
        estimator_output_dim=0,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1.0e-4,
    )