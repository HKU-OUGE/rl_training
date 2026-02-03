# play_moe_split.py

import argparse
import sys
import os
import glob
import json
import yaml
from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === 1. 启动 App ===
parser = argparse.ArgumentParser(description="Play Cross-MoE Policy and Export")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoE-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
# MoE Params
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)
# Checkpoint
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default="model_*.pt")
# Keyboard/Joystick
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--joystick", action="store_true", default=False, help="Whether to use joystick/gamepad.")
# Export
parser.add_argument("--export", action="store_true", default=True, help="Whether to export ONNX/TorchScript and Configs.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==============================================================================
#  依赖导入 (必须在 simulation_app 之后)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from rl_utils import camera_follow 
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.devices import Se2Keyboard
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm

# === [关键] 导入 CrossMoE 模块 ===
try:
    sys.path.append(os.getcwd())
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_split_cross import CrossMoEActorCritic, SplitMoEPPO
    print("[Info] Imported Cross-MoE classes from current directory.")
except ImportError:
    try:
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_split_cross import CrossMoEActorCritic, SplitMoEPPO
        print("[Info] Imported Cross-MoE classes from project path.")
    except ImportError:
        raise ImportError("Could not import CrossMoEActorCritic from moe_split_cross.py")

# === 注入到 RSL-RL ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

# 注入 CrossMoE 类
rsl_modules.CrossMoEActorCritic = CrossMoEActorCritic
runner_module.CrossMoEActorCritic = CrossMoEActorCritic
# 兼容旧配置名
rsl_modules.SplitMoEActorCritic = CrossMoEActorCritic 
runner_module.SplitMoEPPO = SplitMoEPPO

# ==============================================================================
#  Joystick Controller
# ==============================================================================
import struct
import threading

class DirectLinuxGamepad:
    def __init__(self, device_path="/dev/input/js0", x_scale=1.0, y_scale=1.0, w_scale=1.0, deadzone=0.1):
        self.device_path = device_path
        self.running = True
        self.axes = {}
        self.buttons = {}
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.w_scale = w_scale
        self.deadzone = deadzone
        
        try:
            self.js_file = open(self.device_path, "rb")
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"[DirectGamepad] Successfully opened {device_path}")
        except FileNotFoundError:
            print(f"[Error] Could not open {device_path}. Joystick disabled.")
            self.running = False

    def _read_loop(self):
        event_size = 8
        while self.running:
            try:
                event_data = self.js_file.read(event_size)
                if not event_data: break
                time_ms, value, type_, number = struct.unpack("Ihbb", event_data)
                
                if type_ == 2: # Axis
                    norm_val = value / 32767.0
                    if abs(norm_val) < self.deadzone: norm_val = 0.0
                    self.axes[number] = norm_val
                elif type_ == 1: # Button
                    self.buttons[number] = (value == 1)
            except Exception as e:
                break

    def advance(self):
        if not self.running: return torch.zeros(3)
        # Xbox/Logitech mapping: Axis 1=LY, Axis 0=LX, Axis 4=RY, Axis 3=RX
        # D-Pad: Axis 7=DY, Axis 6=DX
        stick_vx = -self.axes.get(1, 0.0) 
        stick_vy = -self.axes.get(0, 0.0)
        dpad_vx = -self.axes.get(7, 0.0) 
        dpad_vy = -self.axes.get(6, 0.0)

        if abs(dpad_vx) > 0.1 or abs(dpad_vy) > 0.1:
            vx = dpad_vx * self.x_scale
            vy = dpad_vy * self.y_scale
        else:
            vx = stick_vx * self.x_scale
            vy = stick_vy * self.y_scale

        wz = -self.axes.get(3, 0.0) * self.w_scale
        return torch.tensor([vx, vy, wz])

    def add_callback(self, btn_index, func):
        pass

# ==============================================================================
#  Export Wrappers (Dual RNN Support)
# ==============================================================================

class ExportablePolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.eval()

    def forward(self, obs, hidden_states=None):
        # CrossMoE 的 forward 返回 (mean, std, (h_leg, h_wheel))
        # 我们只需要 mean 和 next_hidden
        action_mean, _, next_state = self.policy.forward(
            obs, masks=None, hidden_states=hidden_states, save_dist=False
        )
        return action_mean, next_state

class ExportableEstimator(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.estimator = policy.estimator
        self.normalizer = policy.estimator_obs_normalizer
        self.estimator.eval()
        if self.normalizer: self.normalizer.eval()

    def forward(self, obs):
        if self.normalizer is not None: obs = self.normalizer(obs)
        return self.estimator(obs)

# ==============================================================================
#  Helpers
# ==============================================================================

def resolve_checkpoint_path(root_log_dir, run_name_or_path, checkpoint_pattern):
    run_dir = None
    if run_name_or_path is None:
        if not os.path.exists(root_log_dir):
             parent = os.path.dirname(root_log_dir)
             if os.path.exists(parent): root_log_dir = parent
        if not os.path.exists(root_log_dir): raise FileNotFoundError(f"Log dir not found: {root_log_dir}")
        all_runs = [os.path.join(root_log_dir, d) for d in os.listdir(root_log_dir) if os.path.isdir(os.path.join(root_log_dir, d))]
        if not all_runs: raise FileNotFoundError("No runs found")
        all_runs.sort(key=os.path.getmtime)
        run_dir = all_runs[-1]
        print(f"[Info] Auto-selected run: {os.path.basename(run_dir)}")
    elif os.path.isabs(run_name_or_path):
        run_dir = run_name_or_path
    else:
        run_dir = os.path.join(root_log_dir, run_name_or_path)

    search_pattern = os.path.join(run_dir, checkpoint_pattern)
    files = glob.glob(search_pattern)
    if not files: raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    files.sort(key=os.path.getmtime)
    return files[-1], run_dir

def save_configs(log_dir, env_cfg, train_cfg_dict):
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    with open(os.path.join(params_dir, "train_cfg.json"), "w") as f:
        json.dump(train_cfg_dict, f, indent=4, default=str)
    # 简化版 yaml 保存
    def sanitize(obj):
        if hasattr(obj, "to_dict"): return sanitize(obj.to_dict())
        if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [sanitize(v) for v in obj]
        return obj
    try:
        with open(os.path.join(params_dir, "env_cfg.yaml"), "w") as f:
            yaml.dump(sanitize(env_cfg), f, sort_keys=False)
    except: pass

def export_model_files(policy, log_dir, obs_dim, device):
    exported_dir = os.path.join(log_dir, "exported_cross_moe")
    os.makedirs(exported_dir, exist_ok=True)
    print(f"\n[Export] Exporting models to: {exported_dir}")

    # 1. Base Model
    base_model = ExportablePolicy(policy).to(device)
    base_model.eval()
    
    # 构造 Dummy Inputs
    dummy_obs = torch.zeros(1, obs_dim, device=device)
    # CrossMoE 的 Hidden State 是一个 Tuple (h_leg, h_wheel)
    # 假设 latent_dim_leg=256, latent_dim_wheel=64
    leg_dim = getattr(policy, "latent_dim_leg", 256)
    whl_dim = getattr(policy, "latent_dim_wheel", 64)
    
    # GRU 状态形状: (num_layers=1, batch=1, hidden_dim)
    dummy_h_leg = torch.zeros(1, 1, leg_dim, device=device)
    dummy_h_whl = torch.zeros(1, 1, whl_dim, device=device)
    dummy_hidden = (dummy_h_leg, dummy_h_whl)

    # 2. ONNX Export
    try:
        onnx_path = os.path.join(exported_dir, "policy.onnx")
        # 输入输出命名需要体现双 RNN 结构
        input_names = ["obs", "h_leg_in", "h_whl_in"]
        output_names = ["action", "h_leg_out", "h_whl_out"]
        
        torch.onnx.export(
            base_model, (dummy_obs, dummy_hidden), onnx_path, verbose=False,
            input_names=input_names, output_names=output_names, opset_version=13,
            dynamic_axes={
                "obs": {0: "batch"},
                "h_leg_in": {1: "batch"}, "h_whl_in": {1: "batch"},
                "action": {0: "batch"},
                "h_leg_out": {1: "batch"}, "h_whl_out": {1: "batch"}
            }
        )
        print(f"  - ONNX saved: {onnx_path}")
    except Exception as e:
        print(f"  - [Error] ONNX export failed: {e}")

    # 3. Estimator Export
    if hasattr(policy, "estimator") and policy.estimator is not None:
        try:
            est_idx = getattr(policy, "estimator_input_indices", list(range(3, 32)))
            dummy_est_obs = torch.zeros(1, len(est_idx), device=device)
            est_wrapper = ExportableEstimator(policy).to(device)
            est_path = os.path.join(exported_dir, "estimator.onnx")
            torch.onnx.export(
                est_wrapper, dummy_est_obs, est_path, verbose=False,
                input_names=["proprioception"], output_names=["estimated_state"], opset_version=13
            )
            print(f"  - Estimator ONNX saved: {est_path}")
        except Exception as e:
            print(f"  - [Error] Estimator export failed: {e}")

# ==============================================================================
#  Main
# ==============================================================================

def main():
    # 1. 环境配置
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    
    # Controllers
    controller = None
    if args.keyboard:
        print("[Info] Enabling Keyboard Control")
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        kb_cfg = Se2KeyboardCfg(
            v_x_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            v_y_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            omega_z_sensitivity=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
        )
        controller = Se2Keyboard(kb_cfg)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
        )
    elif args.joystick:
        print("[Info] Enabling Direct Joystick Control")
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        controller = DirectLinuxGamepad(
            x_scale=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            y_scale=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            w_scale=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
        )
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
        )

    env = gym.make(args.task, cfg=env_cfg)
    
    # 获取 obs_dim
    try:
        obs_sample, _ = env.reset()
        if isinstance(obs_sample, dict): obs_dim = obs_sample["policy"].shape[-1]
        else: obs_dim = obs_sample.shape[-1]
    except: obs_dim = 48

    # 2. 加载模型配置
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"): train_cfg_dict = train_cfg.to_dict()
    else: train_cfg_dict = train_cfg

    # [Mod] 强制指定 CrossMoE 类名
    print(f"\n[Info] Forcing Policy Class to: CrossMoEActorCritic")
    train_cfg_dict["policy"]["class_name"] = "CrossMoEActorCritic"
    
    if args.num_wheel_experts: train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts: train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]: train_cfg_dict["policy"].pop(k, None)

    # 3. 寻找 Checkpoint
    experiment_name = train_cfg_dict.get("experiment_name", "cross_moe_end2end")
    search_paths = [
        os.path.join("logs", "moe_training", experiment_name),
        os.path.join("logs", experiment_name)
    ]
    root_log_dir = search_paths[0]
    for p in search_paths:
        if os.path.exists(p): 
            root_log_dir = p
            break

    try:
        model_path, log_dir = resolve_checkpoint_path(root_log_dir, args.load_run, args.checkpoint)
        print(f"\n[Success] Loading model from: {model_path}")
    except Exception as e:
        print(f"\n[Error] Failed to load checkpoint: {e}")
        sys.exit(1)

    # 4. 初始化 Runner & Policy
    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    runner.load(model_path)
    
    policy = runner.get_inference_policy(device="cuda:0")
    model_instance = policy.__self__ if hasattr(policy, "__self__") else policy

    # 5. Export
    if args.export:
        save_configs(log_dir, env_cfg, train_cfg_dict)
        export_model_files(model_instance, log_dir, obs_dim, device="cuda:0")

    # 6. Visualization Hooks
    monitor_data = {}
    def hook_fn(name):
        def _hook(model, input, output):
            monitor_data[name] = output.detach()
        return _hook

    # [Mod] CrossMoE 没有 Mode Gate，只有并行的 Leg/Wheel Gate
    if hasattr(model_instance, "wheel_gate"): model_instance.wheel_gate.register_forward_hook(hook_fn("Wheel"))
    if hasattr(model_instance, "leg_gate"): model_instance.leg_gate.register_forward_hook(hook_fn("Leg"))

    # 获取 Robot 实体用于 GT
    try:
        robot_entity = env.unwrapped.scene["robot"]
    except:
        robot_entity = None

    # Visualization Function
    def print_expert_bars(probs, expert_names=None):
        lines = []
        for i, p in enumerate(probs):
            val = p.item()
            bar_len = int(val * 40)
            bar = '█' * bar_len
            if val > 0.9: color, status = "\033[92m", "DOMINANT"
            elif val > 0.1: color, status = "\033[96m", "ACTIVE"
            else: color, status = "\033[90m", " DEAD "
            name = expert_names[i] if expert_names else f"Exp {i}"
            lines.append(f"  {name:<8}: {color}{val:.3f} | {bar:<40} | {status}\033[0m")
        return lines

    def print_estimator_diff(est_vec, gt_vec):
        lines = []
        labels = ["Vx", "Vy", "Wz"]
        dim = min(len(est_vec), len(gt_vec), 3)
        for i in range(dim):
            e, g = est_vec[i].item(), gt_vec[i].item()
            diff = abs(e - g)
            err_color = "\033[92m" if diff < 0.1 else "\033[91m"
            bar = '▒' * min(int(diff * 50), 40)
            lines.append(f"  {labels[i]}: Est={e:6.3f} | GT={g:6.3f} | Err={err_color}{diff:6.3f} {bar}\033[0m")
        return lines

    last_printed_lines = 0
    def visualize(obs_idx=0, est_state=None, gt_state=None):
        nonlocal last_printed_lines
        lines = ["="*25 + " Cross-MoE Dashboard " + "="*25]
        
        # Parallel Gates Visualization
        for name in ["Wheel", "Leg"]:
            if name in monitor_data:
                logits = monitor_data[name]
                if logits.ndim == 3: logits = logits[-1] # Handle time dim if any
                probs = F.softmax(logits[obs_idx], dim=0)
                lines.append(f"[{name} Experts] (Parallel Branch):")
                lines.extend(print_expert_bars(probs))
                lines.append("-" * 30)
        
        # Estimator
        if est_state is not None and gt_state is not None:
            lines.append("State Estimator:")
            lines.extend(print_estimator_diff(est_state[obs_idx], gt_state[obs_idx]))
        
        lines.append("="*70)
        if last_printed_lines > 0:
            sys.stdout.write(f"\033[{last_printed_lines}A\033[J")
        print("\n".join(lines))
        last_printed_lines = len(lines)

    # 7. Inference Loop
    obs, _ = env.reset()
    print("\nStarting Cross-MoE Inference...")
    
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            # CrossMoE act 内部会自动处理 hidden state
            actions = policy(obs)
            
            # Estimator & GT
            est_state = None
            if hasattr(model_instance, "get_estimated_state"):
                est_state = model_instance.get_estimated_state(obs)
            
            gt_state = None
            if robot_entity is not None:
                gt_lin = robot_entity.data.root_lin_vel_b
                gt_ang = robot_entity.data.root_ang_vel_b
                gt_state = torch.cat([gt_lin[:, :2], gt_ang[:, 2:3]], dim=-1)

            obs, _, _, _ = env.step(actions)
            step += 1
            
            if step % 10 == 0:
                visualize(obs_idx=0, est_state=est_state, gt_state=gt_state)
            
            if args.keyboard or args.joystick:
                camera_follow(env)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()