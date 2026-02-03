# play_moe.py

import argparse
import sys
import os
import glob
import json
import yaml
import struct
import threading
import time

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === 1. 启动 App (必须最先执行) ===
parser = argparse.ArgumentParser(description="Play H-MoE Policy and Export")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoE-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
# H-MoE Params
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)
# Checkpoint
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default="model_*.pt")
# Keyboard
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# Export
parser.add_argument("--export", action="store_true", default=True, help="Whether to export ONNX/TorchScript and Configs.")
parser.add_argument("--joystick", action="store_true", default=False, help="Whether to use joystick/gamepad.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==============================================================================
#  依赖导入 (必须在 simulation_app 启动之后)
# ==============================================================================
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_utils import camera_follow 
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.devices import Se2Keyboard
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.utils.math as math_utils

# === 导入自定义模块 ===
try:
    sys.path.append(os.getcwd())
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO
except ImportError:
    try:
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO
    except ImportError:
        pass 

# === 注入到 RSL-RL ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

if "SplitMoEActorCritic" in globals():
    rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
    runner_module.SplitMoEActorCritic = SplitMoEActorCritic
    rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic 
    runner_module.SplitMoEPPO = SplitMoEPPO

# ==============================================================================
#  Joystick Controller (Robust Hot-Swap)
# ==============================================================================
class DirectLinuxGamepad:
    def __init__(self, device_path="/dev/input/js0", x_scale=1.0, y_scale=1.0, w_scale=1.0, deadzone=0.1):
        self.device_path = device_path
        self.axes = {}
        self.buttons = {}
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.w_scale = w_scale
        self.deadzone = deadzone
        self.running = True 
        self.connected = False 
        self.js_file = None
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        event_size = 8
        while self.running:
            try:
                # 1. 连接阶段
                if self.js_file is None:
                    if os.path.exists(self.device_path):
                        try:
                            # buffering=0 确保实时性，避免断开时卡在flush
                            self.js_file = open(self.device_path, "rb", buffering=0)
                            self.connected = True
                        except Exception:
                            self.connected = False
                            time.sleep(1)
                    else:
                        self.connected = False
                        time.sleep(1)
                        continue

                # 2. 读取阶段
                try:
                    event_data = self.js_file.read(event_size)
                except OSError:
                    event_data = None

                # 如果读取为空或长度不够，说明设备已断开
                if not event_data or len(event_data) != event_size:
                    raise OSError("Device disconnected or read error")
                
                time_ms, value, type_, number = struct.unpack("Ihbb", event_data)
                
                if type_ == 2: # Axis
                    norm_val = value / 32767.0
                    self.axes[number] = norm_val
                elif type_ == 1: # Button
                    self.buttons[number] = (value == 1)
                    
            except (OSError, IOError, struct.error):
                # 发生错误时彻底重置状态
                self.connected = False
                if self.js_file:
                    try: self.js_file.close()
                    except: pass
                self.js_file = None
                # 清空按键状态防止“卡键”
                self.axes = {} 
                self.buttons = {}
                time.sleep(1) # 等待一秒重试
            except Exception as e:
                # 捕获其他未知异常，防止线程退出
                print(f"[Joystick Error] {e}")
                time.sleep(1)

    def advance(self):
        if not self.connected: return torch.zeros(3)
        raw_x = self.axes.get(1, 0.0)
        raw_y = self.axes.get(0, 0.0)
        
        # Deadzone
        stick_vx = -raw_x if abs(raw_x) > self.deadzone else 0.0
        stick_vy = -raw_y if abs(raw_y) > self.deadzone else 0.0
        
        # DPAD override
        dpad_vx = -self.axes.get(7, 0.0) 
        dpad_vy = -self.axes.get(6, 0.0)
        if abs(dpad_vx) > 0.1 or abs(dpad_vy) > 0.1:
            vx = dpad_vx * self.x_scale
            vy = dpad_vy * self.y_scale
        else:
            vx = stick_vx * self.x_scale
            vy = stick_vy * self.y_scale
            
        raw_w = self.axes.get(3, 0.0)
        wz = -raw_w * self.w_scale if abs(raw_w) > self.deadzone else 0.0
        return torch.tensor([vx, vy, wz])

    def is_button_pressed(self, btn_index):
        return self.buttons.get(btn_index, False)

    def get_axis(self, axis_index):
        # Default -1.0 for triggers (released state)
        return self.axes.get(axis_index, -1.0) 

    def close(self):
        self.running = False
        if self.thread.is_alive(): self.thread.join(timeout=1.0)
        if self.js_file: self.js_file.close()

# ==============================================================================
#  Helpers
# ==============================================================================
def get_truly_unwrapped_env(env):
    unwrapped = env
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    if hasattr(unwrapped, "unwrapped") and unwrapped.unwrapped != unwrapped:
        unwrapped = unwrapped.unwrapped
    return unwrapped

class ExportablePolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.eval()
    def forward(self, obs, hidden_states):
        batch_size = obs.shape[0]
        masks = torch.ones(batch_size, dtype=torch.bool, device=obs.device)
        action_mean, _, next_state = self.policy.forward(
            obs, masks=masks, hidden_states=hidden_states, save_dist=False
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
    exported_dir = os.path.join(log_dir, "exported")
    os.makedirs(exported_dir, exist_ok=True)
    print(f"\n[Export] Exporting models to: {exported_dir}")
    try:
        base_model = ExportablePolicy(policy).to(device)
        base_model.eval()
        batch_size = 1
        dummy_obs = torch.zeros(batch_size, obs_dim, device=device)
        latent_dim = getattr(policy, "latent_dim", 256)
        if getattr(policy, "rnn_type", "gru") == "lstm":
            dummy_hidden = (torch.zeros(1, 1, latent_dim, device=device), torch.zeros(1, 1, latent_dim, device=device))
        else:
            dummy_hidden = torch.zeros(1, 1, latent_dim, device=device)
        
        torch.onnx.export(base_model, (dummy_obs, dummy_hidden), os.path.join(exported_dir, "policy.onnx"), 
                         input_names=["obs", "hidden_state"], output_names=["action", "next_hidden"], opset_version=13)
        print(f"  - Policy ONNX saved")
    except Exception as e:
        print(f"  - Policy Export Error: {e}")
    
    if hasattr(policy, "estimator") and policy.estimator is not None:
        try:
            if hasattr(policy.estimator, "net") and len(policy.estimator.net) > 0:
                est_input_dim = policy.estimator.net[0].in_features
            elif isinstance(policy.estimator, nn.Linear):
                est_input_dim = policy.estimator.in_features
            else:
                est_idx = getattr(policy, "estimator_input_indices", list(range(3, 32)))
                est_input_dim = len(est_idx)
            
            dummy_est_obs = torch.zeros(1, est_input_dim, device=device)
            est_wrapper = ExportableEstimator(policy).to(device)
            torch.onnx.export(est_wrapper, dummy_est_obs, os.path.join(exported_dir, "estimator.onnx"),
                input_names=["proprioception"], output_names=["estimated_state"], opset_version=13)
            print(f"  - Estimator ONNX saved (Input Dim: {est_input_dim})")
        except Exception as e:
            print(f"  - Estimator Export Error: {e}")

# ==============================================================================
#  Main
# ==============================================================================

def main():
    # 1. 环境配置
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    
    controller = None
    if args.keyboard:
        print("[Info] Enabling Keyboard Control")
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
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
        print("[Info] Enabling Direct Linux Joystick Control")
        print("[Controls] RT/LT: Difficulty +/- | RB/LB: Sub-Terrain +/- | Y: Camera | X: Reset")
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        
        controller = DirectLinuxGamepad(
            device_path="/dev/input/js0",
            x_scale=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            y_scale=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            w_scale=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
            deadzone=0.05
        )
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
        )

    # 2. 创建环境
    env_gym = gym.make(args.task, cfg=env_cfg)
    base_env = get_truly_unwrapped_env(env_gym)
    
    if not hasattr(base_env, "scene"):
        raise RuntimeError(f"Base Env {type(base_env)} does not have 'scene' attribute.")
    robot_entity = base_env.scene["robot"]
    
    # 获取地形信息
    terrain_origins = None
    num_rows = 1
    num_cols = 1
    try:
        if hasattr(base_env.scene.terrain, "terrain_origins") and base_env.scene.terrain.terrain_origins is not None:
            terrain_origins = base_env.scene.terrain.terrain_origins
            num_rows = terrain_origins.shape[0]
            num_cols = terrain_origins.shape[1]
            print(f"[Terrain] Detected Grid: {num_rows} Rows (Levels) x {num_cols} Cols (Types)")
    except Exception:
        pass
    
    try:
        obs_sample, _ = base_env.reset()
        if isinstance(obs_sample, dict): obs_dim = obs_sample["policy"].shape[-1]
        else: obs_dim = obs_sample.shape[-1]
    except: obs_dim = 48

    # 3. 加载模型
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"): train_cfg_dict = train_cfg.to_dict()
    else: train_cfg_dict = train_cfg
    train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
    if args.num_wheel_experts: train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts: train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]: train_cfg_dict["policy"].pop(k, None)

    # 4. 寻找 Checkpoint
    experiment_name = train_cfg_dict.get("experiment_name", "h_moe_end2end")
    search_paths = [
        os.path.join("logs", "moe_training", experiment_name),
        os.path.join("logs", "rsl_rl", experiment_name),
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
        print(f"\n[Error] {e}")
        sys.exit(1)

    # 5. 包装环境
    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env_wrapped = RslRlVecEnvWrapper(env_gym, clip_actions=clip_actions)
    runner = OnPolicyRunner(env_wrapped, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    runner.load(model_path)
    policy = runner.get_inference_policy(device="cuda:0")
    model_instance = policy.__self__ if hasattr(policy, "__self__") else policy

    if args.export:
        save_configs(log_dir, env_cfg, train_cfg_dict)
        export_model_files(model_instance, log_dir, obs_dim, device="cuda:0")

    # 6. Hooks
    monitor_data = {}
    def hook_fn(name):
        def _hook(model, input, output):
            monitor_data[name] = output.detach()
        return _hook
    if hasattr(model_instance, "wheel_gate"): model_instance.wheel_gate.register_forward_hook(hook_fn("Wheel"))
    if hasattr(model_instance, "leg_gate"): model_instance.leg_gate.register_forward_hook(hook_fn("Leg"))

    # === Visualization UI ===
    def draw_progress_bar(val, max_val, width=20, color_on="\033[92m", color_off="\033[90m"):
        if max_val <= 1: 
            fill = width
            disp_txt = "Fixed"
        else:
            ratio = val / (max_val - 1)
            fill = int(ratio * width)
            disp_txt = f"{val}/{max_val-1}"
        fill = max(0, min(fill, width))
        bar = "█" * fill + "░" * (width - fill)
        return f"{color_on}{bar}{color_off} {disp_txt}"

    def print_expert_bars(probs, expert_names=None):
        lines = []
        for i, p in enumerate(probs):
            val = p.item()
            bar_len = int(val * 30)
            bar = '█' * bar_len
            if val > 0.9: color, status = "\033[92m", "DOMINANT"
            elif val > 0.1: color, status = "\033[96m", "ACTIVE"
            else: color, status = "\033[90m", " DEAD "
            name = expert_names[i] if expert_names else f"Exp {i}"
            lines.append(f"  {name:<6}: {color}{val:.3f} | {bar:<30} | {status}\033[0m")
        return lines

    def print_estimator_diff(est_vec, gt_vec):
        lines = []
        labels = ["Vx", "Vy", "Wz"]
        dim = min(len(est_vec), len(gt_vec), 3)
        for i in range(dim):
            e, g = est_vec[i].item(), gt_vec[i].item()
            diff = abs(e - g)
            err_color = "\033[92m" if diff < 0.1 else "\033[91m"
            bar = '▒' * min(int(diff * 40), 30)
            lines.append(f"  {labels[i]}: Est={e:6.3f} | GT={g:6.3f} | Err={err_color}{diff:6.3f} {bar}\033[0m")
        return lines

    last_printed_lines = 0
    status_message = ""
    status_timer = 0

    def visualize(obs_idx=0, est_state=None, gt_state=None, cur_terrain_info=None, controller_debug=None, connected=True):
        nonlocal last_printed_lines, status_message, status_timer
        
        lines = []
        lines.append("="*30 + " H-MoE Dashboard " + "="*30)
        
        # Controller Status
        if args.joystick:
            conn_status = "\033[92mCONNECTED\033[0m" if connected else "\033[91mDISCONNECTED\033[0m"
            lines.append(f"Controller: {conn_status}")
            if controller_debug and connected:
                lines.append(f"Inputs (Raw): RT={controller_debug['rt']:.2f}, LT={controller_debug['lt']:.2f}, RB={int(controller_debug['rb'])}, LB={int(controller_debug['lb'])}")
        
        if cur_terrain_info:
            cur_lvl, cur_type = cur_terrain_info
            lines.append(f"Terrain Status:")
            lines.append(f"  Level : {draw_progress_bar(cur_lvl, num_rows)}")
            lines.append(f"  Type  : {draw_progress_bar(cur_type, num_cols)}")
            lines.append("-" * 65)

        for name in ["Wheel", "Leg"]:
            if name in monitor_data:
                logits = monitor_data[name]
                if logits.ndim == 3: logits = logits[-1]
                probs = F.softmax(logits[obs_idx], dim=0)
                lines.append(f"[{name} Experts]:")
                lines.extend(print_expert_bars(probs))
                lines.append("-" * 30)
        
        if est_state is not None and gt_state is not None:
            lines.append("State Estimator:")
            lines.extend(print_estimator_diff(est_state[obs_idx], gt_state[obs_idx]))
            
        lines.append("="*75)

        if status_timer > 0:
            lines.append(f"\033[93m[EVENT] {status_message}\033[0m")
            status_timer -= 1
        else:
            lines.append("") 

        if last_printed_lines > 0:
            sys.stdout.write(f"\033[{last_printed_lines}A\033[J")
        
        print("\n".join(lines))
        last_printed_lines = len(lines)

    # 7. Inference Loop
    obs, _ = env_wrapped.reset()
    print("\nStarting H-MoE Inference...")
    
    # State
    camera_mode = 0  
    cur_difficulty = 0 
    cur_subterrain = 0 
    
    # Button latches
    y_prev, x_prev = False, False
    rt_prev, lt_prev = False, False
    rb_prev, lb_prev = False, False

    OFFSET_FORWARD = [ -2.5, 0.0, 1.5 ]
    HEIGHT_TOP = 5.0
    OFFSET_BACKWARD = [ 2.5, 0.0, 1.5 ]
    camera_history = []
    
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            
            reset_terrain_needed = False
            ctrl_debug = {}
            is_connected = False

            # === Joystick Logic ===
            if args.joystick and controller is not None:
                is_connected = controller.connected
                
                # Read Inputs
                y_curr = controller.is_button_pressed(3)
                x_curr = controller.is_button_pressed(2)
                rb_curr = controller.is_button_pressed(5)
                lb_curr = controller.is_button_pressed(4)
                
                rt_val = controller.get_axis(5)
                lt_val = controller.get_axis(2)
                
                ctrl_debug = {'rt': rt_val, 'lt': lt_val, 'rb': rb_curr, 'lb': lb_curr}
                
                # [Fix] Threshold changed to -0.5 to catch triggers properly
                rt_curr = rt_val > -0.5
                lt_curr = lt_val > -0.5

                # Camera (Y)
                if y_curr and not y_prev:
                    camera_mode = (camera_mode + 1) % 3
                    mode_names = ["Forward", "Top-Down", "Front-Face"]
                    status_message = f"Camera: {mode_names[camera_mode]}"
                    status_timer = 30
                    camera_history.clear()
                
                # Reset (X)
                if x_curr and not x_prev:
                    status_message = "Resetting..."
                    status_timer = 30
                    reset_terrain_needed = True

                # Difficulty (RT/LT)
                if num_rows > 1:
                    if rt_curr and not rt_prev:
                        if cur_difficulty < num_rows - 1:
                            cur_difficulty += 1
                            status_message = f"Difficulty INCREASED -> {cur_difficulty}"
                            status_timer = 30
                            reset_terrain_needed = True
                    if lt_curr and not lt_prev:
                        if cur_difficulty > 0:
                            cur_difficulty -= 1
                            status_message = f"Difficulty DECREASED -> {cur_difficulty}"
                            status_timer = 30
                            reset_terrain_needed = True
                
                # Sub-Terrain (RB/LB)
                if num_cols > 1:
                    if rb_curr and not rb_prev:
                        cur_subterrain = (cur_subterrain + 1) % num_cols
                        status_message = f"Sub-Terrain NEXT -> {cur_subterrain}"
                        status_timer = 30
                        reset_terrain_needed = True
                    if lb_curr and not lb_prev:
                        cur_subterrain = (cur_subterrain - 1) % num_cols
                        status_message = f"Sub-Terrain PREV -> {cur_subterrain}"
                        status_timer = 30
                        reset_terrain_needed = True
                
                # Update Latches
                y_prev, x_prev = y_curr, x_curr
                rt_prev, lt_prev = rt_curr, lt_curr
                rb_prev, lb_prev = rb_curr, lb_curr

                # Execute Terrain Reset
                if reset_terrain_needed:
                    # 1. Force internal terrain level (for curriculum logic)
                    if hasattr(base_env.scene.terrain, "terrain_levels"):
                        levels_vec = torch.full((env_wrapped.num_envs,), cur_difficulty, device=env_wrapped.device, dtype=torch.long)
                        base_env.scene.terrain.terrain_levels[:] = levels_vec
                    
                    # 2. Standard Reset
                    obs, _ = env_wrapped.reset()

                    # 3. Teleport [CRITICAL FIX]
                    if terrain_origins is not None and robot_entity is not None:
                        r_idx = max(0, min(cur_difficulty, num_rows - 1))
                        c_idx = max(0, min(cur_subterrain, num_cols - 1))
                        target_origin = terrain_origins[r_idx, c_idx].clone()
                        target_origin[2] += 0.55 
                        
                        # Fix: Concatenate position and quaternion into one tensor (N, 7)
                        default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env_wrapped.device).repeat(env_wrapped.num_envs, 1)
                        target_pos = target_origin.unsqueeze(0).repeat(env_wrapped.num_envs, 1).to(env_wrapped.device)
                        
                        root_pose = torch.cat([target_pos, default_quat], dim=-1)
                        robot_entity.write_root_pose_to_sim(root_pose)
                        
                        # [Fix] Zero out velocities instead of calling reset_buffers()
                        root_vel = torch.zeros_like(robot_entity.data.root_link_vel_w)
                        robot_entity.write_root_velocity_to_sim(root_vel)

            # Estimator & GT
            est_state = None
            if hasattr(model_instance, "get_estimated_state"):
                est_state = model_instance.get_estimated_state(obs)
            
            gt_state = None
            if robot_entity is not None:
                gt_lin = robot_entity.data.root_lin_vel_b
                gt_ang = robot_entity.data.root_ang_vel_b
                gt_state = torch.cat([gt_lin[:, :2], gt_ang[:, 2:3]], dim=-1)

            obs, _, _, _ = env_wrapped.step(actions)
            step += 1
            
            if step % 2 == 0: 
                visualize(
                    obs_idx=0, 
                    est_state=est_state, 
                    gt_state=gt_state, 
                    cur_terrain_info=(cur_difficulty, cur_subterrain),
                    controller_debug=ctrl_debug if args.joystick else None,
                    connected=is_connected
                )
            
            # Camera Control
            if robot_entity is not None and (args.joystick or args.keyboard):
                root_pos = robot_entity.data.root_pos_w[0]
                root_quat = robot_entity.data.root_quat_w[0]
                eye, target = None, root_pos

                if camera_mode == 0: # Behind
                    offset_local = torch.tensor(OFFSET_FORWARD, device=root_pos.device)
                    offset_world = math_utils.quat_apply(root_quat, offset_local)
                    eye = root_pos + offset_world
                elif camera_mode == 1: # Top
                    eye = root_pos + torch.tensor([0.0, 0.0, HEIGHT_TOP], device=root_pos.device)
                    target = root_pos + torch.tensor([0.001, 0.0, 0.0], device=root_pos.device)
                elif camera_mode == 2: # Front
                    offset_local = torch.tensor(OFFSET_BACKWARD, device=root_pos.device)
                    offset_world = math_utils.quat_apply(root_quat, offset_local)
                    eye = root_pos + offset_world

                if eye is not None:
                    camera_history.append(eye)
                    if len(camera_history) > 50: camera_history.pop(0)
                    smooth_eye = torch.stack(camera_history).mean(dim=0)
                    base_env.sim.set_camera_view(smooth_eye.cpu().numpy(), target.cpu().numpy())
            else:
                if args.keyboard: camera_follow(env_wrapped)

    env_wrapped.close()
    if args.joystick and controller is not None:
        controller.close()

if __name__ == "__main__":
    main()
    simulation_app.close()