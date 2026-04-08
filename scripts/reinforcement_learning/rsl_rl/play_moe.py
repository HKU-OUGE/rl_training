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
parser.add_argument("--logbag", type=str, default="", help="Path to save offline test logbag (e.g. logbag.jsonl)")
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

import carb
import carb.input
import omni.appwindow

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
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO, SplitMoEStudentTeacher
except ImportError:
    try:
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO, SplitMoEStudentTeacher
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
    
    if "SplitMoEStudentTeacher" in globals():
        import rsl_rl.runners.distillation_runner as dist_runner_module
        dist_runner_module.SplitMoEStudentTeacher = SplitMoEStudentTeacher

# ==============================================================================
#  Keyboard Controller Extension (For Camera, Reset, and Terrain)
# ==============================================================================
class KeyboardExtension:
    """监听 Omniverse 底层键盘事件，用于补充控制视野和地形。"""
    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        self.key_just_pressed = {}

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.key_just_pressed[event.input] = True
        return True # 返回 True 允许其他模块（如 Se2Keyboard）继续处理该事件

    def check_and_clear(self, key):
        if self.key_just_pressed.get(key, False):
            self.key_just_pressed[key] = False
            return True
        return False

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
                if self.js_file is None:
                    if os.path.exists(self.device_path):
                        try:
                            self.js_file = open(self.device_path, "rb", buffering=0)
                            self.connected = True
                        except Exception:
                            self.connected = False
                            time.sleep(1)
                    else:
                        self.connected = False
                        time.sleep(1)
                        continue

                try:
                    event_data = self.js_file.read(event_size)
                except OSError:
                    event_data = None

                if not event_data or len(event_data) != event_size:
                    raise OSError("Device disconnected or read error")
                
                time_ms, value, type_, number = struct.unpack("Ihbb", event_data)
                
                if type_ == 2: # Axis
                    norm_val = value / 32767.0
                    self.axes[number] = norm_val
                elif type_ == 1: # Button
                    self.buttons[number] = (value == 1)
                    
            except (OSError, IOError, struct.error):
                self.connected = False
                if self.js_file:
                    try: self.js_file.close()
                    except: pass
                self.js_file = None
                self.axes = {} 
                self.buttons = {}
                time.sleep(1)
            except Exception as e:
                print(f"[Joystick Error] {e}")
                time.sleep(1)

    def advance(self):
        if not self.connected: return torch.zeros(3)
        raw_x = self.axes.get(1, 0.0)
        raw_y = self.axes.get(0, 0.0)
        
        stick_vx = -raw_x if abs(raw_x) > self.deadzone else 0.0
        stick_vy = -raw_y if abs(raw_y) > self.deadzone else 0.0
        
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

# ==============================================================================
#  Unified Export Helpers
# ==============================================================================

def get_flat_obs_dim(policy):
    """根据网络配置，自动推断部署端需要传入的总 Tensor 维度"""
    dim = policy.proprio_dim
    if not getattr(policy, "blind_vision", False):
        if getattr(policy, "use_elevation_ae", False):
            dim += policy.elevation_dim
        if getattr(policy, "use_multilayer_scan", False):
            dim += policy.scan_dim
        if getattr(policy, "use_cnn", False):
            dim += policy.image_raw_dim
    return dim

class UnifiedExportPolicy(nn.Module):
    """统一导出包装器"""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad = False
        self.rnn_type = getattr(policy, "rnn_type", "gru").lower()

    def forward(self, proprio_and_env, estimator_history, h0, c0=None):
        if self.rnn_type == "lstm":
            hidden_states = (h0, c0)
        else:
            hidden_states = h0
            
        obs_dict = {
            "policy": proprio_and_env[..., :self.policy.proprio_dim],
            "noisy_elevation": proprio_and_env[..., self.policy.proprio_dim:]
        }
        if estimator_history.shape[-1] > 0:
            obs_dict["estimator"] = estimator_history
            
        action_mean, _, next_state = self.policy.forward(
            obs_dict, masks=None, hidden_states=hidden_states, save_dist=False
        )
        
        if self.rnn_type == "lstm":
            return action_mean, next_state[0], next_state[1]
        else:
            return action_mean, next_state

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

def export_model_files(policy, log_dir, device):
    exported_dir = os.path.join(log_dir, "exported")
    os.makedirs(exported_dir, exist_ok=True)
    print(f"\n[Export] Exporting UNIFIED End-to-End models to: {exported_dir}")
    
    try:
        base_model = UnifiedExportPolicy(policy).to(device)
        
        flat_obs_dim = get_flat_obs_dim(policy)
        batch_size = 1
        latent_dim = getattr(policy, "latent_dim", 256)
        rnn_type = getattr(policy, "rnn_type", "gru").lower()
        
        est_dim = 0
        if getattr(policy, "has_estimator_group", False) and policy.estimator_obs_normalizer is not None:
            est_dim = policy.estimator_obs_normalizer.mean.shape[-1]
        
        dummy_obs = torch.zeros(batch_size, flat_obs_dim, device=device)
        dummy_est = torch.zeros(batch_size, est_dim, device=device)
        
        if rnn_type == "lstm":
            dummy_h0 = torch.zeros(1, batch_size, latent_dim, device=device)
            dummy_c0 = torch.zeros(1, batch_size, latent_dim, device=device)
            inputs = (dummy_obs, dummy_est, dummy_h0, dummy_c0)
            input_names = ["proprio_and_env", "estimator_history", "h0", "c0"]
            output_names = ["action", "next_h0", "next_c0"]
        else:
            dummy_h0 = torch.zeros(1, batch_size, latent_dim, device=device)
            inputs = (dummy_obs, dummy_est, dummy_h0)
            input_names = ["proprio_and_env", "estimator_history", "h0"]
            output_names = ["action", "next_h0"]
            
        onnx_path = os.path.join(exported_dir, "unified_policy.onnx")
        torch.onnx.export(
            base_model, inputs, onnx_path, 
            input_names=input_names, output_names=output_names, opset_version=14 
        )
        print(f"  - [Success] Unified ONNX saved: {onnx_path}")

        ts_path = os.path.join(exported_dir, "unified_policy.pt")
        try:
            traced_model = torch.jit.trace(base_model, inputs)
            traced_model.save(ts_path)
            print(f"  - [Success] Unified TorchScript saved: {ts_path}")
        except Exception as e_trace:
            print(f"  [Warning] TorchScript Trace failed: {e_trace}")

    except Exception as e:
        print(f"  [Error] Unified Policy Export Failed: {e}")

def export_sim2real_layout(env_cfg, policy, log_dir):
    """
    解析 env_cfg，提取所有的观测顺序、Scale、Action 缩放等，并生成部署文件
    """
    exported_dir = os.path.join(log_dir, "exported")
    os.makedirs(exported_dir, exist_ok=True)
    
    txt_path = os.path.join(exported_dir, "sim2real_layout.txt")
    json_path = os.path.join(exported_dir, "sim2real_layout.json")
    
    layout_dict = {
        "observations": {},
        "actions": {},
        "policy_info": {}
    }

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*65 + "\n")
        f.write("             SIM2REAL I/O LAYOUT REPORT             \n")
        f.write("="*65 + "\n\n")

        # ================= 1. 解析观测 (Observations) =================
        f.write("1. OBSERVATIONS (拼接顺序自上而下)\n")
        f.write("-" * 40 + "\n")
        for group_name, group_cfg in env_cfg.observations.__dict__.items():
            if group_name.startswith("__") or group_cfg is None:
                continue
            
            f.write(f"\n[ Group: {group_name} ]\n")
            history_len = getattr(group_cfg, "history_length", 0)
            flatten = getattr(group_cfg, "flatten_history_dim", False)
            f.write(f"  - History Length : {history_len}\n")
            f.write(f"  - Flatten History: {flatten}\n")
            f.write(f"  - Terms Order:\n")
            
            layout_dict["observations"][group_name] = {
                "history_length": history_len,
                "flatten": flatten,
                "terms": []
            }

            # 提取每个 Term
            for term_name, term_cfg in group_cfg.__dict__.items():
                if term_name.startswith("__") or term_cfg is None:
                    continue
                # 判断是否是 ObservationTerm (只要包含 func 属性基本就是)
                if hasattr(term_cfg, "func"):
                    scale = getattr(term_cfg, "scale", 1.0)
                    clip = getattr(term_cfg, "clip", None)
                    # 【修复】：加了 str() 防止 None 触发排版异常
                    f.write(f"      -> {term_name:<22} | scale: {str(scale):<6} | clip: {str(clip)}\n")
                    
                    layout_dict["observations"][group_name]["terms"].append({
                        "name": term_name,
                        "scale": scale,
                        "clip": clip
                    })

        # ================= 2. 解析动作 (Actions) =================
        f.write("\n\n2. ACTIONS (部署端应除以 Scale 并减去 Offset)\n")
        f.write("-" * 40 + "\n")
        for action_name, action_cfg in env_cfg.actions.__dict__.items():
            if action_name.startswith("__") or action_cfg is None:
                continue
            if hasattr(action_cfg, "class_type"):
                scale = getattr(action_cfg, "scale", 1.0)
                offset = getattr(action_cfg, "offset", 0.0)
                f.write(f"\n[ Action Group: {action_name} ]\n")
                
                layout_dict["actions"][action_name] = {
                    "offset": offset,
                    "scales": {}
                }
                
                if isinstance(scale, dict):
                    for k, v in scale.items():
                        f.write(f"  - Regex '{k}': scale = {v}\n")
                        layout_dict["actions"][action_name]["scales"][k] = v
                else:
                    f.write(f"  - All joints scale = {scale}\n")
                    layout_dict["actions"][action_name]["scales"]["all"] = scale
                    
                f.write(f"  - Offset = {offset}\n")

        # ================= 3. 解析网络维度 (Policy Info) =================
        f.write("\n\n3. POLICY NETWORK INFO\n")
        f.write("-" * 40 + "\n")
        proprio_dim = getattr(policy, 'proprio_dim', 'Unknown')
        estimator_dim = getattr(policy, 'estimator_dim', 'Unknown')
        rnn_type = getattr(policy, 'rnn_type', 'Unknown')
        
        f.write(f"  - Proprio Dim   : {proprio_dim}\n")
        f.write(f"  - Estimator Dim : {estimator_dim}\n")
        f.write(f"  - RNN Type      : {rnn_type}\n")
        
        layout_dict["policy_info"] = {
            "proprio_dim": proprio_dim,
            "estimator_dim": estimator_dim,
            "rnn_type": rnn_type
        }

    # 保存供 C++ 自动读取的 JSON
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(layout_dict, jf, indent=4)
        
    print(f"  - [Success] Sim2Real Layout Report saved to: {txt_path}")
# ==============================================================================
#  Main
# ==============================================================================

def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    
    controller = None
    kb_ext = None
    
    if args.keyboard:
        print("[Info] Enabling Keyboard Control")
        print("[Controls] W/A/S/D: Move | Q/E: Rotate")
        print("[Controls] T/G: Difficulty +/- | H/F: Sub-Terrain +/- | C: Camera | R: Reset")
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        kb_cfg = Se2KeyboardCfg(
            v_x_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            v_y_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            omega_z_sensitivity=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
        )
        controller = Se2Keyboard(kb_cfg)
        kb_ext = KeyboardExtension()
        
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

    if controller is not None:
        def custom_velocity_commands(env):
            # controller.advance() works for both Se2Keyboard and DirectLinuxGamepad!
            cmds = controller.advance().to(env.device, dtype=torch.float32)
            return cmds.unsqueeze(0).repeat(env.num_envs, 1)

        for attr_name in dir(env_cfg.observations):
            if attr_name.startswith("__"): continue
            group = getattr(env_cfg.observations, attr_name)
            
            if hasattr(group, "velocity_commands"):
                term = getattr(group, "velocity_commands")
                if term is not None:
                    term.func = custom_velocity_commands
                    term.params = {} 
                    print(f"[Info] In-place patched velocity_commands for '{attr_name}' group.")

    # 2. 创建环境
    env_gym = gym.make(args.task, cfg=env_cfg)
    base_env = get_truly_unwrapped_env(env_gym)
    
    if not hasattr(base_env, "scene"):
        raise RuntimeError(f"Base Env {type(base_env)} does not have 'scene' attribute.")
    robot_entity = base_env.scene["robot"]
    
    terrain_origins = None
    num_rows = 1
    num_cols = 1
    try:
        if hasattr(base_env.scene.terrain, "terrain_origins") and base_env.scene.terrain.terrain_origins is not None:
            terrain_origins = base_env.scene.terrain.terrain_origins
            num_rows = terrain_origins.shape[0]
            num_cols = terrain_origins.shape[1]
    except Exception:
        pass

    # 3. 加载模型配置
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"): train_cfg_dict = train_cfg.to_dict()
    else: train_cfg_dict = train_cfg
    
    train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
    
    if args.num_wheel_experts: train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts: train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]: train_cfg_dict["policy"].pop(k, None)

    # 强行兼容推理配置
    algo_class_name = train_cfg_dict["algorithm"].get("class_name", "")
    if "Distillation" in algo_class_name:
        train_cfg_dict["algorithm"]["class_name"] = "PPO"
        for k in ["gradient_length", "loss_type", "optimizer"]:
            train_cfg_dict["algorithm"].pop(k, None)
        train_cfg_dict["algorithm"].setdefault("value_loss_coef", 1.0)
        train_cfg_dict["algorithm"].setdefault("use_clipped_value_loss", True)
        train_cfg_dict["algorithm"].setdefault("clip_param", 0.2)
        train_cfg_dict["algorithm"].setdefault("entropy_coef", 0.01)

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

    # 5. 包装环境并加载权重
    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env_wrapped = RslRlVecEnvWrapper(env_gym, clip_actions=clip_actions)
    
    runner = OnPolicyRunner(env_wrapped, train_cfg_dict, log_dir=log_dir, device=device)
    
    loaded_dict = torch.load(model_path, map_location=device)
    state_dict = loaded_dict["model_state_dict"]
    
    is_distilled = any(k.startswith("student.") for k in state_dict.keys())
    
    if is_distilled:
        print("[Info] Detected Distillation Checkpoint. Loading 'student' subnet...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("student."):
                new_key = k.replace("student.", "", 1)
                if "critic" in new_key:
                    continue
                new_state_dict[new_key] = v
        
        missing, unexpected = torch.nn.Module.load_state_dict(runner.alg.policy, new_state_dict, strict=False)
    else:
        print("[Info] Detected Standard PPO Checkpoint.")
        runner.load(model_path)

    policy = runner.get_inference_policy(device=device)
    model_instance = policy.__self__ if hasattr(policy, "__self__") else policy

    if args.export:
        save_configs(log_dir, env_cfg, train_cfg_dict)
        export_model_files(model_instance, log_dir, device=device)
        export_sim2real_layout(env_cfg, model_instance, log_dir)
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
    
    def print_tracking_diff(cmd_vec, gt_vec):
        lines = []
        labels = ["Vx", "Vy", "Wz"]
        dim = min(len(cmd_vec), len(gt_vec), 3)
        for i in range(dim):
            c, g = cmd_vec[i].item(), gt_vec[i].item()
            diff = abs(c - g)
            # 误差越小越绿，越大越红
            err_color = "\033[92m" if diff < 0.2 else "\033[91m"
            bar = '▒' * min(int(diff * 20), 30)
            lines.append(f"  {labels[i]}: Cmd={c:6.3f} | Act={g:6.3f} | Err={err_color}{diff:6.3f} {bar}\033[0m")
        return lines
    
    last_printed_lines = 0
    status_message = ""
    status_timer = 0

    def visualize(obs_idx=0, est_state=None, gt_state=None, cmd_state=None, cur_terrain_info=None, controller_debug=None, connected=True):
        nonlocal last_printed_lines, status_message, status_timer
        
        lines = []
        # 1. 缩短首行分隔符长度
        lines.append("="*18 + " H-MoE Dashboard " + "="*18)
        
        if args.joystick or args.keyboard:
            ctrl_type = "Gamepad" if args.joystick else "Keyboard"
            conn_status = "\033[92mCONNECTED\033[0m" if connected else "\033[91mDISCONNECTED\033[0m"
            lines.append(f"Controller: [{ctrl_type}] {conn_status}")
            if controller_debug and connected and args.joystick:
                lines.append(f"Inputs (Raw): RT={controller_debug.get('rt',0):.2f}, LT={controller_debug.get('lt',0):.2f}, RB={int(controller_debug.get('rb',0))}, LB={int(controller_debug.get('lb',0))}")
        
        if cur_terrain_info:
            cur_lvl, cur_type = cur_terrain_info
            lines.append(f"Terrain Status:")
            lines.append(f"  Level : {draw_progress_bar(cur_lvl, num_rows)}")
            lines.append(f"  Type  : {draw_progress_bar(cur_type, num_cols)}")
            tracking_weight = max(0.2, 1.0 - (cur_lvl / 30.0))
            lines.append(f"  Tolerance: \033[93m{tracking_weight:.2f}\033[0m (Reward Weight)")
            lines.append("-" * 53)

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

        if cmd_state is not None and gt_state is not None:
            lines.append("-" * 30)
            lines.append("Velocity Tracking (Cmd vs Actual):")
            lines.extend(print_tracking_diff(cmd_state[obs_idx], gt_state[obs_idx]))

        # 缩短底部分隔符
        lines.append("="*53)

        if status_timer > 0:
            lines.append(f"\033[93m[EVENT] {status_message}\033[0m")
            status_timer -= 1
        else:
            lines.append("") 

        if last_printed_lines > 0:
            # 2. 加入 \r 确保光标严格回到最左侧行首，并清空缓冲区
            sys.stdout.write(f"\r\033[{last_printed_lines}A\033[J")
            sys.stdout.flush()
        
        print("\n".join(lines))
        last_printed_lines = len(lines)

    # 7. Inference Loop
    obs, _ = env_wrapped.reset()
    print("\nStarting H-MoE Inference...")
    
    camera_mode = 0  
    cur_difficulty = 0 
    cur_subterrain = 0 
    
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
            obs_dict = base_env.obs_buf
            log_dict = None
            if args.logbag and step > 0: # step 0 的 last_action 是空的，跳过
                robot = base_env.scene["robot"]
                # 1. 基础本体 (Raw)
                omega = robot.data.root_ang_vel_b[0].cpu().tolist()
                proj_g = robot.data.projected_gravity_b[0].cpu().tolist()
                cmd = base_env.command_manager.get_command("base_velocity")[0].cpu().tolist()
                jp = robot.data.joint_pos[0].cpu().tolist()
                jv = robot.data.joint_vel[0].cpu().tolist()
                last_act = prev_actions[0].cpu().tolist()

                # 2. 高程图 (Python已经处理好了187维，直接取这187个值，规避C++重构ROS GridMap的麻烦)
                noisy_ele = obs_dict["noisy_elevation"][0].cpu().tolist()
                processed_heights = noisy_ele[:187]

                # 3. 雷达扫描距离 (完全 Raw，包含 NaN/Inf)
                raw_fwd = []
                raw_bwd = []
                for i in range(6):
                    # 前向
                    f_sens = base_env.scene.sensors[f"forward_scanner_layer{i}"]
                    f_dist = torch.norm(f_sens.data.ray_hits_w[0] - f_sens.data.pos_w[0], dim=-1)
                    f_dist = torch.nan_to_num(f_dist, posinf=5.0, neginf=5.0, nan=5.0).cpu().tolist()
                    raw_fwd.extend(f_dist)
                    # 后向
                    b_sens = base_env.scene.sensors[f"backward_scanner_layer{i}"]
                    b_dist = torch.norm(b_sens.data.ray_hits_w[0] - b_sens.data.pos_w[0], dim=-1)
                    b_dist = torch.nan_to_num(b_dist, posinf=5.0, neginf=5.0, nan=5.0).cpu().tolist()
                    raw_bwd.extend(b_dist)

                log_dict = {
                    "omega": omega, "proj_g": proj_g, "cmd": cmd,
                    "jp": jp, "jv": jv, "last_action": last_act,
                    "processed_heights": processed_heights,
                    "raw_fwd": raw_fwd, "raw_bwd": raw_bwd
                }


            actions = policy(obs_dict)
            

            if log_dict is not None:
                log_dict["gt_action"] = actions[0].cpu().tolist()
                with open(args.logbag, "a") as f:
                    f.write(json.dumps(log_dict) + "\n")
            prev_actions = actions.clone()

            actions = policy(obs_dict)
            
            reset_terrain_needed = False
            ctrl_debug = {}
            is_connected = False

            y_curr, x_curr, rt_curr, lt_curr, rb_curr, lb_curr = False, False, False, False, False, False

            # --- 获取输入逻辑：手柄 vs 键盘 ---
            if args.joystick and controller is not None:
                is_connected = controller.connected
                y_curr = controller.is_button_pressed(3)  # Y
                x_curr = controller.is_button_pressed(2)  # X
                rb_curr = controller.is_button_pressed(5) # RB
                lb_curr = controller.is_button_pressed(4) # LB
                rt_val = controller.get_axis(5)
                lt_val = controller.get_axis(2)
                ctrl_debug = {'rt': rt_val, 'lt': lt_val, 'rb': rb_curr, 'lb': lb_curr}
                rt_curr = rt_val > -0.5
                lt_curr = lt_val > -0.5
                
            elif args.keyboard and kb_ext is not None:
                is_connected = True
                # 利用 KeyboardExtension 读取状态 (按一次只触发一帧)
                y_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.C)
                x_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.R)
                rt_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.T)
                lt_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.G)
                rb_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.H)
                lb_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.F)

            # --- 执行指令处理逻辑 ---
            if y_curr and not y_prev:
                camera_mode = (camera_mode + 1) % 3
                mode_names = ["Forward", "Top-Down", "Front-Face"]
                status_message = f"Camera: {mode_names[camera_mode]}"
                status_timer = 30
                camera_history.clear()
            
            if x_curr and not x_prev:
                status_message = "Resetting..."
                status_timer = 30
                reset_terrain_needed = True

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
            
            # 手柄依然需要 prev 状态记录，键盘由于是 check_and_clear 机制，这里赋值也不影响
            y_prev, x_prev = y_curr, x_curr
            rt_prev, lt_prev = rt_curr, lt_curr
            rb_prev, lb_prev = rb_curr, lb_curr

            if reset_terrain_needed:
                if hasattr(base_env.scene.terrain, "terrain_levels"):
                    levels_vec = torch.full((env_wrapped.num_envs,), cur_difficulty, device=env_wrapped.device, dtype=torch.long)
                    base_env.scene.terrain.terrain_levels[:] = levels_vec
                
                obs, _ = env_wrapped.reset()

                if terrain_origins is not None and robot_entity is not None:
                    r_idx = max(0, min(cur_difficulty, num_rows - 1))
                    c_idx = max(0, min(cur_subterrain, num_cols - 1))
                    target_origin = terrain_origins[r_idx, c_idx].clone()
                    target_origin[2] += 0.55 
                    
                    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env_wrapped.device).repeat(env_wrapped.num_envs, 1)
                    target_pos = target_origin.unsqueeze(0).repeat(env_wrapped.num_envs, 1).to(env_wrapped.device)
                    
                    root_pose = torch.cat([target_pos, default_quat], dim=-1)
                    robot_entity.write_root_pose_to_sim(root_pose)
                    
                    root_vel = torch.zeros_like(robot_entity.data.root_link_vel_w)
                    robot_entity.write_root_velocity_to_sim(root_vel)

            est_state = None
            if hasattr(model_instance, "get_estimated_state"):
                est_state = model_instance.get_estimated_state(obs_dict)
            
            gt_state = None
            if robot_entity is not None:
                gt_lin = robot_entity.data.root_lin_vel_b
                gt_ang = robot_entity.data.root_ang_vel_b
                gt_state = torch.cat([gt_lin[:, :2], gt_ang[:, 2:3]], dim=-1)
            actual_cmd = obs_dict["policy"][:, 6:9]
            obs, _, _, _ = env_wrapped.step(actions)
            step += 1
            
            if step % 2 == 0: 
                visualize(
                    obs_idx=0, 
                    est_state=est_state, 
                    gt_state=gt_state,
                    cmd_state=actual_cmd, 
                    cur_terrain_info=(cur_difficulty, cur_subterrain),
                    controller_debug=ctrl_debug if args.joystick else None,
                    connected=is_connected
                )
            # 统一 Camera Follow 逻辑（适用于手柄和键盘）
            if robot_entity is not None and (args.joystick or args.keyboard):
                root_pos = robot_entity.data.root_pos_w[0]
                root_quat = robot_entity.data.root_quat_w[0]
                eye, target = None, root_pos

                if camera_mode == 0: 
                    offset_local = torch.tensor(OFFSET_FORWARD, device=root_pos.device)
                    offset_world = math_utils.quat_apply(root_quat, offset_local)
                    eye = root_pos + offset_world
                elif camera_mode == 1: 
                    eye = root_pos + torch.tensor([0.0, 0.0, HEIGHT_TOP], device=root_pos.device)
                    target = root_pos + torch.tensor([0.001, 0.0, 0.0], device=root_pos.device)
                elif camera_mode == 2: 
                    offset_local = torch.tensor(OFFSET_BACKWARD, device=root_pos.device)
                    offset_world = math_utils.quat_apply(root_quat, offset_local)
                    eye = root_pos + offset_world

                if eye is not None:
                    camera_history.append(eye)
                    if len(camera_history) > 50: camera_history.pop(0)
                    smooth_eye = torch.stack(camera_history).mean(dim=0)
                    base_env.sim.set_camera_view(smooth_eye.cpu().numpy(), target.cpu().numpy())

    env_wrapped.close()
    if args.joystick and controller is not None:
        controller.close()

if __name__ == "__main__":
    main()
    simulation_app.close()