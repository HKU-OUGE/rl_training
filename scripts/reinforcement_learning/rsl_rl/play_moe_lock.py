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
parser.add_argument("--vis-scan-obs", action="store_true", default=False,
                    help="Visualize post-augmentation LiDAR scan obs as 3D markers (env 0). "
                         "绿球=valid, 红球=blind/dropout. 与 Isaac Sim Scene Debug 里的 forward/backward_lidar 原始点对照看.")
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
try:
    import omni.appwindow  # not available in --headless Kit configs
    _HAS_APPWINDOW = True
except ModuleNotFoundError:
    _HAS_APPWINDOW = False
    omni = None  # placeholder; KeyboardExtension instantiation will be skipped

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
    """统一导出包装器.

    若 policy.use_scan_history=True, ONNX 接口扩展:
      新增输入: scan_history (B, K, scan_out_dim) — 过去 K 帧 ae latent (oldest→newest)
      新增输出: scan_lat_t (B, scan_out_dim) — 当前帧 ae latent (caller 加进 buffer)
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad = False
        self.rnn_type = getattr(policy, "rnn_type", "gru").lower()
        self.use_scan_history = getattr(policy, "use_scan_history", False)
        self.scan_history_len = getattr(policy, "scan_history_len", 0)
        self.scan_out_dim = getattr(policy, "scan_out_dim", 64)

    def forward(self, proprio_and_env, estimator_history, h0, c0_or_scan_history=None, scan_history=None):
        # 兼容三种调用 :
        #   GRU 无 scan_history:  forward(po, eh, h0)
        #   GRU + scan_history:   forward(po, eh, h0, scan_history)  -> c0_or_scan_history 即 scan_history
        #   LSTM:                 forward(po, eh, h0, c0)            -> 无 scan_history
        #   LSTM + scan_history:  forward(po, eh, h0, c0, scan_history)
        if self.rnn_type == "lstm":
            hidden_states = (h0, c0_or_scan_history)
            scan_hist_input = scan_history
        else:
            hidden_states = h0
            scan_hist_input = c0_or_scan_history if c0_or_scan_history is not None else scan_history

        obs_dict = {
            "policy": proprio_and_env[..., :self.policy.proprio_dim],
            "noisy_elevation": proprio_and_env[..., self.policy.proprio_dim:]
        }
        if estimator_history.shape[-1] > 0:
            obs_dict["estimator"] = estimator_history

        # 若使用 scan_history, 通过 forward 的 _scan_history_input kwarg 传入. policy.forward 不直接接受
        # 此 kwarg, 借助实例属性临时透传。
        if self.use_scan_history:
            self.policy._onnx_scan_history_input = scan_hist_input
        try:
            action_mean, _, next_state = self.policy.forward(
                obs_dict, masks=None, hidden_states=hidden_states, save_dist=False
            )
        finally:
            if hasattr(self.policy, "_onnx_scan_history_input"):
                del self.policy._onnx_scan_history_input

        # 当前帧 latent: policy 在 _process_obs 中通过 aux_outputs 暴露
        scan_lat_t = getattr(self.policy, "_last_scan_lat_t", None)

        if self.rnn_type == "lstm":
            outs = [action_mean, next_state[0], next_state[1]]
        else:
            outs = [action_mean, next_state]
        if self.use_scan_history and scan_lat_t is not None:
            outs.append(scan_lat_t)
        return tuple(outs) if len(outs) > 1 else outs[0]

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
        
        use_scan_history = getattr(policy, "use_scan_history", False)
        scan_history_len = getattr(policy, "scan_history_len", 0)
        scan_out_dim = getattr(policy, "scan_out_dim", 64)

        if rnn_type == "lstm":
            dummy_h0 = torch.zeros(1, batch_size, latent_dim, device=device)
            dummy_c0 = torch.zeros(1, batch_size, latent_dim, device=device)
            if use_scan_history:
                dummy_scan_hist = torch.zeros(batch_size, scan_history_len, scan_out_dim, device=device)
                inputs = (dummy_obs, dummy_est, dummy_h0, dummy_c0, dummy_scan_hist)
                input_names = ["proprio_and_env", "estimator_history", "h0", "c0", "scan_history"]
                output_names = ["action", "next_h0", "next_c0", "scan_lat_t"]
            else:
                inputs = (dummy_obs, dummy_est, dummy_h0, dummy_c0)
                input_names = ["proprio_and_env", "estimator_history", "h0", "c0"]
                output_names = ["action", "next_h0", "next_c0"]
        else:
            dummy_h0 = torch.zeros(1, batch_size, latent_dim, device=device)
            if use_scan_history:
                dummy_scan_hist = torch.zeros(batch_size, scan_history_len, scan_out_dim, device=device)
                inputs = (dummy_obs, dummy_est, dummy_h0, dummy_scan_hist)
                input_names = ["proprio_and_env", "estimator_history", "h0", "scan_history"]
                output_names = ["action", "next_h0", "scan_lat_t"]
            else:
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
        print("[Controls] B: lock/cycle Wheel expert | N: lock/cycle Leg expert | M: unlock all")
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
        print("[Controls] A: lock/cycle Wheel expert | B: lock/cycle Leg expert | Back: unlock all")
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
    # === Expert lock (interactive ablation) ===
    # lock_state[g] = None → adaptive gate; = int → force that single expert.
    # The hook rewrites the gate LOGITS to one-hot before softmax runs in
    # _compute_actor_output, so the locked expert gets weight 1.0 and the
    # group's action equals that expert's raw output, while the other group
    # stays fully adaptive. Registered BEFORE the capture hooks below, so the
    # dashboard expert bars also reflect the lock.
    lock_state = {"wheel": None, "leg": None}

    def make_lock_hook(group):
        def _lock_hook(module, inp, out):
            idx = lock_state[group]
            if idx is None:
                return None  # adaptive — leave gate logits untouched
            locked = torch.full_like(out, -1e9)
            locked[..., idx] = 0.0  # softmax → one-hot on expert `idx`
            return locked
        return _lock_hook

    if hasattr(model_instance, "wheel_gate"):
        model_instance.wheel_gate.register_forward_hook(make_lock_hook("wheel"))
    if hasattr(model_instance, "leg_gate"):
        model_instance.leg_gate.register_forward_hook(make_lock_hook("leg"))
    n_wheel_experts = len(getattr(model_instance, "actor_wheel_experts", []))
    n_leg_experts = len(getattr(model_instance, "actor_leg_experts", []))
    print(f"[Lock] wheel experts={n_wheel_experts}, leg experts={n_leg_experts} "
          f"— keyboard B/N/M  or  gamepad A/B/Back")

    def cycle_lock(group, n_experts):
        """Cycle a group's lock: adaptive -> expert 0 -> 1 -> ... -> adaptive."""
        cur = lock_state[group]
        nxt = 0 if cur is None else cur + 1
        lock_state[group] = None if nxt >= n_experts else nxt
        v = lock_state[group]
        tag = "W" if group == "wheel" else "L"
        return f"{group.capitalize()} lock -> {'adaptive' if v is None else tag + str(v)}"

    def clear_lock():
        lock_state["wheel"] = None
        lock_state["leg"] = None
        return "Expert lock cleared (all adaptive)"

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

        wl, ll = lock_state["wheel"], lock_state["leg"]
        w_txt = f"\033[91mW{wl} LOCKED\033[0m" if wl is not None else "\033[92madaptive\033[0m"
        l_txt = f"\033[91mL{ll} LOCKED\033[0m" if ll is not None else "\033[92madaptive\033[0m"
        lines.append(f"Expert Lock [B/N cycle, M clear]: Wheel={w_txt} | Leg={l_txt}")
        lines.append("-" * 30)

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
    lock_w_prev, lock_l_prev, unlock_prev = False, False, False

    OFFSET_FORWARD = [ -2.5, 0.0, 1.5 ]
    HEIGHT_TOP = 5.0
    OFFSET_BACKWARD = [ 2.5, 0.0, 1.5 ]
    camera_history = []
    if hasattr(base_env.scene.terrain, "terrain_levels"):
        cur_difficulty = base_env.scene.terrain.terrain_levels[0].item()
    if hasattr(base_env.scene.terrain, "terrain_types"):
        cur_subterrain = base_env.scene.terrain.terrain_types[0].item()

    # ===== Scan obs 可视化 (post-augmentation, 4 色分类) =====
    # 绿 (valid)     : obs 没被增强干掉, 落在策略真实"看到"的位置 (含 noise+latency 偏移)
    # 黄 (ramp)      : obs blind 且 raw_depth ∈ [0, 0.35]m → 距离渐变盲区干掉的
    # 紫 (random)    : obs blind 且 raw_depth ∈ [0.35, 2.4]m → 30-50% 随机 dropout 干掉的
    # 红 (no_return) : obs blind 且 raw 无 hit / >2.4m → 真无回波 (天空/超量程, 真机也是这样)
    # 黄/紫/红 都画在距 sensor 2.5m 的射线方向上 (= 策略的 "max_distance 球壳" 认知)
    scan_vis_markers = None
    if args.vis_scan_obs:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        scan_vis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/scan_obs_pts",
            markers={
                "valid": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "ramp": sim_utils.SphereCfg(
                    radius=0.028,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.95, 0.0)),
                ),
                "random": sim_utils.SphereCfg(
                    radius=0.028,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.0, 1.0)),
                ),
                "no_return": sim_utils.SphereCfg(
                    radius=0.018,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                # 高程图 (ElevationAE 的输入) — height_scanner 11x17 grid 的真实 ray_hit 位置
                "elevation": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),
                ),
            },
        )
        scan_vis_markers = VisualizationMarkers(scan_vis_cfg)
        print("[ScanVis] post-aug scan markers enabled (env 0). 绿=valid, 黄=ramp, 紫=random_dropout, 红=true_no_return, 青=elevation.")
        print("[ScanVis] 诊断日志写入 /tmp/scan_vis_diag.log — 另开终端 `tail -f /tmp/scan_vis_diag.log` 查看")
        # 清空旧日志
        try:
            with open("/tmp/scan_vis_diag.log", "w") as _f:
                _f.write("# ScanVis diagnostic log\n")
        except Exception:
            pass

    _scan_vis_diag = {"step": 0, "ramp_seen": 0, "last_print": -1}

    # 动态计算 noisy_elevation 组内 forward_scan / backward_scan 的真实切片位置。
    # height_scan 可能出现在 LIDAR 之前 (见 NoisyElevationCfg)，硬编码 (0,496)/(496,992) 会错位。
    _lidar_slices = []
    _height_scan_slice = None
    if args.vis_scan_obs:
        try:
            _om = base_env.observation_manager
            # active_terms: dict[group_name, list[term_name]] (in concatenation order)
            # group_obs_term_dim: dict[group_name, list[tuple[int, ...]]]
            _term_names = _om.active_terms["noisy_elevation"]
            _term_dims = _om.group_obs_term_dim["noisy_elevation"]
            # 支持两种 NoisyElevationCfg 布局:
            # (A) 2-sensor 半球 (main 版): forward_scan / backward_scan → forward_lidar / backward_lidar
            # (B) 12-sensor 单 pitch 弧 (baseline / recover 版):
            #     forward_scan_lN / backward_scan_lN → forward_scanner_layerN / backward_scanner_layerN
            _name_to_sensor = {
                # 2-sensor 半球
                "forward_scan": "forward_lidar",
                "backward_scan": "backward_lidar",
            }
            # 6+6 layer 命名
            for _i in range(6):
                _name_to_sensor[f"forward_scan_l{_i}"] = f"forward_scanner_layer{_i}"
                _name_to_sensor[f"backward_scan_l{_i}"] = f"backward_scanner_layer{_i}"
            _off = 0
            _height_scan_slice = None
            for _n, _d in zip(_term_names, _term_dims):
                _sz = int(_d[0])
                if _n in _name_to_sensor:
                    _lidar_slices.append((_name_to_sensor[_n], slice(_off, _off + _sz)))
                elif _n == "height_scan":
                    _height_scan_slice = slice(_off, _off + _sz)
                _off += _sz
            print(f"[ScanVis] dynamic slices ({len(_lidar_slices)} sensors) = {_lidar_slices} "
                  f"(group dim={_off}, height_scan slice={_height_scan_slice})")
        except Exception as _e:
            print(f"[ScanVis] dynamic slice lookup failed ({_e}); falling back to 12-layer baseline layout")
            _lidar_slices = []
            for _i in range(6):
                _lidar_slices.append((f"forward_scanner_layer{_i}", slice(_i*21, (_i+1)*21)))
            for _i in range(6):
                _lidar_slices.append((f"backward_scanner_layer{_i}", slice((6+_i)*21, (7+_i)*21)))

    # height_scan 可视化: 检测 obs 里是否真的有 height_scan 项, 且 scene 里挂了 height_scanner sensor
    _viz_height_scan = False
    if args.vis_scan_obs:
        try:
            _has_term = any(
                "height_scan" in _terms
                for _terms in base_env.observation_manager.active_terms.values()
            )
            _has_sensor = "height_scanner" in base_env.scene.sensors
            _viz_height_scan = _has_term and _has_sensor
            if _viz_height_scan:
                print("[ScanVis] elevation (青色) markers enabled — height_scanner ray_hits_w")
            else:
                print(f"[ScanVis] elevation viz off (has_term={_has_term}, has_sensor={_has_sensor})")
        except Exception as _e:
            print(f"[ScanVis] elevation viz disabled: {_e}")

    def _update_scan_obs_markers(env_obs_dict, env_idx=0):
        """4 色分类: 区分 valid / ramp / random_dropout / true_no_return."""
        if "noisy_elevation" not in env_obs_dict:
            return
        noisy = env_obs_dict["noisy_elevation"][env_idx]
        pts_list, idx_list = [], []
        # 收集诊断
        diag_counts = [0, 0, 0, 0]   # valid, ramp, random, no_return
        diag_raw_min = float("inf")
        diag_raw_close_count = 0      # 当前帧 raw_d < 0.35 的射线数

        for sname, slc in _lidar_slices:
            sensor = base_env.scene.sensors[sname]
            # 用 _ray_starts_w (真实射线起点, 含 offset) 而非 data.pos_w (= base_link).
            # 注: ray 数量取决于 pattern. 2-sensor 半球 = (496, 3); 12-sensor 单 pitch 弧 = (21, 3).
            ray_starts = sensor._ray_starts_w[env_idx]          # 每条射线的世界起点
            hits = sensor.data.ray_hits_w[env_idx]
            vec = hits - ray_starts
            raw_d = vec.norm(dim=-1)                            # 真正的 sensor→hit 深度
            ray_dir = torch.nan_to_num(
                vec / raw_d.clamp(min=1e-3).unsqueeze(-1),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

            obs_norm = noisy[slc]
            obs_depth = (obs_norm * 2.5).clamp(0.0, 2.5)
            obs_blind = obs_norm > 0.95
            raw_finite = torch.isfinite(raw_d)
            raw_clean = torch.where(raw_finite, raw_d, torch.full_like(raw_d, 99.0))

            # 分类 (默认 0=valid)
            idx = torch.zeros_like(obs_blind, dtype=torch.long)
            no_return = obs_blind & ((~raw_finite) | (raw_clean > 2.4))
            ramp = obs_blind & (~no_return) & (raw_clean < 0.35)
            random_drop = obs_blind & (~no_return) & (raw_clean >= 0.35)
            idx = torch.where(ramp, torch.full_like(idx, 1), idx)
            idx = torch.where(random_drop, torch.full_like(idx, 2), idx)
            idx = torch.where(no_return, torch.full_like(idx, 3), idx)

            # 位置: valid → obs 深度 (含噪/延迟); blind 三类 → 2.5m 球壳 (策略的"max_dist 认知")
            # 起点用 ray_starts (真实 sensor 位置), 不是 base_link
            depth_for_vis = torch.where(obs_blind, torch.full_like(obs_depth, 2.5), obs_depth)
            new_pts = ray_starts + ray_dir * depth_for_vis.unsqueeze(-1)

            pts_list.append(new_pts)
            idx_list.append(idx)

            # 累计诊断
            diag_counts[0] += int((idx == 0).sum().item())
            diag_counts[1] += int((idx == 1).sum().item())
            diag_counts[2] += int((idx == 2).sum().item())
            diag_counts[3] += int((idx == 3).sum().item())
            if raw_finite.any():
                diag_raw_min = min(diag_raw_min, float(raw_clean[raw_finite].min().item()))
            diag_raw_close_count += int((raw_finite & (raw_d < 0.35)).sum().item())

        # 高程图可视化 (post-obs, 与 LIDAR 4 色同口径):
        # obs_hs ∈ [-1, 1] = pos_w_z - ground_z - 0.5  (含 ±0.1 噪声, clip 后)
        # → ground_z_seen_by_policy = pos_w_z - obs_hs - 0.5
        # XY 仍取真实栅格位置 (_ray_starts_w 的 xy 分量, 已含 yaw 旋转后的 sensor 偏移)
        if _viz_height_scan and _height_scan_slice is not None:
            hs = base_env.scene.sensors["height_scanner"]
            obs_hs = noisy[_height_scan_slice]                            # (187,) 已 noise+clip
            ray_xy = hs._ray_starts_w[env_idx, :, :2]                     # (187, 2) world XY
            pos_w_z = hs.data.pos_w[env_idx, 2]                           # scalar body z
            ground_z = pos_w_z - obs_hs - 0.5                             # (187,)
            elev_pts = torch.cat([ray_xy, ground_z.unsqueeze(-1)], dim=-1)
            pts_list.append(elev_pts)
            idx_list.append(torch.full(
                (elev_pts.shape[0],), 4,
                dtype=torch.long, device=elev_pts.device,
            ))

        all_pts = torch.cat(pts_list, dim=0).cpu().numpy()
        all_idx = torch.cat(idx_list, dim=0).cpu().numpy().tolist()
        scan_vis_markers.visualize(translations=all_pts, marker_indices=all_idx)

        # ---- 诊断打印 ----
        s = _scan_vis_diag
        s["step"] += 1
        if diag_counts[1] > 0:
            s["ramp_seen"] += 1
        # 触发条件: 每 60 帧 (~1.2s) 打印一次, 或 raw_d 出现 < 0.35 的射线时强制打印
        force = diag_raw_close_count > 0 and s["last_print"] != s["step"]
        periodic = (s["step"] % 60 == 0)
        if force or periodic:
            s["last_print"] = s["step"]
            line = (f"[ScanVis step={s['step']:5d}] "
                    f"green={diag_counts[0]:3d}  yellow={diag_counts[1]:3d}  "
                    f"purple={diag_counts[2]:3d}  red={diag_counts[3]:3d}  "
                    f"| raw_min={diag_raw_min:.2f}m  raw<0.35: {diag_raw_close_count} rays  "
                    f"| ramp_seen_total={s['ramp_seen']}/{s['step']}\n")
            try:
                with open("/tmp/scan_vis_diag.log", "a") as _f:
                    _f.write(line)
            except Exception:
                pass

    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            obs_dict = base_env.obs_buf
            if scan_vis_markers is not None:
                _update_scan_obs_markers(obs_dict, env_idx=0)
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
                # 半球形 LidarPattern: 16 ch × 31 az = 496 rays per direction (front / back)
                f_sens = base_env.scene.sensors["forward_lidar"]
                f_dist = torch.norm(f_sens.data.ray_hits_w[0] - f_sens.data.pos_w[0], dim=-1)
                raw_fwd = torch.nan_to_num(f_dist, posinf=5.0, neginf=5.0, nan=5.0).cpu().tolist()
                b_sens = base_env.scene.sensors["backward_lidar"]
                b_dist = torch.norm(b_sens.data.ray_hits_w[0] - b_sens.data.pos_w[0], dim=-1)
                raw_bwd = torch.nan_to_num(b_dist, posinf=5.0, neginf=5.0, nan=5.0).cpu().tolist()

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

            # --- Expert-lock input: gamepad A/B/Back  or  keyboard B/N/M ---
            do_lock_w = do_lock_l = do_unlock = False
            if args.joystick and controller is not None:
                lw_curr = controller.is_button_pressed(0)   # A
                ll_curr = controller.is_button_pressed(1)   # B
                un_curr = controller.is_button_pressed(6)   # Back
                do_lock_w = lw_curr and not lock_w_prev
                do_lock_l = ll_curr and not lock_l_prev
                do_unlock = un_curr and not unlock_prev
                lock_w_prev, lock_l_prev, unlock_prev = lw_curr, ll_curr, un_curr
            elif kb_ext is not None:
                do_lock_w = kb_ext.check_and_clear(carb.input.KeyboardInput.B)
                do_lock_l = kb_ext.check_and_clear(carb.input.KeyboardInput.N)
                do_unlock = kb_ext.check_and_clear(carb.input.KeyboardInput.M)
            if do_lock_w:
                status_message = cycle_lock("wheel", n_wheel_experts)
                status_timer = 30
            if do_lock_l:
                status_message = cycle_lock("leg", n_leg_experts)
                status_timer = 30
            if do_unlock:
                status_message = clear_lock()
                status_timer = 30

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
                # 首先执行重置 (让 IsaacLab 内部逻辑跑完，它可能会在这里偷偷改 level)
                obs, _ = env_wrapped.reset()

                # 然后：强制覆盖回用户指定的 Difficulty 和 Type
                if hasattr(base_env.scene.terrain, "terrain_levels"):
                    levels_vec = torch.full((env_wrapped.num_envs,), cur_difficulty, device=env_wrapped.device, dtype=torch.long)
                    base_env.scene.terrain.terrain_levels[:] = levels_vec
                
                if hasattr(base_env.scene.terrain, "terrain_types"):
                    types_vec = torch.full((env_wrapped.num_envs,), cur_subterrain, device=env_wrapped.device, dtype=torch.long)
                    base_env.scene.terrain.terrain_types[:] = types_vec
                
                # 重新计算出生点
                base_env.scene.terrain.update_env_origins(
                    env_ids=torch.arange(env_wrapped.num_envs, device=env_wrapped.device, dtype=torch.long),
                    move_up=torch.zeros(env_wrapped.num_envs, device=env_wrapped.device, dtype=torch.long),
                    move_down=torch.zeros(env_wrapped.num_envs, device=env_wrapped.device, dtype=torch.long)
                )

                # 5. 使用原生场景属性 base_env.scene.env_origins 进行传送
                if robot_entity is not None:
                    # 直接获取 update_env_origins 计算出的坐标
                    target_pos = base_env.scene.env_origins.clone()
                    target_pos[:, 2] += 0.55  # 抬高一点防止穿模
                    
                    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env_wrapped.device).repeat(env_wrapped.num_envs, 1)
                    root_pose = torch.cat([target_pos, default_quat], dim=-1)
                    
                    # 写入仿真
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