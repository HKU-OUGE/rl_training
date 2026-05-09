"""单卡可视化检查 8 种 per-rank 地形 (含键盘 / 手柄交互).

用法:
    conda activate env_isaaclab
    python scripts/reinforcement_learning/rsl_rl/inspect_terrain.py --rank 0
    python scripts/reinforcement_learning/rsl_rl/inspect_terrain.py --rank 1 --max_level
    python scripts/reinforcement_learning/rsl_rl/inspect_terrain.py --rank 3 --num_envs 16 --joystick

键盘热键 (Isaac Sim 视口聚焦时按):
    T / G : terrain_levels 行 +1 / -1   (难度升降, curriculum 维度)
    H / F : terrain_types  列 +1 / -1   (sub_terrain 切换, 不同变体)
    C     : 切换到全图自由俯视视角 (top-down 鸟瞰所有 env)
    V     : 切换到跟随 env 0 视角
    R     : 重置机器人到当前 env_origin
    ESC   : 退出 (或 Ctrl-C)

手柄 (--joystick + /dev/input/js0):
    DPad ↑/↓ : level 升/降
    DPad ←/→ : type 上/下
    Y / X    : 全图视角 / 跟随视角
    B        : 重置机器人

Isaac Sim 视口操作 (无热键, 鼠标):
    左键拖动 : 旋转视角
    中键拖动 : 平移
    滚轮     : 缩放
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=0,
                    help="0=FLAT, 1=STAIR_SLOPE, 2=PLATFORM, 3=SCAN, 4=GAP, 5=RAIL, 6=NOISE, 7=GRID")
parser.add_argument("--num_envs", type=int, default=9,
                    help="一次显示几个 env (3x3 grid 默认 9)")
parser.add_argument("--max_level", action="store_true",
                    help="启动后强制 terrain_levels = max, 看最高难度地形")
parser.add_argument("--joystick", action="store_true",
                    help="启用 /dev/input/js0 手柄输入 (键盘默认始终开)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# 不带 --headless, 默认开视口

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

sys.path.append(os.getcwd())
import rl_training.tasks  # noqa: F401

import carb.input
import omni.appwindow


# ============================================================================
# 键盘监听 (沿用 play_moe.py 的 KeyboardExtension 实现)
# ============================================================================
class KeyboardWatcher:
    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        self._kbd = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._kbd, self._on_evt)
        self._pressed = {}

    def _on_evt(self, e, *_):
        if e.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed[e.input] = True
        return True

    def check(self, key):
        if self._pressed.get(key, False):
            self._pressed[key] = False
            return True
        return False


# ============================================================================
# 简单 /dev/input/js0 手柄监听 (可选)
# ============================================================================
class SimpleJoystick:
    """读 Linux js0, 维护当前 axis/button state. 仅做基础 dpad + 几个按钮."""
    def __init__(self, path="/dev/input/js0"):
        try:
            self.fd = open(path, "rb")
            self.fd_no = self.fd.fileno()
            import fcntl
            fcntl.fcntl(self.fd_no, fcntl.F_SETFL, os.O_NONBLOCK)
            self.connected = True
        except Exception as e:
            print(f"  [joystick] cannot open {path}: {e}, joystick disabled")
            self.connected = False
            self.fd = None
        self.axes = {}
        self.buttons = {}
        self.button_just_pressed = {}

    def poll(self):
        if not self.connected:
            return
        import struct
        try:
            while True:
                data = self.fd.read(8)
                if not data or len(data) < 8:
                    break
                _t, value, evt_type, number = struct.unpack("IhBB", data)
                if evt_type & 0x01:  # button
                    prev = self.buttons.get(number, 0)
                    self.buttons[number] = value
                    if value and not prev:
                        self.button_just_pressed[number] = True
                elif evt_type & 0x02:  # axis
                    self.axes[number] = value / 32767.0
        except BlockingIOError:
            pass

    def button_edge(self, n):
        """Button N just pressed (rising edge), one-shot."""
        if self.button_just_pressed.get(n, False):
            self.button_just_pressed[n] = False
            return True
        return False

    def dpad_dir(self):
        """Return ('up','down','left','right',None) based on dpad axes (typically axis 6/7)."""
        if not self.connected:
            return None
        ax = self.axes.get(7, 0.0)  # vertical: -1 up, 1 down
        ay = self.axes.get(6, 0.0)  # horizontal: -1 left, 1 right
        if ax < -0.5: return "up"
        if ax > 0.5:  return "down"
        if ay < -0.5: return "left"
        if ay > 0.5:  return "right"
        return None
    _last_dpad = None
    def dpad_edge(self):
        """One-shot dpad direction (only triggers on transition)."""
        cur = self.dpad_dir()
        if cur != self._last_dpad:
            self._last_dpad = cur
            return cur
        return None

# ===== 8 种 per-rank 地形 (跟 train_moe.py 派发顺序一致) =====
import isaaclab.terrains as _tg
from isaaclab.terrains import TerrainGeneratorCfg
from rl_training.terrains.config.rough import (
    STAIR_SLOPE_TEACHER_TERRAINS_CFG,
    PLATFORM_TEACHER_TERRAINS_CFG,
    SCAN_TEACHER_TERRAINS_CFG,
    GAP_TEACHER_TERRAINS_CFG,
    STEPPING_STONES_TEACHER_TERRAINS_CFG,
    RAIL_TEACHER_TERRAINS_CFG,
    NOISE_TEACHER_TERRAINS_CFG,
    GRID_TEACHER_TERRAINS_CFG,
)
_FLAT_CFG = TerrainGeneratorCfg(
    size=(12.0, 12.0), border_width=20.0,
    num_rows=10, num_cols=10, curriculum=False,
    sub_terrains={"flat": _tg.MeshPlaneTerrainCfg(proportion=1.0)},
)
RANK_TERRAIN_MAP = [
    _FLAT_CFG,                          # 0
    STAIR_SLOPE_TEACHER_TERRAINS_CFG,   # 1
    PLATFORM_TEACHER_TERRAINS_CFG,      # 2
    SCAN_TEACHER_TERRAINS_CFG,          # 3
    _FLAT_CFG,                          # 4 (was GAP/STONES, 训练失败, 暂用 FLAT)
    RAIL_TEACHER_TERRAINS_CFG,          # 5
    NOISE_TEACHER_TERRAINS_CFG,         # 6
    GRID_TEACHER_TERRAINS_CFG,          # 7
]
RANK_NAMES = ["FLAT", "STAIR_SLOPE", "PLATFORM", "SCAN", "FLAT2", "RAIL", "NOISE", "GRID"]


def main():
    if args.rank < 0 or args.rank >= len(RANK_TERRAIN_MAP):
        print(f"❌ --rank must be 0-{len(RANK_TERRAIN_MAP)-1}")
        return

    chosen = RANK_TERRAIN_MAP[args.rank]
    rank_name = RANK_NAMES[args.rank]

    print("\n" + "=" * 70)
    print(f"  RANK {args.rank}  →  {rank_name}")
    print("=" * 70)
    print(f"  size            : {chosen.size}")
    print(f"  num_rows × cols : {chosen.num_rows} × {chosen.num_cols}")
    print(f"  curriculum      : {chosen.curriculum}")
    print(f"  border_width    : {chosen.border_width}")
    print(f"  sub_terrains    :")
    for name, st in chosen.sub_terrains.items():
        cls = type(st).__name__
        prop = getattr(st, "proportion", "?")
        # 关键参数 dump
        kparams = {}
        for k in ("step_height_range", "slope_range", "gap_width_range",
                  "rail_height_range", "rail_thickness_range",
                  "pit_depth_range", "box_height_range",
                  "noise_range", "grid_height_range", "grid_width",
                  "platform_width", "step_width", "double_pit", "double_box"):
            if hasattr(st, k):
                kparams[k] = getattr(st, k)
        print(f"      • {name:20s}  {cls}  prop={prop}")
        for k, v in kparams.items():
            print(f"          {k}: {v}")
    print("=" * 70 + "\n")

    env_cfg = parse_env_cfg(
        "Rough-MoE-Teacher-Deeprobotics-M20-v0",
        device="cuda:0", num_envs=args.num_envs,
    )
    env_cfg.scene.terrain.terrain_generator = chosen

    print(f">>> Creating env with {args.num_envs} envs ...")
    env_gym = gym.make("Rough-MoE-Teacher-Deeprobotics-M20-v0", cfg=env_cfg)
    env = env_gym.unwrapped

    env_gym.reset()  # 必须用 wrapped env_gym 而非 env, 否则 env_gym.step 报 ResetNeeded

    if args.max_level and chosen.curriculum:
        ti = env.scene.terrain
        max_level = ti.terrain_origins.shape[0] - 1
        ti.terrain_levels[:] = max_level
        ti.env_origins[:] = ti.terrain_origins[ti.terrain_levels, ti.terrain_types]
        env_gym.reset()
        print(f">>> Forced terrain_levels → {max_level} (max difficulty)\n")

    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros((args.num_envs, action_dim), device="cuda:0")

    # ----- 输入设备 -----
    kb = KeyboardWatcher()
    js = SimpleJoystick() if args.joystick else None

    # ----- 相机控制 -----
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        from omni.isaac.core.utils.viewports import set_camera_view

    cam_state = {"mode": "overview"}  # 用 dict 避免 nonlocal

    def snap_overview():
        cam_state["mode"] = "overview"
        eo = env.scene.env_origins
        cx = eo[:, 0].mean().item()
        cy = eo[:, 1].mean().item()
        extent = max(
            (eo[:, 0].max() - eo[:, 0].min()).item(),
            (eo[:, 1].max() - eo[:, 1].min()).item(),
            8.0,
        )
        eye_z = max(extent * 1.5, 30.0)
        set_camera_view([cx, cy, eye_z], [cx, cy, 0])
        print(f"  📷 OVERVIEW @ ({cx:.1f}, {cy:.1f}, {eye_z:.1f})", flush=True)

    def snap_follow_env0():
        cam_state["mode"] = "follow"
        robot = env.scene["robot"]
        p = robot.data.root_pos_w[0].cpu().numpy()
        set_camera_view([p[0]-3, p[1]-3, p[2]+2], [p[0], p[1], p[2]])
        print(f"  📷 FOLLOW env_0 @ ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})", flush=True)

    def reapply_camera():
        """切换地形/重置后重新应用当前相机模式 (env_origin 或 robot.pos 已变化)."""
        if cam_state["mode"] == "follow":
            snap_follow_env0()
        else:
            snap_overview()

    # ----- 地形切换 -----
    def cycle_level(delta):
        ti = env.scene.terrain
        n_rows = int(ti.terrain_origins.shape[0])
        new_levels = ((ti.terrain_levels.long() + delta) % n_rows).long()
        ti.terrain_levels.copy_(new_levels)
        ti.env_origins.copy_(ti.terrain_origins[ti.terrain_levels, ti.terrain_types])
        env_gym.reset()
        # reset 后还要再 step 一次让 robot.data 更新到新位置, 这样 follow 视角才对
        env_gym.step(zero_action)
        print(f"  ↕ level = {ti.terrain_levels[0].item()} / {n_rows-1}", flush=True)
        reapply_camera()

    def cycle_type(delta):
        ti = env.scene.terrain
        n_cols = int(ti.terrain_origins.shape[1])
        new_types = ((ti.terrain_types.long() + delta) % n_cols).long()
        ti.terrain_types.copy_(new_types)
        ti.env_origins.copy_(ti.terrain_origins[ti.terrain_levels, ti.terrain_types])
        env_gym.reset()
        env_gym.step(zero_action)
        sub_names = list(chosen.sub_terrains.keys())
        print(f"  ↔ type = {ti.terrain_types[0].item()} / {n_cols-1} "
              f"(sub_terrains: {sub_names})", flush=True)
        reapply_camera()

    def reset_robots():
        env_gym.reset()
        env_gym.step(zero_action)
        print(f"  🔄 reset", flush=True)
        reapply_camera()

    # 初始相机给个 overview
    snap_overview()

    print("\n" + "=" * 70)
    print("  键盘热键: T/G level±   H/F type±   C overview   V follow   R reset   ESC quit")
    if args.joystick and js and js.connected:
        print("  手柄:    DPad↕ level±   DPad↔ type±   Y overview   X follow   B reset")
    print("=" * 70 + "\n", flush=True)

    step_i = 0
    try:
        while simulation_app.is_running():
            # ---- 输入处理 ----
            ki = carb.input.KeyboardInput
            if kb.check(ki.T):       cycle_level(+1)
            elif kb.check(ki.G):     cycle_level(-1)
            elif kb.check(ki.H):     cycle_type(+1)
            elif kb.check(ki.F):     cycle_type(-1)
            elif kb.check(ki.C):     snap_overview()
            elif kb.check(ki.V):     snap_follow_env0()
            elif kb.check(ki.R):     reset_robots()
            elif kb.check(ki.ESCAPE): break

            if js and js.connected:
                js.poll()
                d = js.dpad_edge()
                if d == "up":     cycle_level(+1)
                elif d == "down": cycle_level(-1)
                elif d == "right": cycle_type(+1)
                elif d == "left":  cycle_type(-1)
                if js.button_edge(3): snap_overview()      # Y
                if js.button_edge(2): snap_follow_env0()   # X
                if js.button_edge(1): reset_robots()       # B

            # ---- 推进物理 ----
            try:
                env_gym.step(zero_action)
            except Exception as e:
                print(f"    [step {step_i}] env.step raised: {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()
                break
            step_i += 1
        print(f">>> Loop exited at step {step_i}", flush=True)
    except KeyboardInterrupt:
        print("\n>>> Caught Ctrl-C", flush=True)
    finally:
        env_gym.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
