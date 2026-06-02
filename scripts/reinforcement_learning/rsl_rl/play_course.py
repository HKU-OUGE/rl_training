"""Manually drive the M20 robot through the v2.1 obstacle course while the
trained SplitMoE (full) policy runs. Diagnostic tool for the ``full × Easy``
failure mode — in particular suspected course-geometry issues at ``stones``.

Usage
-----
Launch with X display::

    python scripts/reinforcement_learning/rsl_rl/play_course.py \
        --level easy --num_envs 1 --keyboard

Headless smoke test (boot env + policy, run N steps, exit)::

    python scripts/reinforcement_learning/rsl_rl/play_course.py \
        --level easy --num_envs 1 --headless --smoke_steps 10

Key bindings (--keyboard): W/S = cmd_vx+/-, A/D = cmd_vy+/-, Q/E = cmd_wz+/-,
P = pause (zero cmd), R = reset (teleport to course start), C = cycle camera
mode (forward / top-down / back-face), L = lock/unlock follow camera.

Joystick (--joystick, /dev/input/js0): L-stick = cmd_vx/vy, R-stick X = cmd_wz,
D-pad = vx/vy digital, A = lock cam, X = reset, Y = cycle camera mode.

The Course env auto-sets cmd_vx = 1.0; this script REPLACES that with the
user's input (falling back to 1.0 m/s forward when idle).
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# Allow `from rl_utils import camera_follow` etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# 1. CLI + AppLauncher (must run BEFORE heavy isaac imports)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Drive the M20 manually through the v2.1 course.")
parser.add_argument("--level", type=str, default="easy",
                    choices=["easy", "med", "hard", "extreme"])
parser.add_argument("--ablation", type=str, default="full",
                    choices=["full", "A1", "A2", "A3", "B1", "B2", "locomoe", "mlp_baseline"],
                    help="Which trained variant to play with.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_run", type=str, default=None,
                    help="Run dir name or absolute path under "
                         "logs/moe_training/split_moe_teacher_parallel; default = latest.")
parser.add_argument("--checkpoint", type=str, default="model_*.pt")
parser.add_argument("--keyboard", action="store_true", default=False)
parser.add_argument("--joystick", action="store_true", default=False)
parser.add_argument("--smoke_steps", type=int, default=0,
                    help="If >0, run N policy steps and exit (smoke test, no interactive loop).")
parser.add_argument("--speed", type=int, default=1,
                    help="Sim steps per render frame (1=real time, 2=2×, 4=4× ...). "
                         "Increment/decrement runtime with +/- keys.")
parser.add_argument("--hud_every", type=int, default=10,
                    help="Refresh HUD every N control steps (default 10 ≈ 5 Hz at dt=0.02).")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 2. Heavy imports
# ---------------------------------------------------------------------------
import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
import gymnasium as gym                                  # noqa: E402

import carb                                              # noqa: E402
import carb.input                                        # noqa: E402
try:
    import omni.appwindow                                # noqa: E402
    _HAS_APPWINDOW = True
except ModuleNotFoundError:
    _HAS_APPWINDOW = False
    omni = None

from isaaclab_tasks.utils import parse_env_cfg           # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper        # noqa: E402
from rsl_rl.runners import OnPolicyRunner                # noqa: E402
from isaaclab.devices import Se2Keyboard                 # noqa: E402
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg  # noqa: E402
import isaaclab.utils.math as math_utils                 # noqa: E402

# Inject SplitMoE classes into rsl_rl namespace (copied from eval_course.py)
sys.path.append(os.getcwd())
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import (  # noqa: E402
    SplitMoEActorCritic,
    SplitMoEPPO,
)
import rsl_rl.modules as rsl_modules                     # noqa: E402
import rsl_rl.runners.on_policy_runner as runner_module  # noqa: E402
rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEPPO = SplitMoEPPO

from rl_training.terrains.config.course import (         # noqa: E402
    COURSE_PATCH_END_X,
    COURSE_LENGTH,
    COURSE_NUM_PATCHES,
    PATCH_NAMES,
)

# Reuse the joystick driver + keyboard helper from play_moe.py (already vetted).
# play_moe.py parses argv and launches its own AppLauncher at import time, so we
# (a) stub AppLauncher → no-op shim returning OUR already-running app
# (b) trim sys.argv so play_moe's parser doesn't choke on our --level / --smoke_steps
import isaaclab.app as _isaac_app                         # noqa: E402

_real_AppLauncher = _isaac_app.AppLauncher


class _StubAppLauncher:
    def __init__(self, *_args, **_kwargs):
        pass

    @property
    def app(self):
        return simulation_app

    @staticmethod
    def add_app_launcher_args(_parser):
        return None


_isaac_app.AppLauncher = _StubAppLauncher
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    from play_moe import (                                # noqa: E402
        DirectLinuxGamepad,
        KeyboardExtension,
        get_truly_unwrapped_env,
        resolve_checkpoint_path,
    )
finally:
    _isaac_app.AppLauncher = _real_AppLauncher
    sys.argv = _saved_argv

DEVICE = "cuda:0"

TASK_PER_LEVEL = {
    "easy": "Course-MoE-Teacher-Deeprobotics-M20-easy-v0",
    "med": "Course-MoE-Teacher-Deeprobotics-M20-med-v0",
    "hard": "Course-MoE-Teacher-Deeprobotics-M20-hard-v0",
    "extreme": "Course-MoE-Teacher-Deeprobotics-M20-extreme-v0",
}


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

def override_env_origins(base_env, env_cfg, num_envs):
    """Copy of eval_course.py:420-443 — pin every env's origin to course start."""
    tgen = env_cfg.scene.terrain.terrain_generator
    # v2.2: 0.5m inset so robot footprint sits fully on the lead-in slab.
    SPAWN_INSET = 0.5
    x_start = -float(tgen.size[0]) * float(tgen.num_rows) / 2.0 + SPAWN_INSET
    cur_z = base_env.scene.env_origins[:, 2].clone()
    new_origin = torch.zeros_like(base_env.scene.env_origins)
    new_origin[:, 0] = x_start
    new_origin[:, 1] = 0.0
    new_origin[:, 2] = cur_z
    base_env.scene.env_origins.copy_(new_origin)
    if hasattr(base_env.scene, "terrain"):
        ti = base_env.scene.terrain
        if hasattr(ti, "env_origins"):
            ti.env_origins.copy_(new_origin)
    print(f"[play_course] env_origins overridden: x_start={x_start:.3f}  "
          f"sample_z={cur_z[0].item():.3f}  num_envs={num_envs}")
    return x_start


def patch_index_for_x(disp_x: float) -> int:
    """Return the 0-based index of the patch the robot is currently on.

    Returns -1 while still in the lead-in pad, COURSE_NUM_PATCHES once past
    the last patch (i.e. on the tail-out pad).
    """
    if disp_x < COURSE_PATCH_END_X[0] - 1.2:  # 1.2 = BUFFER_LENGTH; still in lead-in
        # We approximate: patch_end_x[0] is end of (lead-in + obs0 + buf), so
        # subtracting buf gets us obs0 entry. Anything below = lead-in.
        return -1
    for idx, end_x in enumerate(COURSE_PATCH_END_X):
        if disp_x < end_x:
            return idx
    return COURSE_NUM_PATCHES  # past the last patch


def teleport_to_start(base_env, env_wrapped, x_start):
    """Hard-reset robot pose to the env origin (course-start spawn pad)."""
    robot = base_env.scene["robot"]
    target_pos = base_env.scene.env_origins.clone()
    target_pos[:, 2] += 0.55      # lift above ground to avoid penetration
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env_wrapped.device)
    quat = quat.repeat(env_wrapped.num_envs, 1)
    root_pose = torch.cat([target_pos, quat], dim=-1)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_link_vel_w))


def dump_robot_state(base_env, cmd_state, patch_idx):
    """One-shot diagnostic: print pose, vel, joints, cmd for env 0 to compare
    state at different points in the course (e.g. stones-1 entry vs stones-2)."""
    robot = base_env.scene["robot"]
    pos = robot.data.root_pos_w[0].cpu().numpy()
    quat = robot.data.root_quat_w[0].cpu().numpy()
    lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
    ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()
    jp = robot.data.joint_pos[0].cpu().numpy()
    jv = robot.data.joint_vel[0].cpu().numpy()
    # convert quat → roll, pitch, yaw (intrinsic Tait-Bryan, world-frame, deg)
    import numpy as np
    w, x, y, z = quat
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(max(-1.0, min(1.0, 2*(w*y - z*x)))))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    env_cmd = base_env.command_manager.get_command("base_velocity")[0].cpu().numpy()

    print("=" * 60)
    print(f"[DUMP] patch={patch_idx}")
    print(f"  pos (world):  x={pos[0]:+.3f}  y={pos[1]:+.4f}  z={pos[2]:+.3f}")
    print(f"  rpy (deg):    r={roll:+.2f}  p={pitch:+.2f}  y={yaw:+.2f}")
    print(f"  body vel:     vx={lin_vel[0]:+.3f}  vy={lin_vel[1]:+.3f}  vz={lin_vel[2]:+.3f}")
    print(f"  body angvel:  ωr={ang_vel[0]:+.3f}  ωp={ang_vel[1]:+.3f}  ωy={ang_vel[2]:+.3f}")
    print(f"  env cmd:      vx={env_cmd[0]:+.3f}  vy={env_cmd[1]:+.3f}  wz={env_cmd[2]:+.3f}")
    print(f"  cmd_state:    vx={cmd_state['vx']:+.3f}  vy={cmd_state['vy']:+.3f}  wz={cmd_state['wz']:+.3f}  user_active={cmd_state['user_active']}  paused={cmd_state['paused']}")
    print(f"  joint_pos:    {' '.join(f'{x:+.2f}' for x in jp)}")
    print(f"  joint_vel:    {' '.join(f'{x:+.2f}' for x in jv[:8])} ...  ({len(jv)} dof total)")
    print("=" * 60)


def reset_gru_state(model, batch_size, device):
    """Zero out the GRU hidden state of the policy (and critic) — diagnostic
    aid to test whether cumulative OOD history is the cause of late-cycle
    failures. After reset, the next forward pass starts from a fresh state
    as if the robot just spawned.
    """
    fresh = model._init_rnn_state(batch_size, device)
    model.active_hidden_states = fresh
    if hasattr(model, "active_critic_hidden_states"):
        fresh_c = model._init_rnn_state(batch_size, device)
        model.active_critic_hidden_states = fresh_c


def build_train_cfg_dict(task, ablation="full"):
    """Load + lightly post-process the train cfg. Branches by ablation:
    locomoe / mlp_baseline load their OWN PPOCfg; A1-B2 reuse SplitMoE with
    one-line policy overrides (matches eval_course.py / eval_moe.py).
    """
    if ablation == "locomoe":
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.locomoe_terrain import (
            LocoMoEActorCritic, LocoMoEPPO, LocoMoEPPOCfg,
        )
        import rsl_rl.modules as rsl_modules
        import rsl_rl.runners.on_policy_runner as runner_module
        rsl_modules.LocoMoEActorCritic = LocoMoEActorCritic
        runner_module.LocoMoEActorCritic = LocoMoEActorCritic
        runner_module.LocoMoEPPO = LocoMoEPPO
        train_cfg = LocoMoEPPOCfg()
        train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
        train_cfg_dict["policy"]["class_name"] = "LocoMoEActorCritic"
        train_cfg_dict["algorithm"]["class_name"] = "LocoMoEPPO"
        train_cfg_dict["experiment_name"] = "locomoe_teacher_parallel"
    elif ablation == "mlp_baseline":
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import (
            MlpHeadActorCritic, MlpBaselinePPOCfg,
        )
        import rsl_rl.modules as rsl_modules
        import rsl_rl.runners.on_policy_runner as runner_module
        rsl_modules.MlpHeadActorCritic = MlpHeadActorCritic
        runner_module.MlpHeadActorCritic = MlpHeadActorCritic
        train_cfg = MlpBaselinePPOCfg()
        train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
        train_cfg_dict["policy"]["class_name"] = "MlpHeadActorCritic"
        train_cfg_dict["experiment_name"] = "mlp_baseline_teacher_parallel"
    else:
        train_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
        train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
        for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
            train_cfg_dict["policy"].pop(k, None)
        # SplitMoE ablation overrides (A1 / B2 are arch overrides; A2/A3/B1 inference-equivalent)
        _ABL_OVERRIDES = {"A1": {"single_gate": True}, "B2": {"blind_vision": True}}
        if ablation in _ABL_OVERRIDES:
            for k, v in _ABL_OVERRIDES[ablation].items():
                train_cfg_dict["policy"][k] = v
        if ablation != "full":
            train_cfg_dict["experiment_name"] = f"split_moe_teacher_parallel_abl_{ablation}"

    # Mirror eval_course: collapse any distillation overrides into a standard PPO
    algo_class_name = train_cfg_dict["algorithm"].get("class_name", "")
    if "Distillation" in algo_class_name:
        train_cfg_dict["algorithm"]["class_name"] = "PPO"
        for k in ["gradient_length", "loss_type", "optimizer"]:
            train_cfg_dict["algorithm"].pop(k, None)
        train_cfg_dict["algorithm"].setdefault("value_loss_coef", 1.0)
        train_cfg_dict["algorithm"].setdefault("use_clipped_value_loss", True)
        train_cfg_dict["algorithm"].setdefault("clip_param", 0.2)
        train_cfg_dict["algorithm"].setdefault("entropy_coef", 0.01)
    train_cfg_dict["device"] = DEVICE
    train_cfg_dict["logger"] = "tensorboard"
    return train_cfg_dict


def load_split_moe_ckpt(env_wrapped, train_cfg_dict, ckpt_path, run_dir):
    """Build runner, load weights (handles both PPO and distillation)."""
    runner = OnPolicyRunner(env_wrapped, train_cfg_dict, log_dir=run_dir, device=DEVICE)
    loaded = torch.load(ckpt_path, map_location=DEVICE)
    sd = loaded.get("model_state_dict", loaded)
    if any(k.startswith("student.") for k in sd.keys()):
        print("[play_course] distilled ckpt — loading 'student' subnet")
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("student.") and "critic" not in k:
                new_sd[k.replace("student.", "", 1)] = v
        runner.alg.policy.load_state_dict(new_sd, strict=False)
    else:
        runner.load(ckpt_path)
    model = runner.alg.policy
    model.eval()
    return runner, model


# ---------------------------------------------------------------------------
# 4. HUD
# ---------------------------------------------------------------------------

class CourseHUD:
    """Tiny terminal HUD: x-position, patch index, cmd values, term flags."""

    def __init__(self):
        self._last_lines = 0

    def render(self, *, step, sim_time, disp_x, patch_idx, cmd, paused,
               controller_kind, controller_connected, last_term):
        lines = []
        lines.append("=" * 18 + " play_course HUD " + "=" * 18)
        lines.append(f"step={step:6d}  sim_t={sim_time:7.2f}s   "
                     f"ctrl={controller_kind} "
                     f"{'CONNECTED' if controller_connected else 'OFFLINE'}")
        # course progress bar
        bar_w = 40
        ratio = max(0.0, min(disp_x / COURSE_LENGTH, 1.0))
        fill = int(ratio * bar_w)
        bar = "█" * fill + "░" * (bar_w - fill)
        lines.append(f"  course x: {disp_x:6.2f} / {COURSE_LENGTH:5.2f} m   "
                     f"[{bar}] {100 * ratio:5.1f}%")
        # patch info
        if patch_idx == -1:
            patch_label = "LEAD-IN"
        elif patch_idx >= COURSE_NUM_PATCHES:
            patch_label = "TAIL-OUT"
        else:
            patch_label = f"{patch_idx:2d}  {PATCH_NAMES[patch_idx]}"
        lines.append(f"  patch   : {patch_label}")
        # cmd values
        vx, vy, wz = cmd
        paused_tag = "  \033[91m[PAUSED]\033[0m" if paused else ""
        lines.append(f"  cmd     : vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}{paused_tag}")
        # last termination
        if last_term:
            lines.append(f"  \033[93mlast term: {last_term}\033[0m")
        lines.append("=" * 53)

        if self._last_lines > 0:
            sys.stdout.write(f"\r\033[{self._last_lines}A\033[J")
            sys.stdout.flush()
        print("\n".join(lines))
        self._last_lines = len(lines)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    task = TASK_PER_LEVEL[args.level]
    print(f"[play_course] level={args.level}  task={task}  num_envs={args.num_envs}")

    env_cfg = parse_env_cfg(task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # Disable visual debug for the velocity command arrows — purely cosmetic.
    if hasattr(env_cfg.commands.base_velocity, "debug_vis"):
        env_cfg.commands.base_velocity.debug_vis = False

    # ----- 5.1 input devices ------------------------------------------------
    kb_ext = None
    se2_kb = None
    pad = None
    cmd_state = {"vx": 1.0, "vy": 0.0, "wz": 0.0,
                 "user_active": False,  # true when user is pushing keys / stick
                 "paused": False, "reset_req": False}

    if args.keyboard:
        kb_cfg = Se2KeyboardCfg(
            v_x_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            v_y_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            omega_z_sensitivity=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
        )
        se2_kb = Se2Keyboard(kb_cfg)
    # KeyboardExtension is for "extras" keys (D/G/+/-/P/R/C/L) that should be
    # available regardless of whether primary cmd input is keyboard or joystick.
    if _HAS_APPWINDOW:
        kb_ext = KeyboardExtension()
    else:
        print("[play_course] omni.appwindow unavailable — D/G/+/-/P/R/C/L extras disabled.")
    if args.joystick:
        pad = DirectLinuxGamepad(
            device_path="/dev/input/js0",
            x_scale=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            y_scale=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            w_scale=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
            deadzone=0.05,
        )

    # ----- 5.2 patch velocity_commands: default to env's auto command (with
    # heading_command yaw correction); only override when user is actively
    # pushing keys / stick. Matches eval_course.py semantics exactly except
    # that user input takes precedence when present.
    def manual_velocity_commands(env):
        if cmd_state["paused"]:
            return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        if cmd_state["user_active"]:
            cmds = torch.tensor(
                [cmd_state["vx"], cmd_state["vy"], cmd_state["wz"]],
                device=env.device, dtype=torch.float32,
            )
            return cmds.unsqueeze(0).repeat(env.num_envs, 1)
        # No user input — defer to env's command_manager (UniformVelocityCommand
        # with heading_command=True auto-corrects cmd_wz to keep yaw → +x).
        return env.command_manager.get_command("base_velocity")

    # Patch BOTH the policy obs group and the critic obs group (if present).
    n_patched = 0
    for grp_name in dir(env_cfg.observations):
        if grp_name.startswith("__"):
            continue
        grp = getattr(env_cfg.observations, grp_name)
        if grp is None or not hasattr(grp, "velocity_commands"):
            continue
        term = getattr(grp, "velocity_commands")
        if term is None:
            continue
        term.func = manual_velocity_commands
        term.params = {}
        n_patched += 1
    print(f"[play_course] patched velocity_commands obs in {n_patched} group(s)")

    # ----- 5.3 build env + wrap --------------------------------------------
    env_gym = gym.make(task, cfg=env_cfg)
    train_cfg_dict = build_train_cfg_dict(task, ablation=args.ablation)
    clip_actions = train_cfg_dict.get("clip_actions", True)
    env_wrapped = RslRlVecEnvWrapper(env_gym, clip_actions=clip_actions)
    base_env = get_truly_unwrapped_env(env_gym)

    # ----- 5.4 override env_origins (BEFORE first reset) -------------------
    x_start = override_env_origins(base_env, env_cfg, args.num_envs)

    # ----- 5.5 resolve + load ckpt -----------------------------------------
    experiment_name = train_cfg_dict.get("experiment_name", "split_moe_teacher_parallel")
    root_log_dir = os.path.join("logs", "moe_training", experiment_name)
    ckpt_path, run_dir = resolve_checkpoint_path(root_log_dir, args.load_run, args.checkpoint)
    print(f"[play_course] ckpt: {ckpt_path}")
    runner, model = load_split_moe_ckpt(env_wrapped, train_cfg_dict, ckpt_path, run_dir)

    print(f"[play_course] model loaded. "
          f"num_leg_experts={getattr(model, 'num_leg_experts', '?')} "
          f"num_wheel_experts={getattr(model, 'num_wheel_experts', '?')} "
          f"latent_dim={getattr(model, 'latent_dim', '?')}")

    obs, _ = env_wrapped.reset()
    robot = base_env.scene["robot"]
    spawn_xyz = robot.data.root_pos_w[0].cpu().numpy()
    print(f"[play_course] env_origin (env 0) = "
          f"{base_env.scene.env_origins[0].cpu().numpy().tolist()}")
    print(f"[play_course] robot spawn xyz   = {spawn_xyz.tolist()}")

    policy = runner.get_inference_policy(device=DEVICE)

    # ----- 5.6 camera state ------------------------------------------------
    camera_mode = 0  # 0 = forward chase, 1 = top-down, 2 = back-face
    camera_locked = True
    # Mutable container so _update_camera() (defined inside the main loop) can
    # adjust orbit state via stick input without `nonlocal` declarations.
    nonlocal_state = {"azim": 0.0, "elev": 0.45, "dist": 4.5, "free_eye": None}
    camera_history = []

    OFFSET_FORWARD = [-2.5, 0.0, 1.8]
    OFFSET_BACKWARD = [2.5, 0.0, 1.8]
    HEIGHT_TOP = 5.5

    # ----- 5.7 termination tracking ----------------------------------------
    last_term_label = ""
    term_term_names = []  # populated lazily
    try:
        term_mgr = base_env.termination_manager
        term_term_names = list(term_mgr.active_terms)
    except Exception:
        pass

    # ----- 5.8 main loop ---------------------------------------------------
    hud = CourseHUD()
    step = 0
    sim_dt = float(getattr(env_cfg, "sim", None).dt) if hasattr(env_cfg, "sim") else 0.005
    decimation = int(getattr(env_cfg, "decimation", 4))
    ctrl_dt = sim_dt * decimation

    # Keyboard-driven cmd state changes via Se2Keyboard
    kb_step_delta = 0.05  # m/s per frame held — used when reading Se2Keyboard
    smoke_mode = args.smoke_steps > 0
    smoke_remaining = args.smoke_steps

    # Fast-forward: number of physics steps per visual frame. Mutable so the
    # +/- keybindings can change it at runtime. Clamp to [1, 32].
    speed_mul = [int(args.speed)]
    print(f"[play_course] speed_mul = {speed_mul[0]}×  (toggle with + / -, max 32×)")

    # Track joystick / keyboard prev-press for edge-triggered events
    prev = {"c": False, "l": False, "r": False, "p": False,
            "Y": False, "A": False, "X": False, "lb": False, "rb": False}

    print("\n[play_course] starting inference loop. "
          f"smoke={'YES (' + str(args.smoke_steps) + ' steps)' if smoke_mode else 'no'}\n")

    with torch.inference_mode():
        while simulation_app.is_running():
            # ---- read inputs ------------------------------------------------
            ctrl_kind = "none"
            ctrl_connected = False

            # Default: no user input → defer to env's heading-corrected command.
            cmd_state["user_active"] = False

            if se2_kb is not None:
                ctrl_kind = "keyboard"
                ctrl_connected = True
                kb_cmd = se2_kb.advance()  # tensor[3] in (vx, vy, wz)
                if isinstance(kb_cmd, np.ndarray):
                    kb_cmd = torch.from_numpy(kb_cmd)
                kb_cmd = kb_cmd.float()
                if not torch.allclose(kb_cmd.cpu(), torch.zeros(3), atol=1e-3):
                    cmd_state["vx"] = float(kb_cmd[0].item())
                    cmd_state["vy"] = float(kb_cmd[1].item())
                    cmd_state["wz"] = float(kb_cmd[2].item())
                    cmd_state["user_active"] = True

            if pad is not None:
                ctrl_kind = "joystick"
                ctrl_connected = pad.connected
                if pad.connected:
                    pad_cmd = pad.advance()
                    if not torch.allclose(pad_cmd, torch.zeros(3), atol=1e-2):
                        cmd_state["vx"] = float(pad_cmd[0].item())
                        cmd_state["vy"] = float(pad_cmd[1].item())
                        cmd_state["wz"] = float(pad_cmd[2].item())
                        cmd_state["user_active"] = True

            # ---- edge-triggered actions (kb / pad) --------------------------
            c_curr = l_curr = r_curr = p_curr = False
            Y_curr = A_curr = X_curr = False
            if kb_ext is not None:
                c_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.C)
                l_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.L)
                r_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.R)
                p_curr = kb_ext.check_and_clear(carb.input.KeyboardInput.P)
                # Speed control: EQUAL / MINUS adjust speed_mul[0]
                if kb_ext.check_and_clear(carb.input.KeyboardInput.EQUAL):
                    speed_mul[0] = min(32, max(1, speed_mul[0] * 2))
                    print(f"[play_course] speed → {speed_mul[0]}×")
                if kb_ext.check_and_clear(carb.input.KeyboardInput.MINUS):
                    speed_mul[0] = max(1, speed_mul[0] // 2)
                    print(f"[play_course] speed → {speed_mul[0]}×")
                # GRU reset: G key
                if kb_ext.check_and_clear(carb.input.KeyboardInput.G):
                    reset_gru_state(model, env_wrapped.num_envs, DEVICE)
                    print(f"[play_course] GRU hidden state RESET to zero")
                # Diagnostic state dump: D key
                if kb_ext.check_and_clear(carb.input.KeyboardInput.D):
                    dump_robot_state(base_env, cmd_state, patch_idx)
            if pad is not None and pad.connected:
                Y_curr = pad.is_button_pressed(3)
                A_curr = pad.is_button_pressed(0)
                X_curr = pad.is_button_pressed(2)
                # GRU reset: LB (button 4)
                lb_curr = pad.is_button_pressed(4)
                if lb_curr and not prev["lb"]:
                    reset_gru_state(model, env_wrapped.num_envs, DEVICE)
                    print(f"[play_course] GRU hidden state RESET to zero (LB pressed)")
                prev["lb"] = lb_curr
                # State dump: RB (button 5)
                rb_curr = pad.is_button_pressed(5)
                if rb_curr and not prev["rb"]:
                    dump_robot_state(base_env, cmd_state, patch_idx)
                prev["rb"] = rb_curr

            if (c_curr and not prev["c"]) or (Y_curr and not prev["Y"]):
                camera_mode = (camera_mode + 1) % 3
                camera_history.clear()
            if (l_curr and not prev["l"]) or (A_curr and not prev["A"]):
                camera_locked = not camera_locked
                camera_history.clear()
            if (r_curr and not prev["r"]) or (X_curr and not prev["X"]):
                teleport_to_start(base_env, env_wrapped, x_start)
                last_term_label = "manual_reset"
            if (p_curr and not prev["p"]):
                cmd_state["paused"] = not cmd_state["paused"]
            prev.update({"c": c_curr, "l": l_curr, "r": r_curr, "p": p_curr,
                         "Y": Y_curr, "A": A_curr, "X": X_curr})

            # ---- policy + step (×N for fast-forward), camera-per-step ------
            # Camera follow runs every sim step (inside inner loop) so motion
            # stays smooth at speed > 1. HUD updates only once per outer iter.
            def _update_camera():
                root_pos = robot.data.root_pos_w[0]
                root_quat = robot.data.root_quat_w[0]
                dev = root_pos.device
                if not camera_locked:
                    if pad is not None and pad.connected:
                        ax_x = pad.axes.get(0, 0.0)
                        ax_y = pad.axes.get(1, 0.0)
                        ax_z = pad.axes.get(4, 0.0)
                        dz = 0.12
                        nonlocal_state["azim"] += ax_x * 0.035 if abs(ax_x) > dz else 0.0
                        nonlocal_state["elev"] += -ax_y * 0.020 if abs(ax_y) > dz else 0.0
                        nonlocal_state["dist"] += ax_z * 0.05 if abs(ax_z) > dz else 0.0
                        nonlocal_state["elev"] = max(0.05, min(nonlocal_state["elev"], 1.45))
                        nonlocal_state["dist"] = max(0.8, min(nonlocal_state["dist"], 14.0))
                    ce, se = np.cos(nonlocal_state["elev"]), np.sin(nonlocal_state["elev"])
                    ca, sa = np.cos(nonlocal_state["azim"]), np.sin(nonlocal_state["azim"])
                    d = nonlocal_state["dist"]
                    eye = root_pos + torch.tensor(
                        [d * ce * ca, d * ce * sa, d * se], device=dev, dtype=root_pos.dtype)
                    nonlocal_state["free_eye"] = eye.detach().clone()
                    if hasattr(base_env, "sim"):
                        base_env.sim.set_camera_view(eye.cpu().numpy(), root_pos.cpu().numpy())
                else:
                    eye_t, target_t = None, root_pos
                    if camera_mode == 0:
                        off = torch.tensor(OFFSET_FORWARD, device=dev)
                        eye_t = root_pos + math_utils.quat_apply(root_quat, off)
                    elif camera_mode == 1:
                        eye_t = root_pos + torch.tensor([0.0, 0.0, HEIGHT_TOP], device=dev)
                        target_t = root_pos + torch.tensor([0.001, 0.0, 0.0], device=dev)
                    elif camera_mode == 2:
                        off = torch.tensor(OFFSET_BACKWARD, device=dev)
                        eye_t = root_pos + math_utils.quat_apply(root_quat, off)
                    if eye_t is not None and hasattr(base_env, "sim"):
                        camera_history.append(eye_t)
                        if len(camera_history) > 30:
                            camera_history.pop(0)
                        smooth_eye = torch.stack(camera_history).mean(dim=0)
                        base_env.sim.set_camera_view(smooth_eye.cpu().numpy(),
                                                     target_t.cpu().numpy())

            sf = max(1, int(speed_mul[0]))
            dones = None
            for _sub in range(sf):
                actions = policy(obs)
                obs, _, dones, info = env_wrapped.step(actions)
                # Auto-reset GRU on env done (mirrors training-time rsl_rl
                # rollout). Without this the policy keeps stale hidden state
                # across implicit episode boundaries (Isaac Lab auto-resets
                # physics but doesn't touch the policy RNN).
                if dones.any() and hasattr(model, "reset"):
                    model.reset(dones.bool())
                _update_camera()  # per-step camera update (smooth at any speed)
                if bool(dones[0].item()):
                    break

            # detect terminations for env 0 (single-env diagnostic)
            if dones is not None and bool(dones[0].item()):
                # Try to pull the term-cause name
                try:
                    term_mgr = base_env.termination_manager
                    for tn in term_term_names:
                        if bool(term_mgr.get_term(tn)[0].item()):
                            last_term_label = tn
                            break
                except Exception:
                    last_term_label = "done"

            # ---- HUD --------------------------------------------------------
            disp_x = float(
                robot.data.root_pos_w[0, 0].item()
                - base_env.scene.env_origins[0, 0].item()
            )
            patch_idx = patch_index_for_x(disp_x)
            if step % max(1, args.hud_every) == 0:
                # HUD command: user override if active, else env's auto cmd
                if cmd_state["user_active"]:
                    hud_cmd = (cmd_state["vx"], cmd_state["vy"], cmd_state["wz"])
                else:
                    env_cmd = base_env.command_manager.get_command("base_velocity")[0]
                    hud_cmd = (float(env_cmd[0].item()),
                               float(env_cmd[1].item()),
                               float(env_cmd[2].item()))
                hud.render(
                    step=step,
                    sim_time=step * ctrl_dt,
                    disp_x=disp_x,
                    patch_idx=patch_idx,
                    cmd=hud_cmd,
                    paused=cmd_state["paused"],
                    controller_kind=ctrl_kind,
                    controller_connected=ctrl_connected,
                    last_term=last_term_label,
                )

            # camera follow already handled per-step inside the inner loop above

            step += sf
            if smoke_mode:
                smoke_remaining -= 1
                if smoke_remaining <= 0:
                    print("\n[play_course] smoke test OK — bootstrap clean, "
                          f"{step} policy steps executed, "
                          f"final disp_x={disp_x:.3f}m, patch={patch_idx}")
                    break

    print("[play_course] shutting down ...")
    env_wrapped.close()
    if pad is not None:
        pad.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
