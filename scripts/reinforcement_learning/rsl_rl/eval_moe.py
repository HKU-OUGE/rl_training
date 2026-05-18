# scripts/reinforcement_learning/rsl_rl/eval_moe.py
"""Evaluate a SplitMoEActorCritic policy on Rough-MoE-Teacher and dump raw stats.

Design doc: docs/superpowers/specs/2026-05-18-moe-eval-script-design.md
Plan:       docs/superpowers/plans/2026-05-18-moe-eval-script.md
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate MoE Teacher policy and dump raw stats.")
parser.add_argument("--task", type=str, default="Rough-MoE-Teacher-Deeprobotics-M20-v0")
parser.add_argument("--num_envs", type=int, default=2000)
parser.add_argument("--num_steps", type=int, default=500, help="Control steps (= 10s at dt=0.02)")
parser.add_argument("--success_dist", type=float, default=4.0, help="+x displacement threshold (m) for reached_goal")
parser.add_argument("--cmd_vx", type=float, default=1.0, help="Fixed +x velocity command (m/s)")
parser.add_argument("--load_run", type=str, default=None, help="Run dir name or absolute path; default = latest")
parser.add_argument("--checkpoint", type=str, default="model_*.pt", help="Checkpoint glob")
parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
parser.add_argument("--strict_per_terrain", action="store_true",
                    help="(unimplemented in v1; raises NotImplementedError) Restart sim per sub-terrain")
parser.add_argument("--zero_obs_noise", action="store_true", help="Disable obs-level AdditiveUniformNoiseCfg")
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)
parser.add_argument("--latent_sample_envs", type=int, default=500, help="Subsample envs for GRU latent buffer")
parser.add_argument("--latent_sample_stride", type=int, default=10, help="Subsample stride for GRU latent")
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True  # eval is always headless
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==============================================================================
# Heavy imports (AFTER AppLauncher)
# ==============================================================================
import numpy as np
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.managers import TerminationTermCfg as DoneTerm

# Inject SplitMoE classes into rsl_rl namespace
sys.path.append(os.getcwd())
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import (
    SplitMoEActorCritic,
    SplitMoEPPO,
)
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module
rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEPPO = SplitMoEPPO

DEVICE = "cuda:0"


def apply_eval_overrides(env_cfg, args):
    """In-place edit env_cfg to enforce the eval protocol (spec §2.2.1).

    - heading=0, lin_vel_x=cmd_vx fixed, no resampling
    - episode_length 10s
    - disable all domain randomization events
    - reset pose/velocity ranges all zero
    - disable terrain + command curricula
    - lock terrain to MOE_ROUGH_TERRAINS_CFG with curriculum=False, max_init_terrain_level=num_rows-1
    """
    # ---- 1) Commands ----
    cmds = env_cfg.commands.base_velocity
    cmds.heading_command = True
    cmds.rel_heading_envs = 1.0
    cmds.rel_standing_envs = 0.0
    cmds.resampling_time_range = (100.0, 100.0)
    cmds.ranges.heading = (0.0, 0.0)
    cmds.ranges.lin_vel_x = (float(args.cmd_vx), float(args.cmd_vx))
    cmds.ranges.lin_vel_y = (0.0, 0.0)
    cmds.ranges.ang_vel_z = (0.0, 0.0)
    cmds.debug_vis = False

    # ---- 2) Episode length ----
    env_cfg.episode_length_s = float(args.num_steps) * 0.02  # dt = 0.005 * decimation 4

    # ---- 3) Events: disable all domain randomization ----
    events_to_disable = [
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_actuator_gains",
        "randomize_push_robot",
    ]
    for name in events_to_disable:
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
            print(f"[eval] events.{name} = None")

    # ---- 4) Reset geometry: all zero ----
    if hasattr(env_cfg.events, "randomize_reset_base") and env_cfg.events.randomize_reset_base is not None:
        env_cfg.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        env_cfg.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

    # ---- 5) Curricula off ----
    if hasattr(env_cfg, "curriculum"):
        for name in ["terrain_levels", "command_levels_lin_vel", "command_levels_ang_vel"]:
            if hasattr(env_cfg.curriculum, name):
                setattr(env_cfg.curriculum, name, None)

    # ---- 6) Terrain: spread across all levels, no curriculum ----
    tgen = env_cfg.scene.terrain.terrain_generator
    if tgen is not None:
        tgen.curriculum = False
        env_cfg.scene.terrain.max_init_terrain_level = int(tgen.num_rows) - 1
        print(f"[eval] terrain: {tgen.num_rows} rows × {tgen.num_cols} cols, "
              f"sub_terrains={list(tgen.sub_terrains.keys())}")

    # ---- 7) Optional: zero obs noise ----
    if args.zero_obs_noise:
        _zero_obs_noise(env_cfg.observations)
        print("[eval] obs noise zeroed")

    return env_cfg


def _zero_obs_noise(observations):
    """Recursively set noise=None on all ObsTerm in all groups."""
    for group_name in dir(observations):
        if group_name.startswith("_"):
            continue
        group = getattr(observations, group_name)
        if group is None or not hasattr(group, "__dict__"):
            continue
        for term_name in dir(group):
            if term_name.startswith("_"):
                continue
            term = getattr(group, term_name)
            if term is not None and hasattr(term, "noise"):
                term.noise = None


def _reached_goal_x(env, threshold: float) -> torch.Tensor:
    """Termination: +x displacement from env spawn anchor exceeds threshold."""
    disp_x = env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return disp_x >= threshold


def inject_reached_goal_term(env_cfg, threshold: float):
    """Add reached_goal as a new termination term in env_cfg."""
    env_cfg.terminations.reached_goal = DoneTerm(
        func=_reached_goal_x,
        params={"threshold": float(threshold)},
    )
    print(f"[eval] terminations.reached_goal injected (threshold={threshold}m)")


def resolve_checkpoint(experiment_name: str, load_run, ckpt_glob: str):
    """Return (ckpt_path, run_dir). Default = latest run, latest model_*.pt."""
    root = os.path.join("logs", "moe_training", experiment_name)
    if not os.path.exists(root):
        raise FileNotFoundError(f"Experiment log dir not found: {root}")

    if load_run is None:
        runs = [os.path.join(root, d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]
        if not runs:
            raise FileNotFoundError(f"No runs in {root}")
        runs.sort(key=os.path.getmtime)
        run_dir = runs[-1]
        print(f"[eval] auto-selected run: {os.path.basename(run_dir)}")
    elif os.path.isabs(load_run):
        run_dir = load_run
    else:
        run_dir = os.path.join(root, load_run)

    files = sorted(glob.glob(os.path.join(run_dir, ckpt_glob)),
                   key=lambda p: _iter_num(p))
    if not files:
        raise FileNotFoundError(f"No ckpts matching {ckpt_glob} in {run_dir}")
    return files[-1], run_dir


def _iter_num(path: str) -> int:
    """Extract iter number from model_N.pt name."""
    name = os.path.basename(path).replace("model_", "").replace(".pt", "")
    try:
        return int(name)
    except ValueError:
        return -1


def build_and_load_runner(env_wrapped, train_cfg_dict, ckpt_path):
    """Build OnPolicyRunner and load weights. Returns (runner, model_instance)."""
    runner = OnPolicyRunner(env_wrapped, train_cfg_dict, log_dir=None, device=DEVICE)

    loaded = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = loaded.get("model_state_dict", loaded)

    if any(k.startswith("student.") for k in state_dict.keys()):
        print("[eval] distilled ckpt detected — stripping 'student.' prefix")
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("student.") and "critic" not in k:
                new_sd[k.replace("student.", "", 1)] = v
        runner.alg.policy.load_state_dict(new_sd, strict=False)
    else:
        runner.load(ckpt_path)

    model = runner.alg.policy
    model.eval()
    return runner, model


def install_hooks(model):
    """Register forward hooks on leg_gate, wheel_gate, and the RNN.

    Returns a dict that gets overwritten in-place each step:
      {"leg_logits": Tensor[B, num_leg],
       "wheel_logits": Tensor[B, num_wheel],
       "rnn_out": Tensor[B, latent_dim]}
    """
    sink = {}

    def make_hook(key):
        def _h(module, inp, out):
            # gate output: (B, num_expert) or (T, B, num_expert); take last frame if 3D
            t = out
            if t.ndim == 3:
                t = t[-1]
            sink[key] = t.detach()
        return _h

    def rnn_hook(module, inp, out):
        # GRU returns (rnn_out, hidden); rnn_out is (T, B, latent)
        rnn_out = out[0] if isinstance(out, tuple) else out
        if rnn_out.ndim == 3:
            rnn_out = rnn_out[-1]
        sink["rnn_out"] = rnn_out.detach()

    model.leg_gate.register_forward_hook(make_hook("leg_logits"))
    model.wheel_gate.register_forward_hook(make_hook("wheel_logits"))
    model.rnn.register_forward_hook(rnn_hook)
    return sink


def allocate_buffers(N, T, model, args):
    """Return dict of GPU buffers (zeros)."""
    nL = model.num_leg_experts
    nW = model.num_wheel_experts
    L = model.latent_dim

    # latent subsample: pick min(args.latent_sample_envs, N) envs evenly; every args.latent_sample_stride steps
    n_latent = min(args.latent_sample_envs, N)
    latent_env_idx = torch.linspace(0, N - 1, n_latent, device=DEVICE).long()
    t_stride = max(1, args.latent_sample_stride)
    T_latent = (T + t_stride - 1) // t_stride

    return {
        # constants per env (filled after reset)
        "terrain_types": torch.zeros(N, dtype=torch.int32, device=DEVICE),
        "terrain_levels": torch.zeros(N, dtype=torch.int32, device=DEVICE),
        # episode tracking
        "term_cause": torch.full((N,), -1, dtype=torch.int8, device=DEVICE),
        "term_step": torch.full((N,), -1, dtype=torch.int32, device=DEVICE),
        "first_done": torch.zeros(N, dtype=torch.bool, device=DEVICE),
        # per-step traces
        "root_pos_xy": torch.zeros(N, T, 2, dtype=torch.float32, device=DEVICE),
        "cmd": torch.zeros(N, T, 3, dtype=torch.float32, device=DEVICE),
        "actual_vel": torch.zeros(N, T, 3, dtype=torch.float32, device=DEVICE),
        "gate_leg": torch.zeros(N, T, nL, dtype=torch.float16, device=DEVICE),
        "gate_wheel": torch.zeros(N, T, nW, dtype=torch.float16, device=DEVICE),
        # latent subsample buffer
        "gru_latent_sample": torch.zeros(n_latent, T_latent, L, dtype=torch.float16, device=DEVICE),
        "latent_env_idx": latent_env_idx,
        "t_stride": t_stride,
        # reward_terms: filled after key discovery (Task 7)
        "reward_terms": None,
    }


def main():
    if args.strict_per_terrain:
        raise NotImplementedError("--strict_per_terrain is reserved for v2 (loops 13 sub-terrains with sim restart).")
    print(f"[eval_moe] task={args.task} num_envs={args.num_envs} num_steps={args.num_steps}")
    print(f"[eval_moe] success_dist={args.success_dist}m cmd_vx={args.cmd_vx}m/s")

    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg = apply_eval_overrides(env_cfg, args)
    inject_reached_goal_term(env_cfg, args.success_dist)

    # --- train cfg ---
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
    train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
    train_cfg_dict["device"] = DEVICE
    train_cfg_dict["logger"] = "tensorboard"  # never wandb in eval
    if args.num_wheel_experts is not None:
        train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts is not None:
        train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
        train_cfg_dict["policy"].pop(k, None)
    experiment_name = train_cfg_dict.get("experiment_name", "split_moe_teacher_parallel")

    # --- checkpoint ---
    ckpt_path, run_dir = resolve_checkpoint(experiment_name, args.load_run, args.checkpoint)
    print(f"[eval] using ckpt: {ckpt_path}")
    iter_num = _iter_num(ckpt_path)

    # --- env + runner ---
    env_gym = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env_gym, clip_actions=train_cfg_dict.get("clip_actions", True))
    runner, model = build_and_load_runner(env_wrapped, train_cfg_dict, ckpt_path)
    print(f"[eval] model loaded. num_leg_experts={model.num_leg_experts} "
          f"num_wheel_experts={model.num_wheel_experts} latent_dim={model.latent_dim}")

    sink = install_hooks(model)
    obs, _ = env_wrapped.reset()

    N = env_wrapped.num_envs
    T = args.num_steps
    bufs = allocate_buffers(N, T, model, args)

    # Capture terrain constants
    base_env = env_wrapped.unwrapped
    while hasattr(base_env, "env"):
        base_env = base_env.env
    if hasattr(base_env, "unwrapped"):
        base_env = base_env.unwrapped
    bufs["terrain_types"].copy_(base_env.scene.terrain.terrain_types.to(torch.int32))
    bufs["terrain_levels"].copy_(base_env.scene.terrain.terrain_levels.to(torch.int32))

    print(f"[eval] buffers allocated. N={N} T={T} latent_sample shape={tuple(bufs['gru_latent_sample'].shape)}")
    print(f"[eval] terrain_types unique = {torch.unique(bufs['terrain_types']).cpu().tolist()}")

    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
