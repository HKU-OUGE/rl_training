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


def main():
    if args.strict_per_terrain:
        raise NotImplementedError("--strict_per_terrain is reserved for v2 (loops 13 sub-terrains with sim restart).")
    print(f"[eval_moe] task={args.task} num_envs={args.num_envs} num_steps={args.num_steps}")
    print(f"[eval_moe] success_dist={args.success_dist}m cmd_vx={args.cmd_vx}m/s")

    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg = apply_eval_overrides(env_cfg, args)

    print("[eval_moe] env_cfg overrides applied.")


if __name__ == "__main__":
    main()
    simulation_app.close()
