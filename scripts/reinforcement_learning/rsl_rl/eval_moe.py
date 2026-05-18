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


def main():
    print(f"[eval_moe] task={args.task} num_envs={args.num_envs} num_steps={args.num_steps}")
    print(f"[eval_moe] success_dist={args.success_dist}m cmd_vx={args.cmd_vx}m/s")
    print("[eval_moe] Scaffold OK — implementation follows in later tasks.")


if __name__ == "__main__":
    main()
    simulation_app.close()
