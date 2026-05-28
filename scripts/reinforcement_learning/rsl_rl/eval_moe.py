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
parser.add_argument("--cmd_vx", type=float, default=1.0, help="Fixed +x velocity command (m/s); ignored if --cmd_vx_bidir")
parser.add_argument("--cmd_vx_bidir", action="store_true",
                    help="Sample cmd_vx uniformly from [-cmd_vx_max, -cmd_vx_min_abs] U [cmd_vx_min_abs, cmd_vx_max].")
parser.add_argument("--cmd_vx_min_abs", type=float, default=0.5,
                    help="Minimum |cmd_vx| when --cmd_vx_bidir (m/s).")
parser.add_argument("--cmd_vx_max", type=float, default=1.0,
                    help="Maximum |cmd_vx| when --cmd_vx_bidir (m/s).")
parser.add_argument("--load_run", type=str, default=None, help="Run dir name or absolute path; default = latest")
parser.add_argument("--checkpoint", type=str, default="model_*.pt", help="Checkpoint glob")
parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
parser.add_argument("--strict_per_terrain", action="store_true",
                    help="(unimplemented in v1; raises NotImplementedError) Restart sim per sub-terrain")
parser.add_argument("--zero_obs_noise", action="store_true", help="Disable obs-level AdditiveUniformNoiseCfg")
parser.add_argument("--keep_illegal_contact", action="store_true",
                    help="Keep illegal_contact as a hard termination. Default behavior is "
                         "to DISABLE it because (a) training uses PER_RANK_NO_ILLEGAL_CONTACT=1 "
                         "so the policy learned to tolerate transient base contact, and (b) real "
                         "robots are not killed by a single contact event. Set this flag to "
                         "reproduce pre-2026-05 eval numbers.")
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)
parser.add_argument("--latent_sample_envs", type=int, default=500, help="Subsample envs for GRU latent buffer")
parser.add_argument("--latent_sample_stride", type=int, default=10, help="Subsample stride for GRU latent")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ablation", type=str, default="full",
                    choices=["full", "A1", "A2", "A3", "B1", "B2"],
                    help="Ablation variant whose ckpt to eval. 'full' = baseline (default). "
                         "A1/B2 require architecture/runtime overrides; A2/A3/B1 are inference-"
                         "equivalent to baseline (only training-time differences). Output goes to "
                         "logs/moe_eval/split_moe_teacher_parallel_abl_{X}/ts_iter{N}/ when != full.")
parser.add_argument("--cap_pit_depth", type=float, default=0.72,
                    help="Cap pit_depth_range max at this value (m). Default 0.72 caps eval pit "
                         "difficulty at the baseline's L26-equivalent (above which even the full "
                         "policy fails). Set 0.8 to restore the training-time range; <=0 disables "
                         "the cap entirely.")

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
    if args.cmd_vx_bidir:
        # Set broad range so IsaacLab's built-in resample picks any sign;
        # we will overwrite with our exact distribution after env.reset().
        cmds.ranges.lin_vel_x = (-float(args.cmd_vx_max), float(args.cmd_vx_max))
    else:
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

    # ---- 6) Terrain: spread spawns across all difficulty levels ----
    # Keep terrain_generator.curriculum unchanged: it controls the *mesh layout* —
    # curriculum mode assigns each column one fixed sub-terrain (contiguous), which
    # is exactly what save_outputs()'s col_to_subterrain mapping assumes. Env-level
    # level promotion/demotion is already disabled via curriculum.terrain_levels =
    # None (step 5), so envs stay locked without disabling generator curriculum.
    tgen = env_cfg.scene.terrain.terrain_generator
    if tgen is not None:
        env_cfg.scene.terrain.max_init_terrain_level = int(tgen.num_rows) - 1
        print(f"[eval] terrain: {tgen.num_rows} rows × {tgen.num_cols} cols, "
              f"sub_terrains={list(tgen.sub_terrains.keys())}")

    # ---- 7) Optional: zero obs noise ----
    if args.zero_obs_noise:
        _zero_obs_noise(env_cfg.observations)
        print("[eval] obs noise zeroed")

    # ---- 8) illegal_contact termination: disabled by default (see flag help) ----
    if not args.keep_illegal_contact:
        if hasattr(env_cfg.terminations, "illegal_contact") and env_cfg.terminations.illegal_contact is not None:
            env_cfg.terminations.illegal_contact = None
            print("[eval] terminations.illegal_contact = None "
                  "(default; pass --keep_illegal_contact to restore)")

    # ---- 9) Cap pit_depth at deployment-realistic ceiling ----
    # pit at L27-29 (depths ~0.74-0.80 m) is physically out of reach for the M20
    # quad-leg+wheel platform — even the converged baseline fails 100% there. To
    # avoid drowning the per-row metric in unreachable rows, cap the parametric
    # max so the 30 eval rows span only the baseline's competent range.
    if args.cap_pit_depth > 0.0:
        tgen = env_cfg.scene.terrain.terrain_generator
        if tgen is not None and "pit" in tgen.sub_terrains:
            pit_cfg = tgen.sub_terrains["pit"]
            lo, hi = pit_cfg.pit_depth_range
            new_hi = float(args.cap_pit_depth)
            if hi > new_hi:
                pit_cfg.pit_depth_range = (lo, new_hi)
                print(f"[eval] pit_depth_range capped: ({lo}, {hi}) -> ({lo}, {new_hi}) "
                      f"(new L29 ≈ old L{int(round((new_hi-lo)/(hi-lo)*29))})")

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


def _reached_goal_signed(env, threshold: float) -> torch.Tensor:
    """Direction-aware reached_goal: success when |disp_x|>=threshold AND in cmd direction."""
    disp_x = env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    cmd_vx = env.command_manager.get_command("base_velocity")[:, 0]
    return ((cmd_vx > 0) & (disp_x >= threshold)) | \
           ((cmd_vx < 0) & (disp_x <= -threshold))


def inject_reached_goal_term(env_cfg, threshold: float, signed: bool = False):
    """Add reached_goal as a new termination term in env_cfg."""
    func = _reached_goal_signed if signed else _reached_goal_x
    env_cfg.terminations.reached_goal = DoneTerm(
        func=func,
        params={"threshold": float(threshold)},
    )
    print(f"[eval] terminations.reached_goal injected "
          f"(threshold={threshold}m, signed={signed})")


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


TERM_NAME_TO_ENUM = {
    "time_out": 0,
    "illegal_contact": 1,
    "terrain_out_of_bounds": 2,
    "bad_orientation": 3,
    "reached_goal": 4,
    # bad_orientation_2 is disabled in env_cfg but keep map just in case
    "bad_orientation_2": 3,
}


def handle_dones(t, dones, info, bufs, base_env):
    """For envs that just done for the first time, populate term_cause, term_step, reward_terms."""
    if not dones.any():
        return None  # no key discovery this step

    log = info.get("log", info)  # rsl_rl may wrap differently
    # dones is dtype=torch.long (0/1) from RslRlVecEnvWrapper; convert to bool so
    # that subsequent mask operations (indexing, &) behave correctly.
    fresh = dones.bool() & (~bufs["first_done"])
    if not fresh.any():
        return None
    fresh_idx = torch.where(fresh)[0]

    # ---- discover reward term names on first opportunity ----
    discovered = None
    if bufs["reward_terms"] is None:
        rew_keys = [k for k in log.keys() if k.startswith("Episode_Reward/")]
        rew_names = [k.replace("Episode_Reward/", "") for k in rew_keys]
        N = bufs["term_cause"].shape[0]
        bufs["reward_terms"] = torch.zeros(N, len(rew_names), dtype=torch.float32, device=DEVICE)
        bufs["_reward_term_names"] = rew_names
        discovered = ("reward_terms", rew_names)
        print(f"[eval] discovered {len(rew_names)} reward terms")

    # ---- per-env: find which termination fired ----
    # IsaacLab's TerminationManager.reset() writes Episode_Termination/* into
    # extras["log"] as an aggregated scalar (a count of resetting envs), NOT a
    # per-env tensor. Read per-env termination dones straight from the manager —
    # its _term_dones still hold this step's values until the next compute().
    tm = base_env.termination_manager
    if "_term_discovered" not in bufs:
        bufs["_term_discovered"] = True
        print(f"[eval] termination terms: {list(tm.active_terms)}")
    for name in tm.active_terms:
        enum_val = TERM_NAME_TO_ENUM.get(name, 127)
        try:
            term_done = tm.get_term(name)
        except Exception:
            term_done = getattr(tm, "_term_dones", {}).get(name)
        if term_done is None:
            continue
        hit = term_done.to(torch.bool) & fresh
        bufs["term_cause"][hit] = enum_val

    # default: any fresh env still with term_cause==-1 → time_out (episode_length hit)
    still_unset = fresh & (bufs["term_cause"] == -1)
    bufs["term_cause"][still_unset] = 0  # time_out

    # ---- term_step ----
    bufs["term_step"][fresh] = t

    # ---- reward_terms ----
    # IsaacLab's reward_manager.reset() returns per-key scalar means (averaged over
    # the env_ids that just reset), NOT per-env tensors.  So val is either a 0-d
    # tensor or a Python float.  Store that mean for every fresh env — it's the best
    # available signal when a batch resets together (e.g., all time_out at step T).
    rew_names = bufs.get("_reward_term_names", [])
    for col_i, name in enumerate(rew_names):
        key = f"Episode_Reward/{name}"
        val = log.get(key)
        if val is None:
            continue
        if isinstance(val, torch.Tensor) and val.ndim > 0 and val.shape[0] == bufs["term_cause"].shape[0]:
            # Per-env tensor (future-proof / custom wrappers that expose per-env sums)
            bufs["reward_terms"][fresh_idx, col_i] = val[fresh_idx].to(torch.float32)
        else:
            # Scalar mean over the resetting batch — broadcast to all fresh envs
            scalar = float(val.item() if isinstance(val, torch.Tensor) else val)
            bufs["reward_terms"][fresh_idx, col_i] = scalar

    bufs["first_done"][fresh] = True
    return discovered


def save_outputs(bufs, env_cfg, args, ckpt_path, experiment_name, iter_num):
    """Write raw.npz and summary.json to logs/moe_eval/<exp>/<ts>_iter<N>/"""
    # Output dir
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join("logs", "moe_eval", experiment_name, f"{ts}_iter{iter_num}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    # Sub-terrain name list from terrain_generator
    tgen = env_cfg.scene.terrain.terrain_generator
    sub_terrain_names = list(tgen.sub_terrains.keys())  # 13 unique, in declaration order

    # 18-col → sub-terrain name mapping using cumulative allocation.
    # Last sub-terrain gets remainder so total always equals num_cols (no overshoot/undershoot).
    proportions = [(name, c.proportion) for name, c in tgen.sub_terrains.items()]
    total = sum(p for _, p in proportions) or 1.0
    col_to_subterrain = []
    for i, (name, p) in enumerate(proportions):
        if i == len(proportions) - 1:
            n_cols = tgen.num_cols - len(col_to_subterrain)
        else:
            n_cols = max(1, round(p / total * tgen.num_cols))
        col_to_subterrain.extend([name] * n_cols)
    # Safety clamp (shouldn't trigger after the cumulative logic but cheap insurance)
    col_to_subterrain = col_to_subterrain[:tgen.num_cols]
    while len(col_to_subterrain) < tgen.num_cols:
        col_to_subterrain.append(proportions[-1][0])

    # --- raw.npz ---
    np_buf = {}
    for k, v in bufs.items():
        if isinstance(v, torch.Tensor):
            np_buf[k] = v.cpu().numpy()
        elif k in ("t_stride",):
            np_buf[k] = np.int32(v)
    raw_path = os.path.join(out_dir, "raw.npz")
    np.savez_compressed(raw_path, **np_buf)
    print(f"[eval] raw.npz written ({os.path.getsize(raw_path)/1e6:.1f} MB) → {raw_path}")

    # --- summary.json ---
    summary = {
        "task": args.task,
        "experiment_name": experiment_name,
        "ckpt_path": ckpt_path,
        "iter": iter_num,
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "success_dist": float(args.success_dist),
        "cmd_vx": float(args.cmd_vx),
        "seed": int(args.seed),
        "sub_terrain_names": sub_terrain_names,
        "col_to_subterrain": col_to_subterrain,
        "num_rows": int(tgen.num_rows),
        "num_cols": int(tgen.num_cols),
        "term_enum": {v: k for k, v in TERM_NAME_TO_ENUM.items()},
        "reward_term_names": bufs.get("_reward_term_names", []),
        "model": {
            "num_leg_experts": int(bufs["gate_leg"].shape[-1]),
            "num_wheel_experts": int(bufs["gate_wheel"].shape[-1]),
            "latent_dim": int(bufs["gru_latent_sample"].shape[-1]),
        },
        "tstamp": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] summary.json written → {out_dir}/summary.json")

    return out_dir


def main():
    if args.strict_per_terrain:
        raise NotImplementedError("--strict_per_terrain is reserved for v2 (loops 13 sub-terrains with sim restart).")
    print(f"[eval_moe] task={args.task} num_envs={args.num_envs} num_steps={args.num_steps}")
    print(f"[eval_moe] success_dist={args.success_dist}m cmd_vx={args.cmd_vx}m/s")

    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg = apply_eval_overrides(env_cfg, args)
    inject_reached_goal_term(env_cfg, args.success_dist, signed=args.cmd_vx_bidir)

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

    # === Ablation cfg overrides (matches eval_train.py ABLATIONS registry) ===
    # Only flags that change the constructed network or its forward path matter at
    # eval-time. A2/A3/B1 are training-only diffs (critic placement / loss terms)
    # → leaving them inference-equivalent to full is intentional.
    _ABL_POLICY_OVERRIDES = {
        "A1": {"single_gate": True},          # different architecture
        "B2": {"blind_vision": True},          # zero exteroception at fwd
        # A2/A3/B1: no policy override needed
    }
    if args.ablation != "full":
        for k, v in _ABL_POLICY_OVERRIDES.get(args.ablation, {}).items():
            train_cfg_dict["policy"][k] = v
        # Route output + ckpt resolution to the ablation's experiment_name
        train_cfg_dict["experiment_name"] = f"split_moe_teacher_parallel_abl_{args.ablation}"
        print(f"[eval][ablation={args.ablation}] policy overrides: "
              f"{_ABL_POLICY_OVERRIDES.get(args.ablation, {})}")

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

    # Bidirectional cmd_vx override: sample sign*[cmd_vx_min_abs, cmd_vx_max]
    # uniformly per env. Episode-stable (resampling_time_range >> episode len).
    if args.cmd_vx_bidir:
        base_env_tmp = env_wrapped.unwrapped
        while hasattr(base_env_tmp, "env"):
            base_env_tmp = base_env_tmp.env
        if hasattr(base_env_tmp, "unwrapped"):
            base_env_tmp = base_env_tmp.unwrapped
        vel_cmd = base_env_tmp.command_manager.get_command("base_velocity")
        gen = torch.Generator(device=vel_cmd.device).manual_seed(args.seed)
        sign = torch.where(torch.rand(N, generator=gen, device=vel_cmd.device) < 0.5,
                           -1.0, 1.0)
        mag = (torch.rand(N, generator=gen, device=vel_cmd.device)
               * (args.cmd_vx_max - args.cmd_vx_min_abs)) + args.cmd_vx_min_abs
        vel_cmd[:, 0] = sign * mag
        vel_cmd[:, 1] = 0.0
        vel_cmd[:, 2] = 0.0
        n_fwd = int((vel_cmd[:, 0] > 0).sum())
        n_bwd = int((vel_cmd[:, 0] < 0).sum())
        print(f"[eval] cmd_vx_bidir override: {n_fwd} forward, {n_bwd} backward "
              f"(|vx| in [{args.cmd_vx_min_abs}, {args.cmd_vx_max}])")
        # obs was produced by reset() before this override, so it still embeds the
        # reset-time random command. Refresh it so step 0's policy(obs) — and the
        # recorded bufs["cmd"][:,0] — both reflect the bidir command.
        obs = env_wrapped.get_observations()

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

    # =========================================================================
    # Collection loop
    # =========================================================================
    policy = runner.get_inference_policy(device=DEVICE)
    robot = base_env.scene["robot"]
    cmd_mgr = base_env.command_manager
    t_stride = bufs["t_stride"]
    latent_env_idx = bufs["latent_env_idx"]

    print(f"[eval] starting rollout: N={N} T={T} ...")
    with torch.inference_mode():
        for t in range(T):
            # Step policy
            actions = policy(obs)

            # Hook outputs reflect the most-recent forward pass (already populated)
            leg_logits = sink["leg_logits"]    # (N, nL)
            wheel_logits = sink["wheel_logits"]  # (N, nW)
            rnn_out = sink["rnn_out"]            # (N, L)
            gate_leg = torch.softmax(leg_logits, dim=-1)
            gate_wheel = torch.softmax(wheel_logits, dim=-1)

            # Record per-step (only for envs not yet first_done)
            active = ~bufs["first_done"]
            bufs["root_pos_xy"][active, t] = robot.data.root_pos_w[active, :2]
            bufs["cmd"][active, t] = cmd_mgr.get_command("base_velocity")[active]
            actual = torch.cat([robot.data.root_lin_vel_b[:, :2],
                                robot.data.root_ang_vel_b[:, 2:3]], dim=-1)
            bufs["actual_vel"][active, t] = actual[active]
            bufs["gate_leg"][active, t] = gate_leg[active].to(torch.float16)
            bufs["gate_wheel"][active, t] = gate_wheel[active].to(torch.float16)

            # Latent subsample
            if t % t_stride == 0:
                t_l = t // t_stride
                bufs["gru_latent_sample"][:, t_l] = rnn_out[latent_env_idx].to(torch.float16)

            # Step env
            obs, _, dones, info = env_wrapped.step(actions)

            # Event handling
            handle_dones(t, dones, info, bufs, base_env)

            if (t + 1) % 50 == 0:
                done_frac = bufs["first_done"].float().mean().item()
                print(f"[eval] step {t + 1}/{T}  first_done = {100*done_frac:.1f}%")

            if bufs["first_done"].all():
                print(f"[eval] all envs done at step {t + 1} — early exit")
                break

        # Drain: 2 extra env.step() calls past the loop so IsaacLab fires the
        # pending time_out termination at episode_length_buf == max. Without this,
        # ~all envs hit the safety pass (term_cause=time_out but reward_terms=0)
        # because the boundary step is never observed. handle_dones picks up
        # Episode_Reward/* on the natural done and fills reward_terms properly.
        # Per-step buffers (root_pos_xy, gate_*, etc.) are NOT recorded for drain
        # steps — only termination/reward events are captured.
        n_drain = 2
        pre_drain_done = bufs["first_done"].sum().item()
        for d in range(n_drain):
            if bufs["first_done"].all():
                break
            actions = policy(obs)
            obs, _, dones, info = env_wrapped.step(actions)
            handle_dones(T + d, dones, info, bufs, base_env)
        post_drain_done = bufs["first_done"].sum().item()
        print(f"[eval] drain steps captured {post_drain_done - pre_drain_done} extra dones")

    # Safety: envs that STILL never died after drain → mark as time_out (no reward)
    never_done = ~bufs["first_done"]
    if never_done.any():
        bufs["term_cause"][never_done] = 0  # time_out
        bufs["term_step"][never_done] = T - 1
        print(f"[eval] {never_done.sum().item()} envs never terminated even after drain — marked time_out (no reward data)")

    print(f"[eval] rollout complete.")
    out_dir = save_outputs(bufs, env_cfg, args, ckpt_path, experiment_name, iter_num)
    print(f"[eval] ALL DONE. results at {out_dir}")
    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
