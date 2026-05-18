# MoE Teacher Evaluation Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-script evaluation pipeline for the `Rough-MoE-Teacher-Deeprobotics-M20-v0` policy that produces 9 plots (success heatmap, expert activation, velocity tracking, termination/reward breakdown, plus 5 MoE-specific charts).

**Architecture:** `eval_moe.py` runs one IsaacLab sim pass over `MOE_ROUGH_TERRAINS_CFG` (30 levels × 18 cols, locked, no curriculum) with the policy, collects per-step buffers into `raw.npz`. `plot_moe_eval.py` reads `raw.npz` offline and renders 9 PNGs. Eval protocol: heading=0, x-vel=1.0 m/s, episode 10s, terminate at +x displacement ≥ 4m, all domain-randomization disabled.

**Tech Stack:** Python 3.11, PyTorch, IsaacLab, rsl_rl, gymnasium, matplotlib, numpy, scikit-learn (TSNE only).

**Reference spec:** `docs/superpowers/specs/2026-05-18-moe-eval-script-design.md`

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `scripts/reinforcement_learning/rsl_rl/eval_moe.py` | Sim launcher + data collector (creates `raw.npz` + `summary.json`) |
| `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py` | Pure CPU script: reads `raw.npz` → writes 9 PNGs |
| `logs/moe_eval/<experiment_name>/<timestamp>_iter<N>/raw.npz` | Output (runtime-generated) |
| `logs/moe_eval/<experiment_name>/<timestamp>_iter<N>/summary.json` | Metadata + enum maps |
| `logs/moe_eval/<experiment_name>/<timestamp>_iter<N>/plots/*.png` | 9 final plots |

**Testing strategy:** IsaacLab tasks cannot be unit-tested without a GPU + USD scene. Verification = smoke run (200 envs × 100 steps, ~30s) after Task 10, then full run (2000 envs × 500 steps, ~10 min) after Task 21. Plotting code is verified visually.

---

## Task 1: Scaffold `eval_moe.py` with AppLauncher and arg parser

**Files:**
- Create: `scripts/reinforcement_learning/rsl_rl/eval_moe.py`

- [ ] **Step 1: Write the file with argparse and AppLauncher boot (must be the very first imports — IsaacLab requires it)**

```python
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
```

- [ ] **Step 2: Smoke-run the scaffold to verify AppLauncher boots cleanly**

Run:
```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 4 --num_steps 1
```
Expected: Prints `[eval_moe] Scaffold OK` and exits cleanly. IsaacLab boot logs are OK.

- [ ] **Step 3: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/eval_moe.py
git commit -m "eval(moe): scaffold script with AppLauncher and arg parser"
```

---

## Task 2: Implement `apply_eval_overrides()` for env_cfg

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/eval_moe.py` (add function above `main()`)

- [ ] **Step 1: Add the override helper function**

Insert this function immediately above `def main():` in `eval_moe.py`:

```python
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
```

- [ ] **Step 2: Wire into main(), parse env_cfg and call apply_eval_overrides**

Replace the body of `main()` with:

```python
def main():
    if args.strict_per_terrain:
        raise NotImplementedError("--strict_per_terrain is reserved for v2 (loops 13 sub-terrains with sim restart).")
    print(f"[eval_moe] task={args.task} num_envs={args.num_envs} num_steps={args.num_steps}")
    print(f"[eval_moe] success_dist={args.success_dist}m cmd_vx={args.cmd_vx}m/s")

    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg = apply_eval_overrides(env_cfg, args)

    print("[eval_moe] env_cfg overrides applied.")
```

- [ ] **Step 3: Smoke-run, verify overrides print without crash**

Run:
```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 16 --num_steps 50
```
Expected: Logs show `events.randomize_push_robot = None` (×8 disable lines), terrain info line, no crash.

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/eval_moe.py
git commit -m "eval(moe): apply_eval_overrides — heading=0, DR off, ep=10s"
```

---

## Task 3: Inject `reached_goal` termination + create env

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/eval_moe.py`

- [ ] **Step 1: Add the termination function and injector above `main()`**

```python
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
```

- [ ] **Step 2: Wire into main() — call injector and create env**

Extend `main()`:

```python
def main():
    print(f"[eval_moe] task={args.task} num_envs={args.num_envs} num_steps={args.num_steps}")
    print(f"[eval_moe] success_dist={args.success_dist}m cmd_vx={args.cmd_vx}m/s")

    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg = apply_eval_overrides(env_cfg, args)
    inject_reached_goal_term(env_cfg, args.success_dist)

    env = gym.make(args.task, cfg=env_cfg)
    print(f"[eval_moe] env created. num_envs={env.unwrapped.num_envs}")
    env.close()
```

- [ ] **Step 3: Smoke-run, verify env is created and closes cleanly**

Run:
```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 16 --num_steps 50
```
Expected: `[eval] terminations.reached_goal injected (threshold=4.0m)` and `env created. num_envs=16`. IsaacLab may take ~30s to build the scene.

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/eval_moe.py
git commit -m "eval(moe): inject reached_goal termination + create env"
```

---

## Task 4: Auto-resolve checkpoint + load model

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/eval_moe.py`

- [ ] **Step 1: Add checkpoint resolver (mirrors `play_moe.py:resolve_checkpoint_path`)**

```python
def resolve_checkpoint(experiment_name: str, load_run: str | None, ckpt_glob: str):
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
```

- [ ] **Step 2: Add model loader (handles distilled `student.*` prefix)**

```python
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
```

- [ ] **Step 3: Wire into main() — load train_cfg, resolve ckpt, wrap env, build runner**

Replace `main()` body:

```python
def main():
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

    env_wrapped.close()
```

- [ ] **Step 4: Smoke-run, verify ckpt resolution and model load**

Run:
```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 16 --num_steps 50
```
Expected: `[eval] auto-selected run: 2026-05-15_10-21-22` and `[eval] using ckpt: ...model_19999.pt` and `[eval] model loaded. num_leg_experts=6 num_wheel_experts=3 latent_dim=256`.

- [ ] **Step 5: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/eval_moe.py
git commit -m "eval(moe): auto-resolve checkpoint and load model"
```

---

## Task 5: Forward hooks for gate logits + GRU latent

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/eval_moe.py`

- [ ] **Step 1: Add hook installer**

```python
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
```

- [ ] **Step 2: Wire into main() — install hooks, do a single inference step, dump shapes for verification**

Append to `main()` (before the final `env_wrapped.close()`):

```python
    sink = install_hooks(model)
    obs, _ = env_wrapped.reset()
    policy = runner.get_inference_policy(device=DEVICE)
    with torch.inference_mode():
        _ = policy(obs)
    print(f"[eval] hook test: "
          f"leg_logits={tuple(sink['leg_logits'].shape)} "
          f"wheel_logits={tuple(sink['wheel_logits'].shape)} "
          f"rnn_out={tuple(sink['rnn_out'].shape)}")
```

- [ ] **Step 3: Smoke-run, verify hook shapes**

Run:
```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 16 --num_steps 50
```
Expected: `[eval] hook test: leg_logits=(16, 6) wheel_logits=(16, 3) rnn_out=(16, 256)`

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/eval_moe.py
git commit -m "eval(moe): forward hooks on leg/wheel gates and GRU"
```

---

## Task 6: Pre-allocate per-step GPU buffers

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/eval_moe.py`

- [ ] **Step 1: Add `allocate_buffers()` helper**

```python
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
```

- [ ] **Step 2: Wire into main() — allocate after model load, log shapes**

Replace the post-hook lines with:

```python
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
```

- [ ] **Step 3: Smoke-run, verify buffer shapes and terrain_types**

Run:
```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 16 --num_steps 50
```
Expected: `latent_sample shape=(16, 5, 256)` and `terrain_types unique = [0, 1, ..., 17]` (depends on N — with 16 envs may not see all 18 cols).

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/eval_moe.py
git commit -m "eval(moe): pre-allocate per-step buffers + grab terrain types"
```

---

## Task 7: First-done event handling + key discovery

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/eval_moe.py`

- [ ] **Step 1: Add the event handler**

```python
TERM_NAME_TO_ENUM = {
    "time_out": 0,
    "illegal_contact": 1,
    "terrain_out_of_bounds": 2,
    "bad_orientation": 3,
    "reached_goal": 4,
    # bad_orientation_2 is disabled in env_cfg but keep map just in case
    "bad_orientation_2": 3,
}


def handle_dones(t, dones, info, bufs):
    """For envs that just done for the first time, populate term_cause, term_step, reward_terms."""
    if not dones.any():
        return None  # no key discovery this step

    log = info.get("log", info)  # rsl_rl may wrap differently
    fresh = dones & (~bufs["first_done"])
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

    # ---- discover termination names on first opportunity ----
    if "_term_keys" not in bufs:
        term_keys = [k for k in log.keys() if k.startswith("Episode_Termination/")]
        bufs["_term_keys"] = term_keys
        print(f"[eval] discovered termination keys: {[k.replace('Episode_Termination/', '') for k in term_keys]}")

    # ---- per-env: find which termination fired ----
    for term_key in bufs["_term_keys"]:
        name = term_key.replace("Episode_Termination/", "")
        enum_val = TERM_NAME_TO_ENUM.get(name, 255)
        val = log[term_key]
        if not isinstance(val, torch.Tensor):
            continue  # scalar episode-mean (not per-env); skip
        # val is (N,) bool; assign enum for fresh envs where val=True
        hit = val.to(torch.bool) & fresh
        bufs["term_cause"][hit] = enum_val

    # default: any fresh env still with term_cause==-1 → time_out (episode_length hit)
    still_unset = fresh & (bufs["term_cause"] == -1)
    bufs["term_cause"][still_unset] = 0  # time_out

    # ---- term_step ----
    bufs["term_step"][fresh] = t

    # ---- reward_terms ----
    rew_names = bufs.get("_reward_term_names", [])
    for col_i, name in enumerate(rew_names):
        key = f"Episode_Reward/{name}"
        val = log.get(key)
        if isinstance(val, torch.Tensor) and val.shape[0] == bufs["term_cause"].shape[0]:
            bufs["reward_terms"][fresh_idx, col_i] = val[fresh_idx].to(torch.float32)

    bufs["first_done"][fresh] = True
    return discovered
```

- [ ] **Step 2: Verification deferred to Task 8 (need a full step loop to trigger dones)**

- [ ] **Step 3: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/eval_moe.py
git commit -m "eval(moe): handle_dones — first-episode-only event capture"
```

---

## Task 8: Main collection loop

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/eval_moe.py`

- [ ] **Step 1: Replace `main()` tail with the full collection loop**

Replace the body of `main()` after the buffer-allocation block with:

```python
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
            handle_dones(t, dones, info, bufs)

            if (t + 1) % 50 == 0:
                done_frac = bufs["first_done"].float().mean().item()
                print(f"[eval] step {t + 1}/{T}  first_done = {100*done_frac:.1f}%")

            if bufs["first_done"].all():
                print(f"[eval] all envs done at step {t + 1} — early exit")
                break

    # Safety: envs that never died → mark as time_out
    never_done = ~bufs["first_done"]
    if never_done.any():
        bufs["term_cause"][never_done] = 0  # time_out
        bufs["term_step"][never_done] = T - 1
        print(f"[eval] {never_done.sum().item()} envs never terminated — marked time_out")

    print(f"[eval] rollout complete.")
    env_wrapped.close()
```

- [ ] **Step 2: Smoke-run with tiny rollout, verify loop runs and dones are captured**

Run:
```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 64 --num_steps 100
```
Expected:
- Several `[eval] step ../... first_done = N%` prints
- `[eval] discovered N reward terms` (typically 20-30)
- `[eval] discovered termination keys: [...]` includes `reached_goal`
- No errors

- [ ] **Step 3: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/eval_moe.py
git commit -m "eval(moe): main collection loop with per-step + event capture"
```

---

## Task 9: Save raw.npz + summary.json

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/eval_moe.py`

- [ ] **Step 1: Add `save_outputs()`**

```python
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

    # 18-col → sub-terrain name mapping: terrain_types col i belongs to which named sub-terrain?
    # IsaacLab assigns cols by proportion: build cumulative col-index → name map
    proportions = [(name, c.proportion) for name, c in tgen.sub_terrains.items()]
    total = sum(p for _, p in proportions)
    col_to_subterrain = []
    cur_col = 0
    for name, p in proportions:
        n_cols = max(1, round(p / total * tgen.num_cols))
        for _ in range(n_cols):
            col_to_subterrain.append(name)
            cur_col += 1
    # pad to num_cols
    while len(col_to_subterrain) < tgen.num_cols:
        col_to_subterrain.append(proportions[-1][0])
    col_to_subterrain = col_to_subterrain[:tgen.num_cols]

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
```

- [ ] **Step 2: Wire into main() — call save_outputs after rollout, before env_wrapped.close()**

Replace the final two lines of `main()`:

```python
    out_dir = save_outputs(bufs, env_cfg, args, ckpt_path, experiment_name, iter_num)
    print(f"[eval] ALL DONE. results at {out_dir}")
    env_wrapped.close()
```

(Note: env_wrapped is closed twice — once in save_outputs flow doesn't, only main does. Keep the close at the very end.)

- [ ] **Step 3: Smoke-run and verify output files**

Run:
```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 64 --num_steps 100
ls -lh logs/moe_eval/split_moe_teacher_parallel/*/raw.npz
python -c "import numpy as np; d=np.load(sorted(__import__('glob').glob('logs/moe_eval/split_moe_teacher_parallel/*/raw.npz'))[-1]); print({k:v.shape for k,v in d.items()})"
```
Expected: `raw.npz` is ~5-20MB; printout shows all expected keys with correct shapes (e.g. `root_pos_xy=(64, 100, 2)`, `gate_leg=(64, 100, 6)`).

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/eval_moe.py
git commit -m "eval(moe): save raw.npz + summary.json"
```

---

## Task 10: Smoke test — verify sim end-to-end before plotting

**Files:** None new.

- [ ] **Step 1: Run small-scale eval (~30s)**

```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 200 --num_steps 100
```

- [ ] **Step 2: Inspect output via one-liner**

```bash
python <<'EOF'
import json, glob, numpy as np
out = sorted(glob.glob("logs/moe_eval/split_moe_teacher_parallel/*"))[-1]
print("dir:", out)
print("summary:")
print(json.dumps(json.load(open(f"{out}/summary.json")), indent=2)[:1500])
d = np.load(f"{out}/raw.npz")
print("\nbuffer shapes:")
for k in sorted(d.files): print(f"  {k}: shape={d[k].shape} dtype={d[k].dtype}")
print("\nterm_cause histogram:", np.bincount(d['term_cause'] + 1))  # shift to handle -1
EOF
```

Expected:
- `summary.json` lists 13 sub-terrain names, term_enum has `reached_goal: 4`
- All buffer shapes match `(N=200, T=100, ...)`
- `term_cause` mostly 0 (time_out) or 4 (reached_goal) for easy terrains; some 1 (illegal_contact) on hard ones

- [ ] **Step 3: If any unexpected output (NaN, missing keys), debug now before plotting**

No commit — diagnostic-only.

---

## Task 11: Scaffold `plot_moe_eval.py` + aggregation helpers

**Files:**
- Create: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Create the file with argparse, load helpers, and aggregation utilities**

```python
# scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
"""Render 9 plots from eval_moe.py raw.npz output.

Usage: python plot_moe_eval.py --data_dir logs/moe_eval/<exp>/<run>
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(data_dir):
    raw = np.load(os.path.join(data_dir, "raw.npz"))
    with open(os.path.join(data_dir, "summary.json")) as f:
        summary = json.load(f)
    return raw, summary


def alive_mask(term_step, T):
    """Returns (N, T) bool. True iff env was in first episode at step t."""
    N = term_step.shape[0]
    t_idx = np.arange(T)[None, :]
    end = np.where(term_step < 0, T, term_step + 1)[:, None]
    return t_idx < end


def env_subterrain_name(types, summary):
    """Map per-env terrain_types (int col index) to sub-terrain string name."""
    col_map = summary["col_to_subterrain"]
    return np.array([col_map[int(t)] for t in types])


def aggregate_by_subterrain(values, subterrain_names, unique_names, agg="mean"):
    """Group `values` (N,) by sub-terrain name; return array of len(unique_names)."""
    out = np.zeros(len(unique_names))
    for i, name in enumerate(unique_names):
        mask = subterrain_names == name
        if not mask.any():
            out[i] = np.nan
            continue
        if agg == "mean":
            out[i] = values[mask].mean()
        elif agg == "sum":
            out[i] = values[mask].sum()
        elif agg == "count":
            out[i] = mask.sum()
        else:
            raise ValueError(agg)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--success_dist", type=float, default=None,
                   help="Override; default = summary.json's value")
    return p.parse_args()


def main():
    args = parse_args()
    raw, summary = load_data(args.data_dir)
    print(f"[plot] loaded from {args.data_dir}")
    print(f"[plot] N={summary['num_envs']} T={summary['num_steps']} "
          f"success_dist={summary['success_dist']}m")
    sub_names = summary["sub_terrain_names"]
    print(f"[plot] {len(sub_names)} sub-terrains: {sub_names}")

    plots_dir = os.path.join(args.data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[plot] plots dir: {plots_dir}")

    # 9 plots implemented in Tasks 12-20


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run, verify scaffold loads data**

Run (uses Task 10's smoke output):
```bash
cd /home/ouge/Software/rl_training
DIR=$(ls -1d logs/moe_eval/split_moe_teacher_parallel/* | tail -1)
python scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py --data_dir $DIR
```
Expected: prints sub-terrain list (~13 names), plots dir created, no errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): scaffold plot_moe_eval.py with load + aggregation helpers"
```

---

## Task 12: Plot 1 — success_heatmap.png

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Add `plot_success_heatmap()`**

```python
def plot_success_heatmap(raw, summary, plots_dir):
    """Heatmap of success rate. Rows = level (0..num_rows-1), cols = unique sub-terrains."""
    term_cause = raw["term_cause"]
    levels = raw["terrain_levels"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]
    num_rows = summary["num_rows"]

    REACHED_GOAL = 4
    success = (term_cause == REACHED_GOAL)

    sub_per_env = env_subterrain_name(types, summary)

    M = np.full((num_rows, len(sub_names)), np.nan, dtype=np.float32)
    counts = np.zeros((num_rows, len(sub_names)), dtype=np.int32)
    for r in range(num_rows):
        for c, name in enumerate(sub_names):
            mask = (levels == r) & (sub_per_env == name)
            if mask.any():
                M[r, c] = success[mask].mean()
                counts[r, c] = mask.sum()

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(sub_names)), max(6, 0.2 * num_rows)))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(len(sub_names)))
    ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels([f"L{r}" for r in range(num_rows)], fontsize=6)
    ax.set_xlabel("sub-terrain")
    ax.set_ylabel("difficulty level")
    ax.set_title(f"Success rate (reached +{summary['success_dist']:.1f}m within {summary['num_steps']*0.02:.0f}s)")
    plt.colorbar(im, ax=ax, label="success rate")

    # Annotate sample count in each cell
    for r in range(num_rows):
        for c in range(len(sub_names)):
            if counts[r, c] > 0:
                ax.text(c, r, str(counts[r, c]), ha="center", va="center", fontsize=5,
                        color="white" if M[r, c] < 0.5 else "black")

    plt.tight_layout()
    out = os.path.join(plots_dir, "01_success_heatmap.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[plot] {out}")
```

- [ ] **Step 2: Wire into main()**

Append in `main()`:
```python
    plot_success_heatmap(raw, summary, plots_dir)
```

- [ ] **Step 3: Smoke-run + visually inspect**

```bash
cd /home/ouge/Software/rl_training
DIR=$(ls -1d logs/moe_eval/split_moe_teacher_parallel/* | tail -1)
python scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py --data_dir $DIR
ls $DIR/plots/01_success_heatmap.png
```
Expected: PNG exists, ~50-150KB. Open it manually to confirm structure looks right.

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): #1 success heatmap"
```

---

## Task 13: Plot 2 — expert_activation_bars.png

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Add `plot_expert_activation_bars()`**

```python
def plot_expert_activation_bars(raw, summary, plots_dir):
    """Per sub-terrain stacked bar of avg leg/wheel expert weights."""
    gate_leg = raw["gate_leg"].astype(np.float32)    # (N, T, nL)
    gate_wheel = raw["gate_wheel"].astype(np.float32)  # (N, T, nW)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = gate_leg.shape[1]
    am = alive_mask(term_step, T)  # (N, T)

    # mean over (env, time) where alive
    def avg_by_subterrain(gate):
        # gate: (N, T, K)
        am3 = am[:, :, None]
        per_env_avg = (gate * am3).sum(axis=1) / np.maximum(am.sum(axis=1, keepdims=True), 1)  # (N, K)
        sub_per_env = env_subterrain_name(types, summary)
        nK = gate.shape[-1]
        out = np.zeros((len(sub_names), nK))
        for i, name in enumerate(sub_names):
            mask = sub_per_env == name
            if mask.any():
                out[i] = per_env_avg[mask].mean(axis=0)
        return out

    leg_share = avg_by_subterrain(gate_leg)      # (13, nL)
    wheel_share = avg_by_subterrain(gate_wheel)  # (13, nW)

    fig, (ax_l, ax_w) = plt.subplots(2, 1, figsize=(max(8, 0.7 * len(sub_names)), 7), sharex=True)

    def stacked(ax, data, prefix, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        nK = data.shape[1]
        bottom = np.zeros(data.shape[0])
        for k in range(nK):
            ax.bar(range(data.shape[0]), data[:, k], bottom=bottom,
                   color=cmap(k / max(1, nK - 1)), label=f"{prefix}{k}", edgecolor="white", linewidth=0.5)
            bottom += data[:, k]
        ax.set_ylim(0, 1)
        ax.set_ylabel("avg gate weight")
        ax.legend(loc="upper right", fontsize=7, ncol=nK)

    stacked(ax_l, leg_share, "L", "tab10")
    ax_l.set_title("Leg expert activation per sub-terrain")
    stacked(ax_w, wheel_share, "W", "Set2")
    ax_w.set_title("Wheel expert activation per sub-terrain")
    ax_w.set_xticks(range(len(sub_names)))
    ax_w.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(plots_dir, "02_expert_activation_bars.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[plot] {out}")
```

- [ ] **Step 2: Wire into main()**

Append:
```python
    plot_expert_activation_bars(raw, summary, plots_dir)
```

- [ ] **Step 3: Smoke-run + visually inspect**

```bash
cd /home/ouge/Software/rl_training
DIR=$(ls -1d logs/moe_eval/split_moe_teacher_parallel/* | tail -1)
python scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py --data_dir $DIR
```
Expected: `02_expert_activation_bars.png` exists; bar heights sum to ~1.0 per column.

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): #2 expert activation stacked bars"
```

---

## Task 14: Plot 3 — velocity_tracking_box.png

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Add `plot_velocity_tracking_box()`**

```python
def plot_velocity_tracking_box(raw, summary, plots_dir):
    """3-subplot boxplot of |cmd - actual| for vx, vy, wz, grouped by sub-terrain."""
    cmd = raw["cmd"].astype(np.float32)            # (N, T, 3)
    actual = raw["actual_vel"].astype(np.float32)  # (N, T, 3)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = cmd.shape[1]
    am = alive_mask(term_step, T)
    err = np.abs(cmd - actual)  # (N, T, 3)

    sub_per_env = env_subterrain_name(types, summary)

    # collect per-sub-terrain error samples (flatten over env+time where alive)
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 0.7 * len(sub_names)), 9), sharex=True)
    labels = ["vx_err [m/s]", "vy_err [m/s]", "wz_err [rad/s]"]
    for ax_i, ax in enumerate(axes):
        data = []
        for name in sub_names:
            mask = sub_per_env == name
            if not mask.any():
                data.append(np.array([]))
                continue
            vals = err[mask, :, ax_i][am[mask]]
            data.append(vals)
        ax.boxplot(data, showfliers=False, widths=0.6)
        ax.set_ylabel(labels[ax_i])
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(range(1, len(sub_names) + 1))
    axes[-1].set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_title(f"Velocity tracking error per sub-terrain (cmd_vx={summary['cmd_vx']:.2f} m/s)")
    plt.tight_layout()
    out = os.path.join(plots_dir, "03_velocity_tracking_box.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[plot] {out}")
```

- [ ] **Step 2: Wire into main()**

Append:
```python
    plot_velocity_tracking_box(raw, summary, plots_dir)
```

- [ ] **Step 3: Smoke-run + visually inspect**

Expected: 3 vertically-stacked boxplots, 13 boxes each.

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): #3 velocity tracking boxplot"
```

---

## Task 15: Plot 4 — termination_reward_breakdown.png

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Add `plot_termination_reward_breakdown()`**

```python
def plot_termination_reward_breakdown(raw, summary, plots_dir):
    term_cause = raw["term_cause"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]
    term_enum = {int(k): v for k, v in summary["term_enum"].items()}
    sub_per_env = env_subterrain_name(types, summary)

    # ---- top: stacked bar termination cause % per sub-terrain ----
    cause_names = ["time_out", "illegal_contact", "terrain_out_of_bounds", "bad_orientation", "reached_goal"]
    cause_enums = [0, 1, 2, 3, 4]
    cause_share = np.zeros((len(sub_names), len(cause_enums)))
    for i, name in enumerate(sub_names):
        mask = sub_per_env == name
        if not mask.any():
            continue
        for j, e in enumerate(cause_enums):
            cause_share[i, j] = (term_cause[mask] == e).mean()

    # ---- bottom: reward terms heatmap (sub-terrain × top-8 reward terms) ----
    reward_terms_arr = raw["reward_terms"].astype(np.float32) if "reward_terms" in raw.files else None
    rew_names = summary.get("reward_term_names", [])

    if reward_terms_arr is not None and len(rew_names) > 0:
        rew_per_sub = np.zeros((len(sub_names), len(rew_names)))
        for i, name in enumerate(sub_names):
            mask = sub_per_env == name
            if mask.any():
                rew_per_sub[i] = reward_terms_arr[mask].mean(axis=0)
        # pick top-8 by absolute magnitude
        top_idx = np.argsort(-np.abs(rew_per_sub).mean(axis=0))[:8]
        rew_top = rew_per_sub[:, top_idx]
        rew_top_names = [rew_names[i] for i in top_idx]
    else:
        rew_top = None
        rew_top_names = []

    n_axes = 2 if rew_top is not None else 1
    fig, axes = plt.subplots(n_axes, 1, figsize=(max(8, 0.7 * len(sub_names)), 4 + 3 * n_axes))
    if n_axes == 1:
        axes = [axes]

    # top
    ax_t = axes[0]
    colors = ["#666", "#d62728", "#9467bd", "#bcbd22", "#2ca02c"]
    bottom = np.zeros(len(sub_names))
    for j, (cname, col) in enumerate(zip(cause_names, colors)):
        ax_t.bar(range(len(sub_names)), cause_share[:, j], bottom=bottom,
                 label=cname, color=col, edgecolor="white", linewidth=0.5)
        bottom += cause_share[:, j]
    ax_t.set_ylim(0, 1)
    ax_t.set_ylabel("episode fraction")
    ax_t.set_title("Termination cause per sub-terrain")
    ax_t.legend(loc="upper right", fontsize=8, ncol=5)
    ax_t.set_xticks(range(len(sub_names)))
    ax_t.set_xticklabels(sub_names if n_axes == 1 else [""] * len(sub_names),
                         rotation=45, ha="right", fontsize=8)

    # bottom
    if rew_top is not None:
        ax_r = axes[1]
        im = ax_r.imshow(rew_top.T, cmap="RdBu_r", aspect="auto",
                         vmin=-np.abs(rew_top).max(), vmax=np.abs(rew_top).max())
        ax_r.set_yticks(range(len(rew_top_names)))
        ax_r.set_yticklabels(rew_top_names, fontsize=8)
        ax_r.set_xticks(range(len(sub_names)))
        ax_r.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
        ax_r.set_title("Top-8 reward terms (mean per env per sub-terrain)")
        plt.colorbar(im, ax=ax_r, label="reward (Episode_Reward sum)")

    plt.tight_layout()
    out = os.path.join(plots_dir, "04_termination_reward_breakdown.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[plot] {out}")
```

- [ ] **Step 2: Wire into main()**

Append:
```python
    plot_termination_reward_breakdown(raw, summary, plots_dir)
```

- [ ] **Step 3: Smoke-run + visually inspect**

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): #4 termination + reward breakdown"
```

---

## Task 16: Plot 5 — leg_wheel_coactivation.png

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Add `plot_leg_wheel_coactivation()`**

```python
def plot_leg_wheel_coactivation(raw, summary, plots_dir):
    """Joint probability matrix: P(leg_expert=i, wheel_expert=j), averaged over (env, time, alive)."""
    gate_leg = raw["gate_leg"].astype(np.float32)    # (N, T, nL)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    term_step = raw["term_step"]

    T = gate_leg.shape[1]
    am = alive_mask(term_step, T)
    am3 = am[:, :, None, None]

    # outer product per (env, t)
    co = gate_leg[:, :, :, None] * gate_wheel[:, :, None, :]  # (N, T, nL, nW)
    co_sum = (co * am3).sum(axis=(0, 1))
    norm = max(am.sum(), 1)
    co_avg = co_sum / norm

    nL, nW = co_avg.shape
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * nW + 2), max(5, 0.6 * nL + 2)))
    im = ax.imshow(co_avg, cmap="magma", aspect="auto")
    ax.set_xticks(range(nW)); ax.set_xticklabels([f"W{j}" for j in range(nW)])
    ax.set_yticks(range(nL)); ax.set_yticklabels([f"L{i}" for i in range(nL)])
    ax.set_xlabel("wheel expert"); ax.set_ylabel("leg expert")
    ax.set_title("Leg × Wheel expert co-activation (joint avg weight)")
    for i in range(nL):
        for j in range(nW):
            ax.text(j, i, f"{co_avg[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if co_avg[i, j] < co_avg.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = os.path.join(plots_dir, "05_leg_wheel_coactivation.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"[plot] {out}")
```

- [ ] **Step 2: Wire into main()**

Append:
```python
    plot_leg_wheel_coactivation(raw, summary, plots_dir)
```

- [ ] **Step 3: Smoke-run + visually inspect**

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): #5 leg×wheel co-activation heatmap"
```

---

## Task 17: Plot 6 — gate_entropy_per_terrain.png

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Add `plot_gate_entropy_per_terrain()`**

```python
def plot_gate_entropy_per_terrain(raw, summary, plots_dir):
    gate_leg = raw["gate_leg"].astype(np.float32)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = gate_leg.shape[1]
    am = alive_mask(term_step, T)
    eps = 1e-8

    def entropy(g):  # g: (N, T, K)
        return -(g * np.log(g + eps)).sum(axis=-1)  # (N, T)

    H_leg = entropy(gate_leg)
    H_wheel = entropy(gate_wheel)
    sub_per_env = env_subterrain_name(types, summary)

    def avg_by_subterrain(H):
        out = np.full(len(sub_names), np.nan)
        for i, name in enumerate(sub_names):
            mask = sub_per_env == name
            if not mask.any():
                continue
            vals = H[mask][am[mask]]
            out[i] = vals.mean() if vals.size > 0 else np.nan
        return out

    leg_H_per = avg_by_subterrain(H_leg)
    wheel_H_per = avg_by_subterrain(H_wheel)

    nL = gate_leg.shape[-1]; nW = gate_wheel.shape[-1]
    max_leg_H = np.log(nL); max_wheel_H = np.log(nW)

    fig, (ax_l, ax_w) = plt.subplots(2, 1, figsize=(max(8, 0.7 * len(sub_names)), 6), sharex=True)
    ax_l.bar(range(len(sub_names)), leg_H_per, color="#1f77b4")
    ax_l.axhline(max_leg_H, ls="--", color="red", label=f"log({nL})={max_leg_H:.2f} (max)")
    ax_l.set_ylabel("leg gate entropy [nats]"); ax_l.legend(fontsize=8)
    ax_l.set_title("Gate entropy per sub-terrain (lower = more decisive routing)")

    ax_w.bar(range(len(sub_names)), wheel_H_per, color="#2ca02c")
    ax_w.axhline(max_wheel_H, ls="--", color="red", label=f"log({nW})={max_wheel_H:.2f} (max)")
    ax_w.set_ylabel("wheel gate entropy [nats]"); ax_w.legend(fontsize=8)
    ax_w.set_xticks(range(len(sub_names)))
    ax_w.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(plots_dir, "06_gate_entropy_per_terrain.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"[plot] {out}")
```

- [ ] **Step 2: Wire into main()**

Append:
```python
    plot_gate_entropy_per_terrain(raw, summary, plots_dir)
```

- [ ] **Step 3: Smoke-run + visually inspect**

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): #6 gate entropy per sub-terrain"
```

---

## Task 18: Plot 7 — expert_switching_freq.png

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Add `plot_expert_switching_freq()`**

```python
def plot_expert_switching_freq(raw, summary, plots_dir):
    """Per sub-terrain avg number of dominant-expert switches per episode (leg / wheel)."""
    gate_leg = raw["gate_leg"].astype(np.float32)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = gate_leg.shape[1]
    am = alive_mask(term_step, T)
    N = gate_leg.shape[0]

    dom_leg = gate_leg.argmax(axis=-1)    # (N, T)
    dom_wheel = gate_wheel.argmax(axis=-1)

    # count switches: changes between consecutive alive steps
    def switch_count(dom):
        diff = (dom[:, 1:] != dom[:, :-1]) & am[:, 1:] & am[:, :-1]
        return diff.sum(axis=1)

    sw_leg = switch_count(dom_leg).astype(np.float32)
    sw_wheel = switch_count(dom_wheel).astype(np.float32)
    sub_per_env = env_subterrain_name(types, summary)

    sw_leg_per = aggregate_by_subterrain(sw_leg, sub_per_env, sub_names)
    sw_wheel_per = aggregate_by_subterrain(sw_wheel, sub_per_env, sub_names)

    x = np.arange(len(sub_names))
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(sub_names)), 5))
    ax.bar(x - 0.2, sw_leg_per, width=0.4, label="leg expert switches", color="#1f77b4")
    ax.bar(x + 0.2, sw_wheel_per, width=0.4, label="wheel expert switches", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("avg switches / episode")
    ax.set_title("Dominant expert switching frequency per sub-terrain")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, "07_expert_switching_freq.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"[plot] {out}")
```

- [ ] **Step 2: Wire into main()**

Append:
```python
    plot_expert_switching_freq(raw, summary, plots_dir)
```

- [ ] **Step 3: Smoke-run + visually inspect**

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): #7 expert switching frequency"
```

---

## Task 19: Plot 8 — gru_latent_tsne.png

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Add `plot_gru_latent_tsne()`**

```python
def plot_gru_latent_tsne(raw, summary, plots_dir):
    """t-SNE of GRU latent (one point per latent-sampled env, last alive frame), colored by sub-terrain."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[plot] sklearn not installed — skipping plot 8 (t-SNE)")
        return

    latents = raw["gru_latent_sample"].astype(np.float32)  # (N', T', L)
    latent_env_idx = raw["latent_env_idx"]                  # (N',)
    term_step = raw["term_step"]                            # (N,)
    types = raw["terrain_types"]                            # (N,)
    t_stride = int(raw["t_stride"])
    sub_names = summary["sub_terrain_names"]

    # For each sampled env, take the latent at the last alive sampled frame
    points = []
    labels = []
    sub_per_env = env_subterrain_name(types, summary)
    for i, env_i in enumerate(latent_env_idx):
        ts = int(term_step[env_i])
        last_t = ts if ts >= 0 else latents.shape[1] * t_stride - 1
        last_l_idx = min(last_t // t_stride, latents.shape[1] - 1)
        points.append(latents[i, last_l_idx])
        labels.append(sub_per_env[env_i])
    X = np.stack(points, axis=0)
    print(f"[plot] t-SNE input: {X.shape}")

    Y = TSNE(n_components=2, perplexity=min(30, max(5, X.shape[0] // 5)), n_iter=1000,
             random_state=0, init="pca").fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab20")
    for i, name in enumerate(sub_names):
        mask = np.array([l == name for l in labels])
        if not mask.any():
            continue
        ax.scatter(Y[mask, 0], Y[mask, 1], color=cmap(i / max(1, len(sub_names) - 1)),
                   label=name, alpha=0.7, s=20)
    ax.set_title("GRU latent t-SNE (last alive frame per env, colored by sub-terrain)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2")
    plt.tight_layout()
    out = os.path.join(plots_dir, "08_gru_latent_tsne.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"[plot] {out}")
```

- [ ] **Step 2: Wire into main()**

Append:
```python
    plot_gru_latent_tsne(raw, summary, plots_dir)
```

- [ ] **Step 3: Smoke-run + visually inspect**

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): #8 GRU latent t-SNE"
```

---

## Task 20: Plot 9 — survival_curve.png

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py`

- [ ] **Step 1: Add `plot_survival_curve()`**

```python
def plot_survival_curve(raw, summary, plots_dir):
    """% of envs alive vs step, one curve per sub-terrain."""
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]
    T = summary["num_steps"]
    sub_per_env = env_subterrain_name(types, summary)
    am = alive_mask(term_step, T)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")
    dt = 0.02
    t_axis = np.arange(T) * dt
    for i, name in enumerate(sub_names):
        mask = sub_per_env == name
        if not mask.any():
            continue
        surv = am[mask].mean(axis=0)
        ax.plot(t_axis, surv, label=name, color=cmap(i / max(1, len(sub_names) - 1)), lw=1.5)
    ax.set_xlabel("time [s]"); ax.set_ylabel("fraction alive (first episode)")
    ax.set_title("Survival curve per sub-terrain")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    out = os.path.join(plots_dir, "09_survival_curve.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"[plot] {out}")
```

- [ ] **Step 2: Wire into main()**

Append:
```python
    plot_survival_curve(raw, summary, plots_dir)
```

- [ ] **Step 3: Smoke-run + visually inspect all 9 plots**

```bash
cd /home/ouge/Software/rl_training
DIR=$(ls -1d logs/moe_eval/split_moe_teacher_parallel/* | tail -1)
python scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py --data_dir $DIR
ls -lh $DIR/plots/
```
Expected: 9 PNGs all exist, none zero-byte.

- [ ] **Step 4: Commit**

```bash
git add scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
git commit -m "plot(moe): #9 survival curve"
```

---

## Task 21: Full run on the requested checkpoint + final verification

**Files:** None new.

- [ ] **Step 1: Full eval — 2000 envs × 500 steps on latest ckpt**

```bash
cd /home/ouge/Software/rl_training
python scripts/reinforcement_learning/rsl_rl/eval_moe.py --num_envs 2000 --num_steps 500
```
Expected: ~5-15 minutes wall-clock. Final line: `[eval] ALL DONE. results at logs/moe_eval/split_moe_teacher_parallel/<ts>_iter19999`.

- [ ] **Step 2: Run all 9 plots on full data**

```bash
cd /home/ouge/Software/rl_training
DIR=$(ls -1d logs/moe_eval/split_moe_teacher_parallel/* | tail -1)
python scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py --data_dir $DIR
ls -lh $DIR/plots/
```

- [ ] **Step 3: Visual sanity check**

Open the 9 PNGs (use file browser or `xdg-open`). Verify per spec §6:
- Plot 1 (heatmap): level 0 mostly green (>0.9 success); level 29 mostly purple/dark
- Plot 2 (expert bars): bar heights sum to 1.0 per sub-terrain
- Plot 3 (vel box): vx_err medians < 0.5 m/s
- Plot 6 (entropy): leg entropy < log(6) ≈ 1.79; wheel < log(3) ≈ 1.10
- Plot 9 (survival): flat/random_rough stay near 1.0; hard terrains decay

- [ ] **Step 4: Commit any final fixes (if needed) and announce completion**

```bash
git add -A
git status  # confirm clean
# only commit if there were fixes
```

---

## Self-Review Notes

- **Spec coverage**: All 9 plots covered (Tasks 12-20). Eval protocol covered in Tasks 2-3. Auto-checkpoint in Task 4. Hooks in Task 5. Output in Task 9.
- **Tested at**: smoke test after Task 10 (sim sanity), final full run in Task 21.
- **Risks called out in spec are handled**: distillation strip (Task 4), key discovery (Task 7), latent subsample (Task 6), col→subterrain map (Task 9).
- **--strict_per_terrain**: spec calls this opt-in / v2 — flag exists in arg parser (Task 1) but raises NotImplementedError if used. Acceptable for v1.
