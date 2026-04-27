"""Headless probe for the MoE teacher policy.

Loads a checkpoint, rolls out many envs across all 33 sub-terrain columns,
and dumps a per-terrain-type / per-expert / per-termination summary as JSON.

Usage:
    conda activate env_isaaclab
    python scripts/reinforcement_learning/rsl_rl/probe_moe.py
        # picks newest run + newest model_*.pt, runs 2000 steps over 128 envs
    python scripts/reinforcement_learning/rsl_rl/probe_moe.py \
        --checkpoint model_200.pt --num_envs 256 --num_steps 4000

Designed to be safe to background while training is going on:
    nohup python scripts/reinforcement_learning/rsl_rl/probe_moe.py \
        --checkpoint model_200.pt > probe_iter200.log 2>&1 &
"""

import argparse
import glob
import json
import os
import re
import sys
import time

from isaaclab.app import AppLauncher

# ---- argparse + headless app launch (must happen before isaaclab imports) ----
parser = argparse.ArgumentParser(description="Probe a trained MoE teacher policy.")
parser.add_argument("--task", type=str, default="Rough-MoE-Teacher-Deeprobotics-M20-v0")
parser.add_argument("--num_envs", type=int, default=96, help="probe envs (spread across the 33 cols)")
parser.add_argument("--num_steps", type=int, default=1500, help="rollout steps after warmup")
parser.add_argument("--warmup_steps", type=int, default=150, help="discarded warm-up steps")
parser.add_argument("--load_run", type=str, default=None, help="run dir name; default = newest")
parser.add_argument("--checkpoint", type=str, default="model_*.pt", help="model_NNN.pt or glob; default = newest")
parser.add_argument("--output", type=str, default=None, help="report path (json); default → run_dir/probes/")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# probe is always headless
args.headless = True
# AppLauncher provides args.device via add_app_launcher_args (default cuda:0)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- post-launch imports ----
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

sys.path.append(os.getcwd())

# Inject custom MoE classes — same trick play_moe.py uses
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import (
    SplitMoEActorCritic,
    SplitMoEPPO,
    SplitMoEStudentTeacher,
)
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEPPO = SplitMoEPPO
try:
    import rsl_rl.runners.distillation_runner as dist_runner_module
    dist_runner_module.SplitMoEStudentTeacher = SplitMoEStudentTeacher
except Exception:
    pass

# ============================================================================
# Helpers
# ============================================================================

def _derive_terrain_layout(base_env):
    """Read sub_terrains and num_cols off the live env to mirror IsaacLab's
    curriculum-mode column → sub-terrain assignment (terrain_generator.py)."""
    gen_cfg = base_env.scene.terrain.cfg.terrain_generator
    sub_terrains = gen_cfg.sub_terrains
    num_cols = gen_cfg.num_cols
    names = list(sub_terrains.keys())
    proportions = np.array([sub_terrains[k].proportion for k in names], dtype=np.float64)
    proportions /= proportions.sum()
    cumprop = np.cumsum(proportions)
    column_to_type = []
    for col in range(num_cols):
        idx = int(np.min(np.where(col / num_cols + 0.001 < cumprop)[0]))
        column_to_type.append(idx)
    return column_to_type, names


def _unwrap(env):
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    if hasattr(cur, "unwrapped") and cur.unwrapped is not cur:
        cur = cur.unwrapped
    return cur


def _resolve_ckpt(experiment_name: str, load_run: str | None, checkpoint: str):
    roots = [
        os.path.join("logs", "moe_training", experiment_name),
        os.path.join("logs", "rsl_rl", experiment_name),
        os.path.join("logs", experiment_name),
    ]
    root = next((p for p in roots if os.path.isdir(p)), None)
    if root is None:
        raise FileNotFoundError(f"no log root for experiment '{experiment_name}'")
    if load_run is None:
        runs = sorted(
            (os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))),
            key=os.path.getmtime,
        )
        if not runs:
            raise FileNotFoundError(f"no runs in {root}")
        run_dir = runs[-1]
        print(f"[probe] auto-selected run: {os.path.basename(run_dir)}")
    elif os.path.isabs(load_run):
        run_dir = load_run
    else:
        run_dir = os.path.join(root, load_run)
    files = sorted(glob.glob(os.path.join(run_dir, checkpoint)), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"no checkpoint matching {checkpoint!r} in {run_dir}")
    return files[-1], run_dir


def _term_done_dict(term_manager, env_ids: torch.Tensor) -> dict:
    """Return {term_name: bool tensor for env_ids} from a TerminationManager."""
    out = {}
    try:
        names = term_manager.active_terms
    except Exception:
        names = []
    for name in names:
        try:
            out[name] = term_manager.get_term(name)[env_ids].cpu()
        except Exception:
            try:
                out[name] = term_manager._term_dones[name][env_ids].cpu()
            except Exception:
                pass
    return out


# ============================================================================
# Main
# ============================================================================

def main():
    device = getattr(args, "device", "cuda:0")
    print(f"[probe] device={device}, num_envs={args.num_envs}, num_steps={args.num_steps}")

    # 1. Build env (no joystick — UniformThresholdVelocityCommand will sample randomly)
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_gym = gym.make(args.task, cfg=env_cfg)
    base_env = _unwrap(env_gym)

    # 2. train_cfg massaging — mirror play_moe.py so PPO inference loads cleanly
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else train_cfg
    train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
    for k in ("checkpoint_wheel", "checkpoint_leg", "freeze_experts"):
        train_cfg_dict["policy"].pop(k, None)
    algo_class = train_cfg_dict["algorithm"].get("class_name", "")
    if "Distillation" in algo_class:
        train_cfg_dict["algorithm"]["class_name"] = "PPO"
        for k in ("gradient_length", "loss_type", "optimizer"):
            train_cfg_dict["algorithm"].pop(k, None)
        train_cfg_dict["algorithm"].setdefault("value_loss_coef", 1.0)
        train_cfg_dict["algorithm"].setdefault("use_clipped_value_loss", True)
        train_cfg_dict["algorithm"].setdefault("clip_param", 0.2)
        train_cfg_dict["algorithm"].setdefault("entropy_coef", 0.01)

    experiment_name = train_cfg_dict.get("experiment_name", "split_moe_teacher_parallel")
    model_path, run_dir = _resolve_ckpt(experiment_name, args.load_run, args.checkpoint)
    iter_match = re.search(r"model_(\d+)\.pt$", os.path.basename(model_path))
    iter_num = int(iter_match.group(1)) if iter_match else -1
    print(f"[probe] ckpt: {model_path}  (iter {iter_num})")

    env_wrapped = RslRlVecEnvWrapper(env_gym, clip_actions=train_cfg_dict.get("clip_actions", True))
    runner = OnPolicyRunner(env_wrapped, train_cfg_dict, log_dir=run_dir, device=device)
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = loaded["model_state_dict"]
    if any(k.startswith("student.") for k in state_dict):
        print("[probe] distillation ckpt detected → loading student subnet")
        new_sd = {}
        for k, v in state_dict.items():
            if not k.startswith("student."):
                continue
            nk = k[len("student."):]
            if nk.startswith("critic"):
                continue
            new_sd[nk] = v
        torch.nn.Module.load_state_dict(runner.alg.policy, new_sd, strict=False)
    else:
        runner.load(model_path)
    policy = runner.get_inference_policy(device=device)
    model_instance = policy.__self__ if hasattr(policy, "__self__") else policy

    # 3. Hook MoE gates — keep a running sum of probs / entropy instead of
    # accumulating raw logits per step. The previous append-to-list pattern
    # leaked CUDA allocator pages over 1500+ steps and could OOM.
    gate_running: dict[str, dict[str, object]] = {
        "Wheel": {"prob_sum": None, "entropy_sum": 0.0, "count": 0, "n_experts": 0},
        "Leg":   {"prob_sum": None, "entropy_sum": 0.0, "count": 0, "n_experts": 0},
    }

    def make_hook(name):
        def _hook(_mod, _inp, output):
            with torch.no_grad():
                tensor = output.detach()
                if tensor.ndim == 3:
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                # tensor: (B, K) where B = num_envs (or T*N for recurrent)
                probs = torch.softmax(tensor, dim=-1)
                eps = 1e-9
                entropy = -(probs * (probs.add(eps).log())).sum(dim=-1).mean()
                rec = gate_running[name]
                if rec["prob_sum"] is None:
                    rec["prob_sum"] = probs.sum(dim=0).cpu()
                    rec["n_experts"] = probs.shape[-1]
                else:
                    rec["prob_sum"] += probs.sum(dim=0).cpu()
                rec["entropy_sum"] += float(entropy.item())
                rec["count"] += int(probs.shape[0])  # number of (env, step) samples
                # critically: do NOT keep tensor reference; let GC reclaim
        return _hook

    if hasattr(model_instance, "wheel_gate"):
        model_instance.wheel_gate.register_forward_hook(make_hook("Wheel"))
    if hasattr(model_instance, "leg_gate"):
        model_instance.leg_gate.register_forward_hook(make_hook("Leg"))

    # 4. Roll-out
    obs, _ = env_wrapped.reset()
    column_to_type_list, terrain_type_names = _derive_terrain_layout(base_env)
    print(f"[probe] terrain layout: {len(column_to_type_list)} cols, types={terrain_type_names}")
    col2type = torch.tensor(column_to_type_list, dtype=torch.long, device=device)
    n_cols = col2type.numel()

    # Per-term reward accumulator: env.reward_manager._step_reward is (N, K) and
    # stores reward/dt for the latest step; we accumulate * dt over each episode
    # and snapshot at done time, since reward_manager._episode_sums is zeroed in
    # _reset_idx before env.step() returns.
    rm = base_env.reward_manager
    reward_term_names = list(rm.active_terms)
    n_terms = len(reward_term_names)
    own_reward_sums = torch.zeros(args.num_envs, n_terms, device=device)
    step_dt = float(getattr(base_env, "step_dt", 0.02))

    def get_terrain_state():
        terrain_types_t = base_env.scene.terrain.terrain_types.long().clamp(max=n_cols - 1)
        sub_t = col2type[terrain_types_t]
        try:
            lvl = base_env.scene.terrain.terrain_levels.long()
        except Exception:
            lvl = torch.zeros_like(terrain_types_t)
        return terrain_types_t, sub_t, lvl

    print(f"[probe] warmup {args.warmup_steps} steps...")
    for _ in range(args.warmup_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env_wrapped.step(actions)

    # Reset running gate stats post-warmup
    for rec in gate_running.values():
        rec["prob_sum"] = None
        rec["entropy_sum"] = 0.0
        rec["count"] = 0
        rec["n_experts"] = 0
    own_reward_sums.zero_()
    # Free any cached fragments after warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cur_step = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    cum_lin_err = torch.zeros(args.num_envs, device=device)
    cum_yaw_err = torch.zeros(args.num_envs, device=device)
    cum_speed = torch.zeros(args.num_envs, device=device)
    cum_action_mag = torch.zeros(args.num_envs, device=device)

    episodes = {
        "len": [],
        "type": [],
        "col": [],
        "level": [],
        "term": [],     # "time_out" / "illegal_contact" / "terrain_out_of_bounds" / "unknown"
        "lin_err": [],  # mean ‖cmd_xy − v_xy‖ across episode
        "yaw_err": [],  # mean |cmd_wz − w_z|
        "mean_speed": [],   # mean |v_xy| (for activity check)
        "mean_action": [],  # mean |action|
        # episode-summed reward per term, shape per-episode = (n_terms,)
        "term_sums": [],
    }

    types_now, subt_now, levels_now = get_terrain_state()

    print(f"[probe] rollout {args.num_steps} steps over {args.num_envs} envs...")
    t0 = time.time()
    for step in range(args.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env_wrapped.step(actions)
        cur_step += 1

        cmd = base_env.command_manager.get_command("base_velocity")
        robot = base_env.scene["robot"]
        v_b = robot.data.root_lin_vel_b
        w_b = robot.data.root_ang_vel_b
        err_lin = torch.norm(cmd[:, :2] - v_b[:, :2], dim=1)
        err_yaw = (cmd[:, 2] - w_b[:, 2]).abs()
        cum_lin_err += err_lin
        cum_yaw_err += err_yaw
        cum_speed += torch.norm(v_b[:, :2], dim=1)
        cum_action_mag += actions.abs().mean(dim=-1)

        # Per-term reward: _step_reward is reward/dt; multiply back by dt so we
        # accumulate the actual per-step reward contribution.
        own_reward_sums += rm._step_reward * step_dt

        if dones.any():
            done_idx = torch.where(dones)[0]
            term_dict = _term_done_dict(base_env.termination_manager, done_idx)

            steps_d = cur_step[done_idx].clamp(min=1)
            for i, env_id in enumerate(done_idx.tolist()):
                # determine termination reason — illegal_contact / terrain_out_of_bounds win over time_out
                reason = "unknown"
                priority = ["illegal_contact", "terrain_out_of_bounds", "bad_orientation_2", "time_out"]
                for r in priority:
                    if r in term_dict and bool(term_dict[r][i]):
                        reason = r
                        break

                ep_len = int(cur_step[env_id].item())
                episodes["len"].append(ep_len)
                episodes["type"].append(int(subt_now[env_id].item()))
                episodes["col"].append(int(types_now[env_id].item()))
                episodes["level"].append(int(levels_now[env_id].item()))
                episodes["term"].append(reason)
                episodes["lin_err"].append(float((cum_lin_err[env_id] / max(ep_len, 1)).item()))
                episodes["yaw_err"].append(float((cum_yaw_err[env_id] / max(ep_len, 1)).item()))
                episodes["mean_speed"].append(float((cum_speed[env_id] / max(ep_len, 1)).item()))
                episodes["mean_action"].append(float((cum_action_mag[env_id] / max(ep_len, 1)).item()))
                episodes["term_sums"].append(own_reward_sums[env_id].cpu().numpy().copy())

            cur_step[done_idx] = 0
            cum_lin_err[done_idx] = 0
            cum_yaw_err[done_idx] = 0
            cum_speed[done_idx] = 0
            cum_action_mag[done_idx] = 0
            own_reward_sums[done_idx] = 0
            # Refresh terrain state — curriculum may have advanced
            types_now, subt_now, levels_now = get_terrain_state()

        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            sps = (step + 1) * args.num_envs / max(elapsed, 1e-6)
            print(f"[probe] step {step + 1}/{args.num_steps}  episodes={len(episodes['len'])}  {sps:.0f} env-steps/s")
            # Periodic CUDA cache flush to keep allocator happy across long rollouts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 5. Aggregate
    ep_len = np.array(episodes["len"])
    ep_type = np.array(episodes["type"])
    ep_col = np.array(episodes["col"])
    ep_lvl = np.array(episodes["level"])
    ep_lin = np.array(episodes["lin_err"])
    ep_yaw = np.array(episodes["yaw_err"])
    ep_spd = np.array(episodes["mean_speed"])
    ep_act = np.array(episodes["mean_action"])
    ep_term = np.array(episodes["term"])

    def frac(mask):
        return float(mask.mean()) if mask.size else 0.0

    overall = {
        "iter": iter_num,
        "num_episodes": int(len(ep_len)),
        "mean_ep_len": float(ep_len.mean()) if ep_len.size else 0.0,
        "term_pct": {
            "time_out": frac(ep_term == "time_out"),
            "illegal_contact": frac(ep_term == "illegal_contact"),
            "bad_orientation_2": frac(ep_term == "bad_orientation_2"),
            "terrain_out_of_bounds": frac(ep_term == "terrain_out_of_bounds"),
            "unknown": frac(ep_term == "unknown"),
        },
        "mean_lin_err_xy": float(ep_lin.mean()) if ep_lin.size else 0.0,
        "mean_yaw_err": float(ep_yaw.mean()) if ep_yaw.size else 0.0,
        "mean_speed_xy": float(ep_spd.mean()) if ep_spd.size else 0.0,
        "mean_action_mag": float(ep_act.mean()) if ep_act.size else 0.0,
    }

    # episode-summed reward per term: (n_eps, n_terms)
    term_sums_arr = np.stack(episodes["term_sums"], axis=0) if episodes["term_sums"] else np.zeros((0, n_terms))

    per_type = {}
    for t_idx, t_name in enumerate(terrain_type_names):
        m = ep_type == t_idx
        if not m.any():
            per_type[t_name] = None
            continue
        # Average per-step reward contribution per term on this terrain:
        #   total reward over episode / mean episode length on this terrain.
        # This lets us compare terms on equal footing across terrains where
        # ep_len varies (failing terrains have shorter episodes).
        ep_len_safe = np.maximum(ep_len[m], 1)
        per_step_term = term_sums_arr[m] / ep_len_safe[:, None]
        per_step_mean = per_step_term.mean(axis=0)
        per_type[t_name] = {
            "n_eps": int(m.sum()),
            "mean_len": float(ep_len[m].mean()),
            "illegal_pct": frac(ep_term[m] == "illegal_contact"),
            "timeout_pct": frac(ep_term[m] == "time_out"),
            "lin_err": float(ep_lin[m].mean()),
            "yaw_err": float(ep_yaw[m].mean()),
            "mean_speed": float(ep_spd[m].mean()),
            "mean_level": float(ep_lvl[m].mean()),
            "reward_per_step": {
                term_name: float(per_step_mean[k])
                for k, term_name in enumerate(reward_term_names)
            },
        }

    per_col = {}
    for c in range(n_cols):
        m = ep_col == c
        if not m.any():
            continue
        per_col[c] = {
            "n_eps": int(m.sum()),
            "mean_len": float(ep_len[m].mean()),
            "illegal_pct": frac(ep_term[m] == "illegal_contact"),
            "lin_err": float(ep_lin[m].mean()),
            "mean_speed": float(ep_spd[m].mean()),
            "mean_level": float(ep_lvl[m].mean()),
        }

    expert_summary = {}
    for name in ("Wheel", "Leg"):
        rec = gate_running[name]
        if rec["count"] == 0 or rec["prob_sum"] is None:
            continue
        avg = (rec["prob_sum"] / rec["count"]).tolist()
        # entropy was averaged over batches inside the hook (mean of per-sample entropy);
        # rec["entropy_sum"] holds sum of those per-batch means → divide by num_batches.
        # Number of batches ≈ number of forward calls. We tracked count of samples;
        # store batch count separately would be cleaner but this approximation matches
        # the original report's mean_entropy semantics within ~1%.
        # Better: rebuild entropy from avg probs as a stable per-step proxy:
        avg_t = torch.tensor(avg)
        eps = 1e-9
        H_from_avg = float(-(avg_t * (avg_t.add(eps).log())).sum())
        expert_summary[name] = {
            "num_experts": int(rec["n_experts"]),
            "avg_prob_per_expert": [round(p, 4) for p in avg],
            "mean_entropy": H_from_avg,
            "max_entropy": float(np.log(rec["n_experts"])),
        }

    # Overall mean per-step reward (across all episodes, regardless of terrain)
    if term_sums_arr.size:
        ep_len_arr = np.maximum(ep_len, 1)
        overall_per_step_term = term_sums_arr / ep_len_arr[:, None]
        overall_reward_per_step = {
            term_name: float(overall_per_step_term[:, k].mean())
            for k, term_name in enumerate(reward_term_names)
        }
    else:
        overall_reward_per_step = {}

    report = {
        "task": args.task,
        "ckpt": model_path,
        "run_dir": run_dir,
        "rollout": {"num_envs": args.num_envs, "num_steps": args.num_steps, "warmup_steps": args.warmup_steps},
        "overall": overall,
        "overall_reward_per_step": overall_reward_per_step,
        "per_terrain_type": per_type,
        "per_terrain_col": per_col,
        "experts": expert_summary,
    }

    out_path = args.output
    if out_path is None:
        probes_dir = os.path.join(run_dir, "probes")
        os.makedirs(probes_dir, exist_ok=True)
        out_path = os.path.join(probes_dir, f"probe_iter{iter_num}_{int(time.time())}.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # 6. Pretty console summary
    print("\n" + "=" * 72)
    print(f"  Probe Summary — iter {iter_num}  ({len(ep_len)} episodes)")
    print("=" * 72)
    print(f"  ep_len:         mean={overall['mean_ep_len']:.1f}")
    print(f"  termination:    timeout={overall['term_pct']['time_out']:.1%}  "
          f"illegal={overall['term_pct']['illegal_contact']:.1%}  "
          f"oob={overall['term_pct']['terrain_out_of_bounds']:.1%}")
    print(f"  tracking:       lin_err={overall['mean_lin_err_xy']:.3f} m/s  "
          f"yaw_err={overall['mean_yaw_err']:.3f} rad/s")
    print(f"  motion:         |v_xy|={overall['mean_speed_xy']:.3f} m/s  "
          f"|action|={overall['mean_action_mag']:.3f}")

    print("\n  Per-terrain-type:")
    print(f"    {'type':<14} {'n':>5} {'len':>6} {'illegal':>8} {'lin_err':>8} {'spd':>6} {'lvl':>6}")
    for t_name in terrain_type_names:
        s = per_type.get(t_name)
        if s is None:
            continue
        print(f"    {t_name:<14} {s['n_eps']:>5} {s['mean_len']:>6.1f} "
              f"{s['illegal_pct']:>7.1%} {s['lin_err']:>8.3f} "
              f"{s['mean_speed']:>6.3f} {s['mean_level']:>6.1f}")

    if expert_summary:
        print("\n  MoE experts (avg gate prob):")
        for name, s in expert_summary.items():
            probs_str = ", ".join(f"{p:.3f}" for p in s["avg_prob_per_expert"])
            print(f"    {name:<6}  [{probs_str}]  entropy={s['mean_entropy']:.3f}/{s['max_entropy']:.3f}")

    # Reward signal per terrain — flag terms that vary most across terrains.
    # These are candidates for terrain-conditional weighting in the future.
    if overall_reward_per_step and per_type:
        print("\n  Reward terms with highest spread across terrains:")
        per_term_terrain_means = {}
        for term in reward_term_names:
            vals = [s["reward_per_step"][term] for s in per_type.values() if s is not None]
            if not vals:
                continue
            arr = np.array(vals)
            per_term_terrain_means[term] = (arr.min(), arr.max(), arr.std(), arr.mean())
        # rank by std
        ranked = sorted(per_term_terrain_means.items(), key=lambda kv: -abs(kv[1][2]))[:10]
        print(f"    {'term':<35} {'mean':>9} {'min':>9} {'max':>9} {'std':>9}")
        for term, (mn, mx, sd, mean) in ranked:
            print(f"    {term:<35} {mean:>9.4f} {mn:>9.4f} {mx:>9.4f} {sd:>9.4f}")

    print(f"\n  → {out_path}")
    print("=" * 72)

    env_gym.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
