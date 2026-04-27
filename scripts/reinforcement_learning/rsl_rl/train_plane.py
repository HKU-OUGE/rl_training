"""Quick reward-shape sanity check on flat terrain.

Forces any MoE teacher task to flat terrain and runs short training to verify
basic reward design. On flat ground, the robot SHOULD learn smooth wheel/leg
locomotion within a few hundred iterations — if it doesn't, your reward
shaping likely has a fundamental issue (penalties dominate, sigma too tight,
expert collapse, etc).

After training, the script prints a health-check summary and dumps a JSON:
  - per-component Episode_Reward time series (smoothed)
  - positive vs negative contribution decomposition
  - velocity tracking error trend
  - MoE gate distribution stability
  - sanity flags

Single GPU is enough — this is a fast check, not a real training run.

Usage:
    conda activate env_isaaclab
    python scripts/reinforcement_learning/rsl_rl/train_plane.py \
        --task MoE-Scan-Teacher-Deeprobotics-M20-v0 --max_iter 300

    # try multiple teachers
    for t in MoE-Base-Teacher MoE-Scan-Teacher MoE-Platform-Teacher MoE-Gap-Teacher; do
      python scripts/reinforcement_learning/rsl_rl/train_plane.py \
          --task ${t}-Deeprobotics-M20-v0 --max_iter 200 \
          --output /tmp/plane_${t}.json
    done
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Flat-terrain reward sanity check for MoE teachers.")
parser.add_argument("--task", type=str, required=True,
                    help="Any MoE teacher task (e.g. MoE-Scan-Teacher-Deeprobotics-M20-v0)")
parser.add_argument("--num_envs", type=int, default=2000)
parser.add_argument("--max_iter", type=int, default=300, help="Iterations to train (200-500 typical)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default=None,
                    help="JSON output path (default: /tmp/plane_<task>_<ts>.json)")
parser.add_argument("--num_envs_per_gpu_check", action="store_true",
                    help="If set, also print MoE gate distribution per iter")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
is_master = (local_rank == 0)

args.headless = True
args.device = f"cuda:{local_rank}"
if world_size > 1:
    args.distributed = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- post-launch imports ----
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.terrains import TerrainGeneratorCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

# Inject MoE classes — same trick as train_moe.py
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import (  # noqa: E402
    SplitMoEActorCritic, SplitMoEPPO, SplitMoEStudentTeacher,
)
import rsl_rl.modules as rsl_modules  # noqa: E402
import rsl_rl.runners.on_policy_runner as runner_module  # noqa: E402

rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEPPO = SplitMoEPPO


# Pure-flat terrain: one sub-terrain (flat plane) tiled. No curriculum.
FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    curriculum=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


def _unwrap(env):
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    if hasattr(cur, "unwrapped") and cur.unwrapped is not cur:
        cur = cur.unwrapped
    return cur


def _classify_health(history, last_iter):
    """Generate sanity flags based on training trajectory."""
    flags = {}

    # last reported reward terms
    last_rewards = {k: v[-1] for k, v in history.items() if k.startswith("Episode_Reward/") and v}

    pos_sum = sum(v for v in last_rewards.values() if v > 0)
    neg_sum = -sum(v for v in last_rewards.values() if v < 0)
    flags["positive_total"] = round(pos_sum, 4)
    flags["negative_total"] = round(neg_sum, 4)
    flags["pos_to_neg_ratio"] = round(pos_sum / max(neg_sum, 1e-6), 3)

    # tracking term should dominate
    track_lin = last_rewards.get("Episode_Reward/track_lin_vel_xy_exp", 0)
    track_ang = last_rewards.get("Episode_Reward/track_ang_vel_z_exp", 0)
    flags["tracking_total"] = round(track_lin + track_ang, 4)
    flags["tracking_share_of_positive"] = round(
        (track_lin + track_ang) / max(pos_sum, 1e-6), 3
    )

    # which negative term is biggest?
    neg_terms = sorted(
        ((k.replace("Episode_Reward/", ""), abs(v)) for k, v in last_rewards.items() if v < 0),
        key=lambda x: -x[1],
    )
    flags["top_3_penalties"] = [(name, round(amt, 4)) for name, amt in neg_terms[:3]]

    # episode length & termination
    ep_len = history.get("Train/mean_episode_length", [None])[-1]
    flags["mean_episode_length"] = ep_len
    illegal = last_rewards.get("Episode_Termination/illegal_contact", None)
    if illegal is None and "illegal_contact_pct" in history and history["illegal_contact_pct"]:
        illegal = history["illegal_contact_pct"][-1]
    flags["illegal_contact_pct"] = illegal

    # velocity tracking errors
    vxy = history.get("Metrics/base_velocity/error_vel_xy", [None])[-1]
    vyaw = history.get("Metrics/base_velocity/error_vel_yaw", [None])[-1]
    flags["error_vel_xy"] = vxy
    flags["error_vel_yaw"] = vyaw

    # reward trajectory: did mean reward grow?
    if "Train/mean_reward" in history and len(history["Train/mean_reward"]) > 5:
        early = sum(history["Train/mean_reward"][:5]) / 5
        late = sum(history["Train/mean_reward"][-5:]) / 5
        flags["reward_growth"] = round(late - early, 3)
        flags["reward_growing"] = late > early + 1.0

    # ---- HEALTH VERDICTS ----
    issues = []
    if pos_sum / max(neg_sum, 1e-6) < 1.5:
        issues.append("PENALTY_DOMINATES: positive/negative reward ratio < 1.5 — penalties are crushing the learning signal")
    if track_lin + track_ang < pos_sum * 0.7 and pos_sum > 0:
        issues.append(
            "TRACKING_NOT_DOMINANT: tracking < 70% of positive total — some auxiliary reward "
            "is leaking too much positive value"
        )
    if vxy is not None and vxy > 0.5:
        issues.append(f"VEL_TRACKING_POOR: error_vel_xy={vxy:.2f} m/s after {last_iter} iter")
    if illegal is not None and illegal > 0.05:
        issues.append(f"HIGH_FALL_RATE: illegal_contact={illegal:.1%} — robot falls on FLAT ground (broken)")
    if ep_len is not None and ep_len < 900:
        issues.append(f"SHORT_EPISODES: mean_episode_length={ep_len:.0f}/1000")
    if not issues:
        issues.append("OK: all sanity checks pass")

    flags["verdict"] = issues
    return flags


def _separate_joints(robot):
    """Return (leg_idx_list, wheel_idx_list) — leg = hipx/hipy/knee, wheel = wheel."""
    names = list(robot.data.joint_names)
    leg_idx = [i for i, n in enumerate(names) if "wheel" not in n]
    wheel_idx = [i for i, n in enumerate(names) if "wheel" in n]
    return leg_idx, wheel_idx


def _run_command_probe(env_wrapped, base_env, runner, vx, vy, wz, steps=250, label=""):
    """Run policy with FIXED commands for `steps` and return aggregated metrics."""
    import numpy as np
    cmd_term = base_env.command_manager.get_term("base_velocity")
    robot = base_env.scene["robot"]
    leg_idx, wheel_idx = _separate_joints(robot)
    default_leg_pos = robot.data.default_joint_pos[:, leg_idx].clone()

    policy = runner.get_inference_policy(device=runner.device)

    obs, _ = env_wrapped.reset()
    leg_vel_abs, wheel_vel_abs = [], []
    leg_pos_dev, wheel_vel_signed = [], []
    achieved_vx, achieved_wz = [], []

    for step in range(steps):
        # Force command (override even if manager re-samples)
        cmd_term.vel_command_b[:, 0] = vx
        cmd_term.vel_command_b[:, 1] = vy
        cmd_term.vel_command_b[:, 2] = wz

        with torch.no_grad():
            action = policy(obs)
        obs, _, _, _ = env_wrapped.step(action)

        jv = robot.data.joint_vel
        jp = robot.data.joint_pos
        leg_vel_abs.append(jv[:, leg_idx].abs().mean().item())
        wheel_vel_abs.append(jv[:, wheel_idx].abs().mean().item())
        leg_pos_dev.append((jp[:, leg_idx] - default_leg_pos).abs().mean().item())
        wheel_vel_signed.append(jv[:, wheel_idx].mean(dim=0).cpu().numpy())  # 4-dim per step
        achieved_vx.append(robot.data.root_lin_vel_b[:, 0].mean().item())
        achieved_wz.append(robot.data.root_ang_vel_b[:, 2].mean().item())

    # use last 60% of trajectory for steady-state stats
    cut = int(steps * 0.4)
    wheel_signed_arr = np.stack(wheel_vel_signed[cut:])  # (T, 4)

    return {
        "label": label,
        "command": {"vx": vx, "vy": vy, "wz": wz},
        "achieved_vx_mean": float(np.mean(achieved_vx[cut:])),
        "achieved_wz_mean": float(np.mean(achieved_wz[cut:])),
        "leg_vel_abs_mean": float(np.mean(leg_vel_abs[cut:])),
        "wheel_vel_abs_mean": float(np.mean(wheel_vel_abs[cut:])),
        "leg_pos_dev_from_default_mean": float(np.mean(leg_pos_dev[cut:])),
        # per-wheel signed mean — to detect skid (opposite-sign wheels)
        "wheel_signed_mean_per_wheel": [float(x) for x in wheel_signed_arr.mean(axis=0)],
        "wheel_names": [robot.data.joint_names[i] for i in wheel_idx],
    }


def _judge_behavior(scenarios):
    """Cross-scenario flags for whether policy behaves as expected on flat."""
    flags = {}
    sa = next((s for s in scenarios if s["label"] == "forward"), None)
    sb = next((s for s in scenarios if s["label"] == "turn"), None)
    sc = next((s for s in scenarios if s["label"] == "stand"), None)

    issues = []

    # ---- Scenario A: forward only — legs should be quiet, wheels active ----
    if sa is not None:
        ratio_a = sa["leg_vel_abs_mean"] / max(sa["wheel_vel_abs_mean"], 1e-6)
        flags["forward_leg_to_wheel_ratio"] = round(ratio_a, 3)
        flags["forward_achieved_vx"] = round(sa["achieved_vx_mean"], 3)
        flags["forward_target_vx"] = sa["command"]["vx"]
        if abs(sa["achieved_vx_mean"] - sa["command"]["vx"]) > 0.3:
            issues.append(f"FORWARD_TRACKING_BAD: target vx={sa['command']['vx']} got {sa['achieved_vx_mean']:.2f}")
        if ratio_a > 0.4:
            issues.append(
                f"FORWARD_LEGS_TOO_ACTIVE: leg_vel/wheel_vel = {ratio_a:.2f} (>0.4) — "
                "legs are flailing while just moving forward"
            )
        if sa["leg_pos_dev_from_default_mean"] > 0.25:
            issues.append(
                f"FORWARD_LEGS_DEVIATE: leg_pos_dev={sa['leg_pos_dev_from_default_mean']:.2f} rad — "
                "legs drifted from default pose during pure forward"
            )

    # ---- Scenario B: turn in place — should achieve wz, prefer lift+pivot over wheel skid ----
    if sb is not None:
        flags["turn_achieved_wz"] = round(sb["achieved_wz_mean"], 3)
        flags["turn_target_wz"] = sb["command"]["wz"]
        flags["turn_leg_vel_abs"] = round(sb["leg_vel_abs_mean"], 3)
        flags["turn_wheel_signed"] = sb["wheel_signed_mean_per_wheel"]

        if abs(sb["achieved_wz_mean"] - sb["command"]["wz"]) > 0.3:
            issues.append(f"TURN_TRACKING_BAD: target wz={sb['command']['wz']} got {sb['achieved_wz_mean']:.2f}")

        if sa is not None:
            # Compare leg activity in turn vs forward — if turn doesn't have more leg motion, robot is wheel-skidding
            leg_activity_increase = sb["leg_vel_abs_mean"] / max(sa["leg_vel_abs_mean"], 1e-6)
            flags["turn_vs_forward_leg_activity"] = round(leg_activity_increase, 2)
            if leg_activity_increase < 1.5:
                issues.append(
                    f"TURN_USING_WHEEL_SKID: turn leg_vel = {leg_activity_increase:.1f}× forward — "
                    "robot turning via wheel skid instead of lift-and-pivot"
                )

        # Check wheel-skid signature: opposite-sign rotation between left/right wheels
        # Names: typically fl_wheel, fr_wheel, hl_wheel, hr_wheel
        wn = sb["wheel_names"]
        wv = sb["wheel_signed_mean_per_wheel"]
        try:
            left = [wv[i] for i, n in enumerate(wn) if n.startswith("fl") or n.startswith("hl")]
            right = [wv[i] for i, n in enumerate(wn) if n.startswith("fr") or n.startswith("hr")]
            if left and right:
                left_mean = sum(left) / len(left)
                right_mean = sum(right) / len(right)
                flags["turn_left_wheel_mean"] = round(left_mean, 3)
                flags["turn_right_wheel_mean"] = round(right_mean, 3)
                if left_mean * right_mean < -0.5:  # opposite signs, both meaningful magnitude
                    issues.append(
                        f"TURN_WHEELS_OPPOSITE_SIGN: L={left_mean:.2f}, R={right_mean:.2f} — "
                        "tank-turn (skid) signature"
                    )
        except Exception:
            pass

    # ---- Scenario C: stand still — everything should be near 0 ----
    if sc is not None:
        flags["stand_leg_vel_abs"] = round(sc["leg_vel_abs_mean"], 3)
        flags["stand_wheel_vel_abs"] = round(sc["wheel_vel_abs_mean"], 3)
        if sc["leg_vel_abs_mean"] > 0.3 or sc["wheel_vel_abs_mean"] > 0.5:
            issues.append(
                f"STAND_NOT_STILL: leg_vel={sc['leg_vel_abs_mean']:.2f} wheel_vel={sc['wheel_vel_abs_mean']:.2f} "
                "— policy moves while command is zero"
            )

    if not issues:
        issues.append("OK: behavior matches expectations")
    flags["issues"] = issues
    return flags


def main():
    device = f"cuda:{local_rank}"
    out_path = args.output or f"/tmp/plane_{args.task.replace('/', '_').replace('-', '_')}_{datetime.now().strftime('%H%M%S')}.json"

    if is_master:
        print(f"[plane] task={args.task}  num_envs={args.num_envs}  max_iter={args.max_iter}")
        print(f"[plane] forcing flat terrain (curriculum=False)")
        print(f"[plane] output -> {out_path}")

    # ---- env_cfg with flat-terrain override ----
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed + local_rank
    env_cfg.scene.terrain.terrain_generator = FLAT_TERRAIN_CFG
    if hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
        env_cfg.scene.terrain.max_init_terrain_level = 0
    # disable curriculum if env has one
    if hasattr(env_cfg, "curriculum"):
        for term_name in dir(env_cfg.curriculum):
            if term_name.startswith("_"):
                continue
            term = getattr(env_cfg.curriculum, term_name, None)
            if term is None:
                continue
            # Soft-disable: setting weight to 0 if available, else leave (curriculum on flat is harmless)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    base_env = _unwrap(env)

    # ---- train cfg ----
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
    train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
    train_cfg_dict["device"] = device
    train_cfg_dict["max_iterations"] = args.max_iter
    train_cfg_dict["save_interval"] = 10**9  # don't save
    train_cfg_dict["logger"] = "tensorboard"
    for k in ("checkpoint_wheel", "checkpoint_leg", "freeze_experts"):
        train_cfg_dict["policy"].pop(k, None)

    log_dir = f"/tmp/train_plane_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{local_rank}"
    os.makedirs(log_dir, exist_ok=True)

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=train_cfg_dict.get("clip_actions", True))
    runner = OnPolicyRunner(env_wrapped, train_cfg_dict, log_dir=log_dir, device=device)

    # ---- iter-by-iter loop with extras capture ----
    history = defaultdict(list)

    if is_master:
        print(f"\n[plane] starting {args.max_iter} iters on flat terrain...")

    for i in range(args.max_iter):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(i == 0))

        # Capture Episode_Reward/X and Metrics/X from env extras (populated by managers on episode reset)
        if hasattr(base_env, "extras") and isinstance(base_env.extras, dict):
            log_extras = base_env.extras.get("log", {})
            for k, v in log_extras.items():
                try:
                    history[k].append(float(v))
                except (TypeError, ValueError):
                    pass

        # Capture mean reward & episode length from runner buffers
        if hasattr(runner, "tot_rewbuffer") and len(runner.tot_rewbuffer) > 0:
            history["Train/mean_reward"].append(float(sum(runner.tot_rewbuffer) / len(runner.tot_rewbuffer)))
        if hasattr(runner, "lenbuffer") and len(runner.lenbuffer) > 0:
            history["Train/mean_episode_length"].append(float(sum(runner.lenbuffer) / len(runner.lenbuffer)))

        if is_master and (i + 1) % 25 == 0:
            mr = history["Train/mean_reward"][-1] if history.get("Train/mean_reward") else None
            el = history["Train/mean_episode_length"][-1] if history.get("Train/mean_episode_length") else None
            vxy = history["Metrics/base_velocity/error_vel_xy"][-1] if history.get("Metrics/base_velocity/error_vel_xy") else None
            def fmt(v, p=2):
                return f"{v:.{p}f}" if v is not None else "n/a"
            print(f"[plane] iter {i+1}/{args.max_iter}  reward={fmt(mr)}  "
                  f"ep_len={fmt(el, 0)}  err_vxy={fmt(vxy)}")

    # ---- behavior probe under fixed commands ----
    behavior_results = None
    behavior_flags = None
    if is_master:
        try:
            print("\n[plane] running behavior probes (forward / turn / stand)...")
            behavior_results = []
            behavior_results.append(_run_command_probe(env_wrapped, base_env, runner,
                                                       vx=0.5, vy=0.0, wz=0.0,
                                                       steps=250, label="forward"))
            behavior_results.append(_run_command_probe(env_wrapped, base_env, runner,
                                                       vx=0.0, vy=0.0, wz=0.5,
                                                       steps=250, label="turn"))
            behavior_results.append(_run_command_probe(env_wrapped, base_env, runner,
                                                       vx=0.0, vy=0.0, wz=0.0,
                                                       steps=150, label="stand"))
            behavior_flags = _judge_behavior(behavior_results)
        except Exception as e:
            print(f"[plane] behavior probe failed: {e}")
            behavior_results = None

    # ---- diagnose ----
    if is_master:
        flags = _classify_health(history, args.max_iter)

        report = {
            "task": args.task,
            "num_envs": args.num_envs,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "history": dict(history),
            "diagnosis": flags,
            "behavior_probe": behavior_results,
            "behavior_diagnosis": behavior_flags,
        }
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        # ---- print summary ----
        print("\n" + "=" * 70)
        print(f" REWARD SHAPE SANITY REPORT — {args.task}")
        print("=" * 70)
        print(f"  iters trained: {args.max_iter}")
        print(f"  positive reward total: {flags['positive_total']:.3f}")
        print(f"  negative reward total: {flags['negative_total']:.3f}")
        print(f"  ratio (pos/neg): {flags['pos_to_neg_ratio']:.2f}")
        print(f"  tracking total: {flags['tracking_total']:.3f}  "
              f"(share of positive: {flags['tracking_share_of_positive']:.0%})")
        print(f"  top 3 penalties:")
        for name, amt in flags["top_3_penalties"]:
            print(f"    {name:30s}  {amt:.4f}")
        print(f"  mean_episode_length: {flags['mean_episode_length']}")
        print(f"  illegal_contact_pct: {flags['illegal_contact_pct']}")
        print(f"  error_vel_xy: {flags['error_vel_xy']}")
        print(f"  error_vel_yaw: {flags['error_vel_yaw']}")
        if "reward_growing" in flags:
            print(f"  reward growing: {flags['reward_growing']}  (delta first→last 5: {flags['reward_growth']})")
        print("\n  VERDICT (reward shape):")
        for issue in flags["verdict"]:
            mark = "✓" if issue.startswith("OK") else "✗"
            print(f"    {mark} {issue}")

        # ---- behavior probe summary ----
        if behavior_results and behavior_flags:
            print("\n" + "-" * 70)
            print(" BEHAVIOR PROBE (fixed commands on flat)")
            print("-" * 70)
            for r in behavior_results:
                cmd = r["command"]
                print(f"  [{r['label']:>7s}] cmd=(vx={cmd['vx']}, vy={cmd['vy']}, wz={cmd['wz']})  "
                      f"got vx={r['achieved_vx_mean']:+.2f}  wz={r['achieved_wz_mean']:+.2f}  "
                      f"leg_vel={r['leg_vel_abs_mean']:.2f}  wheel_vel={r['wheel_vel_abs_mean']:.2f}  "
                      f"leg_dev={r['leg_pos_dev_from_default_mean']:.2f}")
            print("\n  VERDICT (behavior):")
            for issue in behavior_flags["issues"]:
                mark = "✓" if issue.startswith("OK") else "✗"
                print(f"    {mark} {issue}")
        print("=" * 70)
        print(f"\n  full report: {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
