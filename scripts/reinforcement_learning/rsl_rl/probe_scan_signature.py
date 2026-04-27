"""Probe forward-scan obs signatures for hurdle vs platform terrain.

Builds a teacher env (no policy needed), teleports each env's robot to a
fixed offset in front of its sub-terrain's obstacle, refreshes sensors, and
dumps the per-pitch-layer forward-scan distance distribution per
sub-terrain. Used to verify whether the 6-layer scan can geometrically
distinguish hurdle (open below the bar) from platform/pit (solid wall).

Usage (run separately, results merged manually):
    conda activate env_isaaclab
    # SCAN env, current default config (CFG with 4 hurdle variants)
    python scripts/reinforcement_learning/rsl_rl/probe_scan_signature.py \
        --task MoE-Scan-Teacher-Deeprobotics-M20-v0 \
        --output probe_scan_cfg.json

    # SCAN env, override to CFG2 (5 hurdle variants incl. hurdle5)
    python scripts/reinforcement_learning/rsl_rl/probe_scan_signature.py \
        --task MoE-Scan-Teacher-Deeprobotics-M20-v0 --use-cfg2 \
        --output probe_scan_cfg2.json

    # PLATFORM env (pit + box variants)
    python scripts/reinforcement_learning/rsl_rl/probe_scan_signature.py \
        --task MoE-Platform-Teacher-Deeprobotics-M20-v0 \
        --output probe_platform.json
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe scan obs for hurdle/platform discrimination.")
parser.add_argument("--task", type=str, required=True,
                    help="MoE-Scan-Teacher-Deeprobotics-M20-v0 or MoE-Platform-Teacher-Deeprobotics-M20-v0")
parser.add_argument("--use-cfg2", action="store_true",
                    help="For scan task: override terrain_generator to SCAN_TEACHER_TERRAINS_CFG2 (5 hurdle variants)")
parser.add_argument("--num_envs", type=int, default=300,
                    help="Number of envs (≥ num_rows*num_cols recommended for full coverage)")
parser.add_argument("--distances", type=str, default="0.5,1.0,1.5,2.0,3.0",
                    help="Comma-separated distances (m) from robot to obstacle to probe")
parser.add_argument("--output", type=str, default=None,
                    help="JSON output path (default: probe_<task>.json in cwd)")
parser.add_argument("--force-max-difficulty", action="store_true",
                    help="Force all envs to terrain_levels = num_rows-1 (hardest variant)")
parser.add_argument("--simulated-max-distance", type=str, default=None,
                    help="Comma-separated max_distance values to post-process (e.g. '5.0,3.0,2.0,1.5'). "
                         "Each value caps raw depths and is reported separately.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

sys.path.append(os.getcwd())

# Trigger rl_training task registration
import rl_training.tasks  # noqa: F401


def _derive_col_to_subterrain(gen_cfg):
    """Reproduce IsaacLab's curriculum column → sub-terrain assignment."""
    sub_names = list(gen_cfg.sub_terrains.keys())
    proportions = np.array([gen_cfg.sub_terrains[k].proportion for k in sub_names], dtype=np.float64)
    proportions /= proportions.sum()
    cumprop = np.cumsum(proportions)
    out = []
    for col in range(gen_cfg.num_cols):
        idx = int(np.min(np.where(col / gen_cfg.num_cols + 0.001 < cumprop)[0]))
        out.append(sub_names[idx])
    return out


def main():
    distances = [float(d) for d in args.distances.split(",")]
    sim_max_caps = (
        [float(v) for v in args.simulated_max_distance.split(",")]
        if args.simulated_max_distance is not None
        else [5.0]
    )
    out_path = args.output or f"probe_{args.task.replace('-', '_')}.json"

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    if args.use_cfg2:
        from rl_training.terrains.config.rough import SCAN_TEACHER_TERRAINS_CFG2
        env_cfg.scene.terrain.terrain_generator = SCAN_TEACHER_TERRAINS_CFG2
        print(f"[probe] overriding scan terrain to SCAN_TEACHER_TERRAINS_CFG2")

    env_gym = gym.make(args.task, cfg=env_cfg)
    env = env_gym.unwrapped

    print(f"[probe] task={args.task}  num_envs={args.num_envs}")
    env.reset()

    # Force all envs to hardest difficulty (last row of curriculum) if requested
    if args.force_max_difficulty:
        ti = env.scene.terrain
        max_level = ti.terrain_origins.shape[0] - 1
        ti.terrain_levels[:] = max_level
        ti.env_origins[:] = ti.terrain_origins[ti.terrain_levels, ti.terrain_types]
        print(f"[probe] forced terrain_levels → {max_level} (hardest)")

    # ---- Identify each env's sub-terrain & difficulty ----
    gen_cfg = env.scene.terrain.cfg.terrain_generator
    col_to_sub = _derive_col_to_subterrain(gen_cfg)
    num_rows = gen_cfg.num_rows
    terrain_levels = env.scene.terrain.terrain_levels.cpu().numpy()
    terrain_types = env.scene.terrain.terrain_types.cpu().numpy()
    sub_per_env = np.array([col_to_sub[t] for t in terrain_types])
    diff_per_env = terrain_levels / max(num_rows - 1, 1)

    print(f"[probe] sub-terrains: {sorted(set(col_to_sub))}")
    print(f"[probe] env count per sub: " + ", ".join(
        f"{s}={int((sub_per_env == s).sum())}" for s in sorted(set(col_to_sub))))

    origins = env.scene.terrain.env_origins.clone()  # (N, 3)
    robot = env.scene["robot"]
    device = robot.device
    num_envs = args.num_envs

    default_jpos = robot.data.default_joint_pos.clone()
    default_jvel = torch.zeros_like(robot.data.default_joint_vel)
    zero_root_vel = torch.zeros(num_envs, 6, device=device)

    sim_dt = env.sim.get_physics_dt()
    n_pin_steps = max(int(0.15 / sim_dt) + 1, 30)

    # ---- All sub-terrain types in this run share a +x obstacle at ~2m from cell origin ----
    # Hurdle: bar at center_x + W/2 + post/2 = 2.04m
    # Box: edge at center_x + platform_width/2 = 2.0m
    # Pit: inner wall at center_x + platform_width/2 = 2.0m
    # We use 2.0m uniformly; the hurdle 4cm offset is negligible vs 5m max range.
    OBSTACLE_OFFSET_X = 2.0
    SPAWN_Z = 0.52  # M20 init z

    results = {"task": args.task, "use_cfg2": args.use_cfg2, "distances": distances,
               "simulated_max_caps": sim_max_caps, "data": {}}

    for distance in distances:
        new_pos = origins.clone()
        new_pos[:, 0] += OBSTACLE_OFFSET_X - distance
        new_pos[:, 2] += SPAWN_Z

        new_pose = torch.zeros(num_envs, 7, device=device)
        new_pose[:, 0:3] = new_pos
        new_pose[:, 3] = 1.0  # quat w (identity → facing +x)

        # Pin robot for several physics ticks so RayCaster (update_period=0.1s) refreshes
        for _ in range(n_pin_steps):
            robot.write_root_pose_to_sim(new_pose)
            robot.write_root_velocity_to_sim(zero_root_vel)
            robot.write_joint_state_to_sim(default_jpos, default_jvel)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(sim_dt)

        # Read each pitch layer's raw distances; post-process per simulated cap
        # raw_layer[li] shape (N, 21) holds finite depths or +inf for no-hit
        raw_layer = []
        for li in range(6):
            sensor = env.scene.sensors[f"forward_scanner_layer{li}"]
            rel = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
            depths = torch.norm(rel, dim=-1)
            depths = torch.nan_to_num(depths, posinf=float("inf"), neginf=float("inf"), nan=float("inf"))
            raw_layer.append(depths)

        # Aggregate by sub_terrain × simulated_cap
        for sub in sorted(set(col_to_sub)):
            mask = sub_per_env == sub
            if mask.sum() == 0:
                continue
            for cap in sim_max_caps:
                key = f"D={distance:.1f}/cap={cap:.1f}/{sub}"
                entry = {"n_envs": int(mask.sum()), "layers": {}}
                for li in range(6):
                    capped = torch.clamp(raw_layer[li], max=cap).cpu().numpy()
                    d = capped[mask]  # (n_match, 21)
                    central = d[:, 8:13]
                    entry["layers"][f"L{li}"] = {
                        "mean": round(float(central.mean()), 3),
                        "min":  round(float(central.min()),  3),
                        "max":  round(float(central.max()),  3),
                        "std":  round(float(central.std()),  3),
                    }
                results["data"][key] = entry

        print(f"[probe] D={distance}m done")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe] wrote {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
