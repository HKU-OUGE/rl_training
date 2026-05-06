"""Verify forward/backward LiDAR sensor extrinsics + geometric blind-zone reachability.

Tests:
  1. Sensor offset consistency: forward_lidar.pos_w ?= root_pos_w + R(root_quat) @ (0.32028, 0, -0.013)
  2. Open-ground baseline: how many rays land < 0.35m on flat terrain?
  3. Wall-approach: teleport robot to fixed distances [1.0, 0.5, 0.3, 0.15, 0.05]m from a platform wall,
     check polar=0 (boresight) raw_d == expected, and how many rays hit < 0.35m.

Usage:
    conda activate env_isaaclab
    python scripts/reinforcement_learning/rsl_rl/verify_lidar_extrinsics.py
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="MoE-Platform-Teacher-Deeprobotics-M20-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

sys.path.append(os.getcwd())
import rl_training.tasks  # noqa


EXPECTED_FWD_OFFSET = torch.tensor([0.32028, 0.0, -0.013])
EXPECTED_BWD_OFFSET = torch.tensor([-0.32028, 0.0, -0.013])
SPAWN_Z = 0.52


def quat_rotate(q, v):
    """Rotate vector v by quaternion q=(w,x,y,z). Inputs (..., 4), (..., 3)."""
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    t = 2 * torch.cross(torch.cat([x, y, z], dim=-1), v, dim=-1)
    return v + w * t + torch.cross(torch.cat([x, y, z], dim=-1), t, dim=-1)


def main():
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_gym = gym.make(args.task, cfg=env_cfg)
    env = env_gym.unwrapped

    print(f"\n=== Task: {args.task}  num_envs={args.num_envs} ===")
    env.reset()

    # Force max difficulty so platform walls are tallest
    ti = env.scene.terrain
    max_level = ti.terrain_origins.shape[0] - 1
    ti.terrain_levels[:] = max_level
    ti.env_origins[:] = ti.terrain_origins[ti.terrain_levels, ti.terrain_types]
    print(f"[setup] forced terrain_levels → {max_level} (max difficulty)")

    robot = env.scene["robot"]
    fwd = env.scene.sensors["forward_lidar"]
    bwd = env.scene.sensors["backward_lidar"]
    device = robot.device
    num_envs = args.num_envs

    default_jpos = robot.data.default_joint_pos.clone()
    default_jvel = torch.zeros_like(robot.data.default_joint_vel)
    zero_root_vel = torch.zeros(num_envs, 6, device=device)
    sim_dt = env.sim.get_physics_dt()
    n_pin_steps = max(int(0.2 / sim_dt) + 1, 40)

    origins = env.scene.terrain.env_origins.clone()

    def teleport_and_pin(target_xy_offset):
        """target_xy_offset: (dx, dy) in env-local frame. Robot facing +x (identity quat)."""
        new_pose = torch.zeros(num_envs, 7, device=device)
        new_pose[:, 0:2] = origins[:, 0:2] + torch.tensor(target_xy_offset, device=device)
        new_pose[:, 2] = origins[:, 2] + SPAWN_Z
        new_pose[:, 3] = 1.0  # identity quat
        for _ in range(n_pin_steps):
            robot.write_root_pose_to_sim(new_pose)
            robot.write_root_velocity_to_sim(zero_root_vel)
            robot.write_joint_state_to_sim(default_jpos, default_jvel)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(sim_dt)

    # ===== Test 1: sensor offset consistency =====
    print("\n========== TEST 1: sensor offset (root + R(quat)·offset == sensor.pos_w?) ==========")
    teleport_and_pin((0.0, 0.0))  # spawn at origin

    root_pos = robot.data.root_pos_w[0].cpu()
    root_quat = robot.data.root_quat_w[0].cpu()  # (w,x,y,z)
    fwd_sensor_pos = fwd.data.pos_w[0].cpu()
    bwd_sensor_pos = bwd.data.pos_w[0].cpu()

    expected_fwd_world = root_pos + quat_rotate(
        root_quat.unsqueeze(0), EXPECTED_FWD_OFFSET.unsqueeze(0)
    ).squeeze(0)
    expected_bwd_world = root_pos + quat_rotate(
        root_quat.unsqueeze(0), EXPECTED_BWD_OFFSET.unsqueeze(0)
    ).squeeze(0)

    print(f"  root_pos_w[0]      = {root_pos.numpy().round(3)}")
    print(f"  root_quat_w[0]     = {root_quat.numpy().round(3)}  (wxyz)")
    print(f"  fwd sensor pos_w   = {fwd_sensor_pos.numpy().round(3)}")
    print(f"  fwd expected pos_w = {expected_fwd_world.numpy().round(3)}  (root + R·{EXPECTED_FWD_OFFSET.numpy()})")
    print(f"  fwd offset error   = {(fwd_sensor_pos - expected_fwd_world).norm().item():.4f} m")
    print(f"  bwd sensor pos_w   = {bwd_sensor_pos.numpy().round(3)}")
    print(f"  bwd expected pos_w = {expected_bwd_world.numpy().round(3)}")
    print(f"  bwd offset error   = {(bwd_sensor_pos - expected_bwd_world).norm().item():.4f} m")

    # CHECK: _ray_starts_w is the *actual* ray origin (with offset baked in)
    fwd_ray_start_world = fwd._ray_starts_w[0, 0].cpu()  # any ray index works for translation
    err_to_expected = (fwd_ray_start_world - expected_fwd_world).norm().item()
    print(f"  fwd _ray_starts_w[0, 0] = {fwd_ray_start_world.numpy().round(3)}  (actual ray world origin)")
    print(f"     expected vs actual error = {err_to_expected:.4f} m  (should be ~0 if offset really applied)")

    # ===== Test 2: open ground baseline =====
    print("\n========== TEST 2: open ground baseline (no walls near) ==========")
    # spawn at a corner of platform terrain hopefully open
    teleport_and_pin((-3.0, 0.0))  # 3m back from sub-terrain center
    raw_d_fwd = (fwd.data.ray_hits_w[0] - fwd.data.pos_w[0]).norm(dim=-1)
    raw_d_bwd = (bwd.data.ray_hits_w[0] - bwd.data.pos_w[0]).norm(dim=-1)
    finite_fwd = torch.isfinite(raw_d_fwd)
    finite_bwd = torch.isfinite(raw_d_bwd)
    print(f"  fwd: rays_finite={int(finite_fwd.sum())}/496, "
          f"min={raw_d_fwd[finite_fwd].min().item():.3f}m, "
          f"<0.35: {int((finite_fwd & (raw_d_fwd < 0.35)).sum())}, "
          f"<0.5: {int((finite_fwd & (raw_d_fwd < 0.5)).sum())}")
    print(f"  bwd: rays_finite={int(finite_bwd.sum())}/496, "
          f"min={raw_d_bwd[finite_bwd].min().item():.3f}m, "
          f"<0.35: {int((finite_bwd & (raw_d_bwd < 0.35)).sum())}, "
          f"<0.5: {int((finite_bwd & (raw_d_bwd < 0.5)).sum())}")

    # ===== Test 3: wall approach =====
    # Box terrains in PLATFORM_TEACHER_TERRAINS_CFG: edge at OBSTACLE_OFFSET_X = 2.0m from env_origin
    # Robot teleports to (2.0 - distance) from origin, facing +x. Sensor at robot.x + 0.32m, so
    #   expected polar=0 raw_d = (2.0) - (2.0 - distance) - 0.32 = distance - 0.32
    OBSTACLE_X = 2.0
    print(f"\n========== TEST 3: wall approach (box edge at +{OBSTACLE_X}m from env origin) ==========")
    print(f"           sensor offset is +0.32m forward, so expected polar=0 raw_d = (distance - 0.32)")
    print(f"  {'distance(m)':<12}{'expected raw_d(m)':<22}{'polar=0 actual(m)':<22}{'min raw_d':<12}{'<0.35':<8}{'<0.50':<8}")
    print(f"  {'-'*82}")

    print("           对比两种距离公式: WRONG = ||hit - pos_w||, FIXED = ||hit - _ray_starts_w||")
    print(f"  {'distance(m)':<12}{'expected(m)':<14}{'WRONG p=0':<12}{'FIXED p=0':<12}{'WRONG min':<12}{'FIXED min':<12}{'FIXED <0.35':<12}")
    print(f"  {'-'*82}")

    for dist in [1.5, 1.0, 0.7, 0.5, 0.4, 0.35, 0.32, 0.30, 0.20, 0.10, 0.05]:
        teleport_and_pin((OBSTACLE_X - dist, 0.0))
        # WRONG: ||hit - pos_w||  (pos_w = base_link)
        raw_d_wrong = (fwd.data.ray_hits_w[0] - fwd.data.pos_w[0]).norm(dim=-1)
        # FIXED: ||hit - _ray_starts_w||  (true sensor world position per ray)
        raw_d_fix = (fwd.data.ray_hits_w[0] - fwd._ray_starts_w[0]).norm(dim=-1)
        finite = torch.isfinite(raw_d_wrong)
        polar0_wrong = raw_d_wrong[0:31][finite[0:31]].mean().item() if finite[0:31].any() else float("nan")
        polar0_fix = raw_d_fix[0:31][finite[0:31]].mean().item() if finite[0:31].any() else float("nan")
        min_wrong = raw_d_wrong[finite].min().item() if finite.any() else float("nan")
        min_fix = raw_d_fix[finite].min().item() if finite.any() else float("nan")
        n_blind_fix = int((finite & (raw_d_fix < 0.35)).sum().item())
        expected = dist - 0.32028
        print(f"  {dist:<12.2f}{expected:<14.3f}{polar0_wrong:<12.3f}{polar0_fix:<12.3f}{min_wrong:<12.3f}{min_fix:<12.3f}{n_blind_fix:<12d}")

    # Also dump robot world pos for the closest test (sanity check no wall penetration)
    rp = robot.data.root_pos_w[0].cpu().numpy()
    sp = fwd.data.pos_w[0].cpu().numpy()
    print(f"\n  At dist=0.05: root_pos_w={rp.round(3)}, fwd sensor pos_w={sp.round(3)}")
    print(f"               wall is at env_origin.x + {OBSTACLE_X} = {origins[0,0].item() + OBSTACLE_X:.3f}")
    print(f"               sensor.x to wall = {origins[0,0].item() + OBSTACLE_X - sp[0]:.3f}m")

    env_gym.close()
    simulation_app.close()
    print("\n✅ DONE")


if __name__ == "__main__":
    main()
