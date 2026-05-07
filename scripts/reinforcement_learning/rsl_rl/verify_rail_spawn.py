"""Diagnose why Rail isn't visible in MoE-Rail-Teacher env.

Loads the env, dumps stage tree to see if /World/envs/env_0/Rail exists at all.

Usage:
    conda activate env_isaaclab
    python scripts/reinforcement_learning/rsl_rl/verify_rail_spawn.py
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="MoE-Rail-Teacher-Deeprobotics-M20-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

sys.path.append(os.getcwd())
import rl_training.tasks  # noqa


def dump_tree(prim, depth=0, max_depth=3):
    indent = "  " * depth
    print(f"{indent}{prim.GetPath()} [{prim.GetTypeName()}]")
    if depth >= max_depth:
        return
    for child in prim.GetChildren():
        dump_tree(child, depth + 1, max_depth)


def main():
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)

    # Inspect cfg BEFORE env creation
    print("\n=== A. SceneCfg fields (after __post_init__) ===")
    for name, val in env_cfg.scene.__dict__.items():
        cls = type(val).__name__ if val is not None else "None"
        print(f"  scene.{name}: {cls}")

    print(f"\n  scene.rail: {env_cfg.scene.__dict__.get('rail', 'MISSING')}")
    if hasattr(env_cfg.scene, "rail"):
        print(f"  scene.rail.prim_path: {env_cfg.scene.rail.prim_path}")
        print(f"  scene.rail.spawn: {type(env_cfg.scene.rail.spawn).__name__}")
        print(f"  scene.rail.init_state.pos: {env_cfg.scene.rail.init_state.pos}")

    # Now create env (this triggers spawn)
    print("\n=== B. Creating env ... ===")
    try:
        env_gym = gym.make(args.task, cfg=env_cfg)
        env = env_gym.unwrapped
        print("  env created OK")
        # 显式 reset + 一步, 让 RigidObject runtime 把 env_origin 偏移写到 sim
        env.reset()
        # 跑几步让 USD 跟 PhysX 同步
        import torch
        zero_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(3):
            env_gym.step(zero_action)
        print("  env reset + 3 steps done")
    except Exception as e:
        print(f"  env creation FAILED: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return

    # Check rail world pos via PhysX state (this is what physics actually sees)
    print("\n=== B2. Rail PhysX root pose vs USD pose ===")
    if "rail" in env.scene.rigid_objects:
        rail_obj = env.scene["rail"]
        for env_id in range(min(args.num_envs, 4)):
            phys_pos = rail_obj.data.root_pos_w[env_id].cpu().numpy()
            print(f"  env_{env_id} rail PhysX world pos: ({phys_pos[0]:.2f}, {phys_pos[1]:.2f}, {phys_pos[2]:.2f})")

    # Dump stage tree (focus on /World/envs/env_0)
    print("\n=== C. Stage tree under /World/envs/env_0 ===")
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    env0 = stage.GetPrimAtPath("/World/envs/env_0")
    if env0.IsValid():
        dump_tree(env0, max_depth=2)
    else:
        print("  /World/envs/env_0 NOT FOUND")

    print("\n=== D. Rail prims search ===")
    rail_prim = stage.GetPrimAtPath("/World/envs/env_0/Rail")
    print(f"  /World/envs/env_0/Rail valid?  {rail_prim.IsValid()}")
    if rail_prim.IsValid():
        print(f"    type: {rail_prim.GetTypeName()}")
        print(f"    children:")
        for c in rail_prim.GetChildren():
            print(f"      - {c.GetPath()} [{c.GetTypeName()}]")
        # Position
        from pxr import UsdGeom
        xform = UsdGeom.Xformable(rail_prim)
        if xform:
            ops = xform.GetOrderedXformOps()
            for op in ops:
                print(f"    xform op: {op.GetOpName()} = {op.Get()}")

    print("\n=== E. Per-env rail check (世界位置) ===")
    from pxr import UsdGeom
    for env_id in range(min(args.num_envs, 4)):
        env_path = f"/World/envs/env_{env_id}"
        rail_path = f"{env_path}/Rail"
        env_prim = stage.GetPrimAtPath(env_path)
        rail_prim = stage.GetPrimAtPath(rail_path)
        if env_prim.IsValid():
            env_xform = UsdGeom.Xformable(env_prim)
            env_translate = "(no xform)"
            for op in env_xform.GetOrderedXformOps():
                if "translate" in op.GetOpName():
                    env_translate = str(op.Get())
            print(f"  env_{env_id} translate: {env_translate}")
        if rail_prim.IsValid():
            rail_xform = UsdGeom.Xformable(rail_prim)
            rail_translate = "(no xform)"
            for op in rail_xform.GetOrderedXformOps():
                if "translate" in op.GetOpName():
                    rail_translate = str(op.Get())
            # Compute world pos via Xform cache
            world_xform = UsdGeom.XformCache().GetLocalToWorldTransform(rail_prim)
            world_pos = world_xform.ExtractTranslation()
            print(f"    rail local translate: {rail_translate}")
            print(f"    rail WORLD pos: ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})")

    print("\n=== F. /World children (top level) ===")
    world = stage.GetPrimAtPath("/World")
    if world.IsValid():
        for c in world.GetChildren():
            print(f"  {c.GetPath()} [{c.GetTypeName()}]")

    env_gym.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
