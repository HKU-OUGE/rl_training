# scripts/reinforcement_learning/rsl_rl/render_eval_terrain.py
"""Headless render of the evaluation terrain grid for paper Fig. 4.

Loads MOE_ROUGH_TERRAINS_CFG (the eval terrain), auto-frames a single oblique
top-down camera onto the generated grid, renders one frame and saves a PNG.
Terrain only -- no robots.

Run from the repo root:
    python scripts/reinforcement_learning/rsl_rl/render_eval_terrain.py
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render the eval terrain grid (Fig. 4).")
parser.add_argument("--num_rows", type=int, default=8, help="Difficulty rows to generate.")
parser.add_argument("--out", type=str, default="/tmp/eval_terrain.png", help="Output PNG path.")
parser.add_argument("--res_w", type=int, default=2000, help="Render width (px).")
parser.add_argument("--res_h", type=int, default=900, help="Render height (px).")
parser.add_argument("--elev_deg", type=float, default=45.0, help="Camera elevation angle.")
parser.add_argument("--margin", type=float, default=1.15, help="Frame margin factor (>1).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.sensors import Camera, CameraCfg

sys.path.append(os.getcwd())
from rl_training.terrains.config.rough import MOE_ROUGH_TERRAINS_CFG

_FOCAL = 24.0
_APERTURE = 20.955
_TILE = 8.0  # MOE_ROUGH_TERRAINS_CFG size = (8, 8)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # ---- lighting: low ambient dome + strong tilted sun for terrain relief --
    dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.85, 0.88, 0.95))
    dome.func("/World/DomeLight", dome)
    sun = sim_utils.DistantLightCfg(intensity=3500.0, angle=1.0, color=(1.0, 0.96, 0.90))
    sun.func("/World/Sun", sun, orientation=(0.946, 0.0, 0.324, 0.0))  # ~38 deg tilt

    # ---- terrain: a few difficulty rows of the full eval grid ---------------
    terrain_gen = MOE_ROUGH_TERRAINS_CFG.replace(num_rows=args.num_rows, curriculum=True)
    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen,
        num_envs=1,
        env_spacing=1.0,
        max_init_terrain_level=None,
    )
    terrain = TerrainImporter(terrain_cfg)

    # ---- auto-frame the camera from the terrain bounding box ----------------
    # terrain_origins is the (num_rows, num_cols, 3) grid of tile centres;
    # env_origins (used before) is per-env spawn points and is wrong here.
    origins = terrain.terrain_origins.reshape(-1, 3).detach().cpu().numpy()
    xmin, xmax = float(origins[:, 0].min()), float(origins[:, 0].max())
    ymin, ymax = float(origins[:, 1].min()), float(origins[:, 1].max())
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    hx = (xmax - xmin) / 2.0 + _TILE / 2.0   # half-extent incl. tile size
    hy = (ymax - ymin) / 2.0 + _TILE / 2.0
    long_half = max(hx, hy)

    hfov = 2.0 * math.atan(_APERTURE / (2.0 * _FOCAL))
    # slant distance so the long axis fills the frame width (with margin)
    R = (2.0 * long_half * args.margin) / (2.0 * math.tan(hfov / 2.0))
    phi = math.radians(args.elev_deg)
    ground = R * math.cos(phi)
    up = R * math.sin(phi)
    # look along the SHORT axis so the long axis spans the image width
    if hx >= hy:                       # x is the long axis -> camera offset in -y
        eye = [cx, cy - ground, up]
    else:                              # y is the long axis -> camera offset in -x
        eye = [cx - ground, cy, up]
    target = [cx, cy, 0.0]

    info = (f"[terrain] tiles={origins.shape[0]}  x=[{xmin:.1f},{xmax:.1f}]  "
            f"y=[{ymin:.1f},{ymax:.1f}]  long_half={long_half:.1f}\n"
            f"[camera]  eye={eye}  target={target}  R={R:.1f}  elev={args.elev_deg}")
    print(info)
    with open(os.path.splitext(args.out)[0] + "_info.txt", "w") as f:
        f.write(info + "\n")

    # ---- camera -------------------------------------------------------------
    cam_cfg = CameraCfg(
        prim_path="/World/RenderCam",
        update_period=0,
        height=args.res_h,
        width=args.res_w,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FOCAL, horizontal_aperture=_APERTURE,
            clipping_range=(0.1, 1.0e6),
        ),
    )
    camera = Camera(cfg=cam_cfg)

    sim.reset()
    camera.set_world_poses_from_view(
        eyes=torch.tensor([eye], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
    )

    for _ in range(24):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    plt.imsave(args.out, rgb)
    print(f"[done] saved {args.out}  shape={rgb.shape}")


if __name__ == "__main__":
    main()
    simulation_app.close()
