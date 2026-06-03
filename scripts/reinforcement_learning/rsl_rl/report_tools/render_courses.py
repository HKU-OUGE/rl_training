"""Render the 4 difficulty courses: top-down + centerline elevation profile."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str,
                   default="logs/moe_eval/course_sota_report/course_terrain")
    args = p.parse_args()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app

    try:
        sys.path.insert(0, "source/rl_training")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np

        from rl_training.terrains.course_terrain_generator import MeshLinearObstacleCourseGenerator as LinearCourseTerrainGenerator
        from rl_training.terrains.config.course import (
            COURSE_EASY_CFG, COURSE_MED_CFG, COURSE_HARD_CFG, COURSE_EXTREME_CFG,
            COURSE_PATCH_END_X,
        )

        PATCH_NAMES = ["hurdle","slope","stairs","rail","stones","step_up",
                       "hurdle·2","slope·2","stairs·2","rail·2","stones·2","step_up·2"]
        from rl_training.terrains.config.course import COURSE_LENGTH
        COURSE_X0 = -COURSE_LENGTH / 2.0  # mesh centred at 0

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cfgs = [
            ("Easy",    "difficulty 0.40", COURSE_EASY_CFG),
            ("Med",     "difficulty 0.50", COURSE_MED_CFG),
            ("Hard",    "difficulty 0.70", COURSE_HARD_CFG),
            ("Extreme", "difficulty 0.95", COURSE_EXTREME_CFG),
        ]

        def height_profile(mesh, n_bins=600):
            """Return (x_edges, h_max) — max-z per x-bin along centerline (|y|<0.4 m)."""
            V = np.asarray(mesh.vertices)
            x = V[:, 0]; y = V[:, 1]; z = V[:, 2]
            m = (x >= -23.0) & (x <= 23.0) & (np.abs(y) <= 0.4)
            x, z = x[m], z[m]
            bins = np.linspace(-23.0, 23.0, n_bins + 1)
            # max & min z per bin
            idx = np.clip(np.searchsorted(bins, x) - 1, 0, n_bins - 1)
            h_max = np.full(n_bins, np.nan)
            h_min = np.full(n_bins, np.nan)
            for i in range(n_bins):
                m_i = idx == i
                if m_i.any():
                    h_max[i] = z[m_i].max()
                    h_min[i] = z[m_i].min()
            mid = 0.5 * (bins[:-1] + bins[1:])
            return mid, h_max, h_min

        for level_name, subtitle, cfg in cfgs:
            gen = LinearCourseTerrainGenerator(cfg=cfg, device="cpu")
            mesh = gen.terrain_mesh
            V = np.asarray(mesh.vertices, dtype=np.float32)
            F = np.asarray(mesh.faces, dtype=np.int64)

            x_min, x_max = -23.5, 23.5
            y_min, y_max = -4.0, 4.0
            keep_v = (V[:, 0] >= x_min) & (V[:, 0] <= x_max) \
                & (V[:, 1] >= y_min) & (V[:, 1] <= y_max)
            face_keep = keep_v[F].all(axis=1)
            F_c = F[face_keep]

            fig = plt.figure(figsize=(18, 5))

            # ---------- Top-down ----------
            ax1 = fig.add_subplot(1, 1, 1)
            z_face = V[F_c].mean(axis=1)[:, 2]
            tri = ax1.tripcolor(V[:, 0], V[:, 1], F_c, z_face,
                                cmap="terrain", shading="flat",
                                vmin=-0.8, vmax=0.7)
            ax1.set_aspect("equal")
            ax1.set_xlim(COURSE_X0, -COURSE_X0)
            ax1.set_ylim(-3.2, 4.2)
            ax1.set_xlabel("x along course (m) — robot spawns at the left edge")
            ax1.set_ylabel("y (m)")
            ax1.set_title("Top-down view (color = terrain height z, m)", pad=22)
            # patch boundaries + labels above the plot
            prev = 2.0  # lead-in
            for i, end in enumerate(COURSE_PATCH_END_X):
                bx = COURSE_X0 + end
                ax1.axvline(bx, color="red", linewidth=0.5, alpha=0.4, linestyle="--")
                center_x = COURSE_X0 + (prev + end) / 2.0 - 0.6  # mid of slot
                ax1.text(center_x, 3.05, PATCH_NAMES[i], ha="center", va="bottom",
                         fontsize=8, color="black", rotation=40,
                         bbox=dict(facecolor="#fff8e1", alpha=0.9, pad=1.5, edgecolor="#d4a200"))
                prev = end
            # corridor lines
            ax1.axhline(1.5, color="orange", linewidth=0.7, alpha=0.7, linestyle=":")
            ax1.axhline(-1.5, color="orange", linewidth=0.7, alpha=0.7, linestyle=":")
            ax1.text(22.0, -2.6, "|y|<1.5 m corridor", fontsize=8, ha="right",
                     color="darkorange", style="italic")
            fig.colorbar(tri, ax=ax1, fraction=0.018, pad=0.01, label="z (m)")

            fig.suptitle(f"Course — {level_name}  ({subtitle})",
                         fontsize=20, y=1.02)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            out = out_dir / f"course_{level_name.lower()}.png"
            plt.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {out}", flush=True)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
