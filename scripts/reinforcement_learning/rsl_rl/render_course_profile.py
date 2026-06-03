"""Side-elevation (centerline cross-section) of the obstacle course for the paper.

Slices the course mesh at y=0 and plots the x-z cross-section as a clean line
drawing, so each obstacle's actual shape/height is visible (crawl hurdle = a
floating bar; slope = a ramp; stairs = steps; rail = a low bar; stones = two
blocks with a void; step-up = a raised platform). Shows ONE cycle (6 obstacles);
the full course repeats it x2 (33.2 m). Outputs PDF (vector) + PNG.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", type=str, default="hard", choices=["easy", "med", "hard", "extreme"])
    p.add_argument("--out_dir", type=str, default="logs/moe_eval/course_sota_report/course_terrain")
    args = p.parse_args()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    app = launcher.app
    try:
        sys.path.insert(0, "source/rl_training")
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from rl_training.terrains.config.course import (
            COURSE_EASY_CFG, COURSE_MED_CFG, COURSE_HARD_CFG, COURSE_EXTREME_CFG)
        from rl_training.terrains.course_terrain_generator import MeshLinearObstacleCourseGenerator
        cfg = {"easy": COURSE_EASY_CFG, "med": COURSE_MED_CFG,
               "hard": COURSE_HARD_CFG, "extreme": COURSE_EXTREME_CFG}[args.difficulty]
        dval = {"easy": 0.40, "med": 0.50, "hard": 0.70, "extreme": 0.95}[args.difficulty]

        gen = MeshLinearObstacleCourseGenerator(cfg=cfg, device="cpu")
        mesh = gen.terrain_mesh

        # cross-section at y=0 (the centerline) -> 3D path with y~0
        sec = mesh.section(plane_origin=(0.0, 0.0, 0.0), plane_normal=(0.0, 1.0, 0.0))
        V = np.asarray(sec.vertices)

        fig, ax = plt.subplots(figsize=(13, 2.6))
        for e in sec.entities:
            idx = np.asarray(e.points)
            pts = V[idx]
            ax.plot(pts[:, 0], pts[:, 2], color="#222", lw=0.7, solid_capstyle="round")

        # obstacle spans (object x) for the first cycle + label centers
        spans = {"hurdle (crawl)": (2.0, 2.5), "slope": (3.7, 5.3), "stairs": (6.5, 8.0),
                 "rail": (9.2, 10.2), "stepping stones": (11.4, 13.2), "step-up": (14.4, 15.4)}
        ztop = float(V[:, 2].max())
        for name, (x0, x1) in spans.items():
            cx = 0.5 * (x0 + x1)
            ax.axvspan(x0, x1, color="#fff3cd", alpha=0.45, zorder=0)
            ax.text(cx, ztop + 0.12, name, ha="center", va="bottom", fontsize=8,
                    rotation=0, color="#444")

        ax.set_xlim(1.5, 16.8)
        ax.set_ylim(V[:, 2].min() - 0.1, ztop + 0.45)
        ax.set_xlabel("distance along course (m) — robot enters from the left", fontsize=9)
        ax.set_ylabel("height z (m)", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_title(f"Obstacle-course centerline profile (difficulty $d={dval}$; one of two repeated cycles)",
                     fontsize=10)
        plt.tight_layout()

        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            f = out / f"course_profile_{args.difficulty}.{ext}"
            plt.savefig(f, dpi=200, bbox_inches="tight")
            print(f"[saved] {f}", flush=True)
        plt.close(fig)
    finally:
        app.close()


if __name__ == "__main__":
    main()
