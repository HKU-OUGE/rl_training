"""Standalone course side-elevation — stubs isaaclab so it runs locally (no Isaac/Warp)."""
import sys, types, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- stub isaaclab so importing the generator module doesn't init Warp/CUDA ---
for m in ["isaaclab", "isaaclab.terrains", "isaaclab.terrains.terrain_generator",
          "isaaclab.terrains.terrain_generator_cfg", "isaaclab.utils"]:
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["isaaclab.terrains.terrain_generator"].TerrainGenerator = object
sys.modules["isaaclab.terrains.terrain_generator_cfg"].TerrainGeneratorCfg = object
sys.modules["isaaclab.utils"].configclass = (lambda c: c)

GEN = "/home/ouge/Software/rl_training/source/rl_training/rl_training/terrains/course_terrain_generator.py"
spec = importlib.util.spec_from_file_location("ctg", GEN)
ctg = importlib.util.module_from_spec(spec)
sys.modules["ctg"] = ctg  # register before exec so @dataclass can resolve cls.__module__
spec.loader.exec_module(ctg)
print("loaded generator standalone; TRACK_Z=%s NUM_CYCLES=%s" % (ctg.TRACK_Z, ctg.NUM_CYCLES))

_DIFF_RANGES = {
    "hurdle":  {"hurdle_height_min": 0.35, "hurdle_height_max": 0.55, "bar_thickness": 0.08},
    "slope":   {"slope_min": 0.10, "slope_max": 0.55},
    "stairs":  {"step_rise_min": 0.05, "step_rise_max": 0.24, "step_run": 0.2, "num_steps": 3},
    "rail":    {"rail_height_min": 0.05, "rail_height_max": 0.40, "rail_thickness": 0.10},
    "stones":  {"stone_gap_min": 0.20, "stone_gap_max": 0.55, "stone_len": 0.3, "num_stones": 2},
    "step_up": {"step_height_min": 0.05, "step_height_max": 0.60},
}
D = 0.70  # Hard, representative
specs = [ctg.ObstacleSpec(kind=k, params=dict(_DIFF_RANGES[k])) for k in ctg.OBSTACLE_ORDER]  # one cycle
mesh, end_x, total = ctg.build_course_mesh(specs=specs, difficulty=D)
TZ = ctg.TRACK_Z
print("mesh: verts %s  total_len %.2f  obstacle ends %s" % (np.asarray(mesh.vertices).shape, total, [round(e,2) for e in end_x]))

sec = mesh.section(plane_origin=(0.0, 0.0, 0.0), plane_normal=(0.0, 1.0, 0.0))
V = np.asarray(sec.vertices)

fig, ax = plt.subplots(figsize=(13, 2.5))
for e in sec.entities:
    pts = V[np.asarray(e.points)]
    ax.plot(pts[:, 0], pts[:, 2] - TZ, color="#222", lw=0.7, solid_capstyle="round", zorder=3)

spans = [("hurdle\n(crawl-under)", 2.0, 2.5), ("slope", 3.7, 5.3), ("stairs", 6.5, 8.0),
         ("rail", 9.2, 10.2), ("stepping\nstones", 11.4, 13.2), ("step-up", 14.4, 15.4)]
ztop = float((V[:, 2] - TZ).max())
for name, x0, x1 in spans:
    ax.axvspan(x0, x1, color="#fff3cd", alpha=0.5, zorder=0)
    ax.text(0.5*(x0+x1), ztop + 0.10, name, ha="center", va="bottom", fontsize=7.5, color="#555", linespacing=0.9)
ax.axhline(0, color="#bbb", lw=0.5, ls=":", zorder=1)  # track datum

ax.set_xlim(1.2, 16.9)
ax.set_ylim((V[:, 2] - TZ).min() - 0.08, ztop + 0.55)
ax.set_xlabel("distance along course (m) — robot enters from the left", fontsize=8.5)
ax.set_ylabel("height above\ntrack (m)", fontsize=8.5)
ax.set_aspect("equal", adjustable="box")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=7.5)
ax.set_title("Obstacle-course centerline profile (difficulty $d=0.70$; the six patches repeat for two cycles, 33.2 m total)", fontsize=9)
plt.tight_layout()
import shutil
plt.savefig("/tmp/course_profile_hard.pdf", bbox_inches="tight")
plt.savefig("/tmp/course_profile_hard.png", dpi=200, bbox_inches="tight")
for d in ["/home/ouge/Desktop/SplitMoE/figures", "/home/ouge/Desktop/SplitMoE_EN/figures"]:
    shutil.copy("/tmp/course_profile_hard.pdf", d + "/course_profile_hard.pdf")
print("[saved] /tmp/course_profile_hard.{pdf,png} + copied to both papers")
plt.close(fig)
