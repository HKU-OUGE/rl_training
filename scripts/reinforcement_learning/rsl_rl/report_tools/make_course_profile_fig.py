"""Polished course side-elevation via ray-cast top-surface profile. Local (isaaclab stubbed)."""
import sys, types, importlib.util, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Circle, FancyBboxPatch

for m in ["isaaclab", "isaaclab.terrains", "isaaclab.terrains.terrain_generator",
          "isaaclab.terrains.terrain_generator_cfg", "isaaclab.utils"]:
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["isaaclab.terrains.terrain_generator"].TerrainGenerator = object
sys.modules["isaaclab.terrains.terrain_generator_cfg"].TerrainGeneratorCfg = object
sys.modules["isaaclab.utils"].configclass = (lambda c: c)
GEN = "/home/ouge/Software/rl_training/source/rl_training/rl_training/terrains/course_terrain_generator.py"
spec = importlib.util.spec_from_file_location("ctg", GEN); ctg = importlib.util.module_from_spec(spec)
sys.modules["ctg"] = ctg; spec.loader.exec_module(ctg)
_DR = {"hurdle": {"hurdle_height_min": 0.35, "hurdle_height_max": 0.55, "bar_thickness": 0.08},
       "slope": {"slope_min": 0.10, "slope_max": 0.55},
       "stairs": {"step_rise_min": 0.05, "step_rise_max": 0.24, "step_run": 0.2, "num_steps": 3},
       "rail": {"rail_height_min": 0.05, "rail_height_max": 0.40, "rail_thickness": 0.10},
       "stones": {"stone_gap_min": 0.20, "stone_gap_max": 0.55, "stone_len": 0.3, "num_stones": 2},
       "step_up": {"step_height_min": 0.05, "step_height_max": 0.60}}
D, TZ = 0.70, ctg.TRACK_Z
specs = [ctg.ObstacleSpec(kind=k, params=dict(_DR[k])) for k in ctg.OBSTACLE_ORDER]
mesh, end_x, total = ctg.build_course_mesh(specs=specs, difficulty=D)

# --- ray-cast straight down at a fine x-grid (y=0) to get solid intervals per x ---
xs = np.linspace(0.05, total - 0.05, 520)
origins = np.c_[xs, np.zeros_like(xs), np.full_like(xs, 3.0)]
dirs = np.tile([0.0, 0.0, -1.0], (len(xs), 1))
loc, iray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
hits = {i: [] for i in range(len(xs))}
for p, ir in zip(loc, iray):
    hits[ir].append(p[2] - TZ)   # display z = world z - TRACK_Z

# top surface = highest hit per x. Only the hurdle floats (known x-range): there the
# ground is the track top (hits below 0.3) and the bar is the hits above.
HURDLE = (1.95, 2.55)
ground_top = np.full(len(xs), np.nan)
bx, blo, bhi = [], [], []
for i in range(len(xs)):
    z = hits[i]
    if not z:
        continue
    if HURDLE[0] <= xs[i] <= HURDLE[1]:
        below = [v for v in z if v < 0.30]
        ground_top[i] = max(below) if below else 0.0
        above = [v for v in z if v >= 0.30]
        if above:
            bx.append(xs[i]); blo.append(min(above)); bhi.append(max(above))
    else:
        ground_top[i] = max(z)
ok = ~np.isnan(ground_top)
ground_top = np.interp(xs, xs[ok], ground_top[ok])
bar_pts = list(zip(bx, blo, bhi))

# ---- plot ----
GROUND, GEDGE = "#cdbb98", "#6e5d45"
BAR, BAREDGE = "#c0392b", "#7d241a"
VOID, ROBOT = "#bcd6e6", "#33475b"
THIN, VOID_D = -0.15, -0.32          # thin ground slab below track; displayed void depth
sl = (1.8 - 0.45) / 2.0; vx0 = 11.4 + sl; vx1 = vx0 + 0.45   # stepping-stone void x-range
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(13, 2.5))

# thin solid ground slab (track) + obstacles on top
top_g = np.maximum(ground_top, THIN)
ax.fill_between(xs, THIN, top_g, facecolor=GROUND, edgecolor="none", zorder=2)
ax.plot(xs, np.where(ground_top < THIN - 1e-3, np.nan, ground_top), color=GEDGE, lw=0.8,
        zorder=4, solid_capstyle="round")                       # top contour (skips the void)
ax.plot([xs[0], xs[-1]], [THIN, THIN], color=GEDGE, lw=0.8, zorder=4)   # slab underside
# stepping-stone void: a pit cut below the track
ax.add_patch(MplPoly([(vx0, 0), (vx0, VOID_D), (vx1, VOID_D), (vx1, 0)], closed=True,
                     facecolor=VOID, edgecolor=GEDGE, lw=0.7, zorder=3))
ax.axhline(0, color="#9a8f78", lw=0.6, ls=(0, (4, 3)), zorder=2)   # track datum

# crawl-under hurdle bar (red) from the floating ray hits
if bar_pts:
    bx = np.array([p[0] for p in bar_pts]); blo = min(p[1] for p in bar_pts); bhi = max(p[2] for p in bar_pts)
    ax.add_patch(MplPoly([(bx.min(), blo), (bx.max(), blo), (bx.max(), bhi), (bx.min(), bhi)],
                         closed=True, facecolor=BAR, edgecolor=BAREDGE, lw=0.9, zorder=6))

# tiny M20 silhouette at spawn (scale)
rx = 1.05
ax.add_patch(FancyBboxPatch((rx - 0.28, 0.14), 0.56, 0.17, boxstyle="round,pad=0.01,rounding_size=0.05",
                            facecolor=ROBOT, edgecolor="none", zorder=8))
for dx in (-0.19, 0.19):
    ax.plot([rx + dx, rx + dx], [0.17, 0.085], color=ROBOT, lw=2.2, zorder=8)
    ax.add_patch(Circle((rx + dx, 0.085), 0.085, facecolor="#22303d", edgecolor=ROBOT, lw=1.0, zorder=9))
ax.text(rx, 0.40, "M20", ha="center", fontsize=6, color=ROBOT, zorder=9)

# obstacle labels
for name, cx in [("hurdle", 2.25), ("slope", 4.5), ("stairs", 7.25), ("rail", 9.7),
                 ("stepping\nstones", 12.3), ("step-up", 14.9)]:
    ax.text(cx, 0.74, name, ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="#3a3a3a", linespacing=0.9, zorder=10)

# dimension callouts
def vdim(x, z0, z1, txt, c="#444"):
    ax.annotate("", (x, z1), (x, z0), arrowprops=dict(arrowstyle="<->", color=c, lw=0.8), zorder=11)
    ax.text(x + 0.12, 0.5 * (z0 + z1), txt, fontsize=6, va="center", color=c, zorder=11)
vdim(2.62, 0.0, 0.41, r"$h_1$")     # hurdle clearance
ax.annotate("", (vx1, -0.16), (vx0, -0.16), arrowprops=dict(arrowstyle="<->", color="#3d6b82", lw=0.8), zorder=11)
ax.text(0.5 * (vx0 + vx1), -0.11, r"$l_1$", fontsize=7, ha="center", va="bottom", color="#3d6b82", zorder=11)
ax.text(0.5 * (vx0 + vx1), -0.30, "void", fontsize=5.5, ha="center", va="top", color="#3d6b82", style="italic", zorder=11)
vdim(15.55, 0.0, 0.44, r"$h_2$")    # step-up height

ax.set_xlim(0.3, 16.9); ax.set_ylim(VOID_D - 0.10, 0.96)
ax.set_xlabel("distance along course (m) — robot enters from the left", fontsize=8.5)
ax.set_ylabel("height above\ntrack (m)", fontsize=8.5)
ax.set_aspect("equal", adjustable="box")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=7.5, length=2); ax.set_yticks([0.0, 0.5])
plt.tight_layout()
plt.savefig("/tmp/course_profile_pretty.pdf", bbox_inches="tight")
plt.savefig("/tmp/course_profile_pretty.png", dpi=200, bbox_inches="tight")
print("[saved]  bars=%d  surf range [%.2f, %.2f]" % (len(bar_pts), np.nanmin(ground_top), np.nanmax(ground_top)))
plt.close(fig)
