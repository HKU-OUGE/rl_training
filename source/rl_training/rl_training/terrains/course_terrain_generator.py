"""Linear obstacle-course terrain generators for the M20 eval.

Two generators live in this module:

1. ``LinearCourseTerrainGenerator_v1_pyramidstyle``: the v1 implementation that
   reused Isaac Lab pyramid sub-terrains (6 sub_terrains × 7.5m each = 45m).
   Kept around for reference / reproducibility of older eval runs.

2. ``MeshLinearObstacleCourseGenerator``: the v2 implementation. Builds one
   contiguous trimesh course (~30m total) directly from short obstacle
   primitives that mirror the MuJoCo deploy terrain
   ``generate_mujoco_terrain.py``:

       hurdle → slope → stairs → rail → stones → step_up  (× 2 cycles = 12 patches)

   Each obstacle is followed by a 1.5m flat platform buffer. Difficulty scales
   each parameter linearly between ``(min_value @ d=0, max_value @ d=1)``.

Both generators behave like ``isaaclab.terrains.TerrainGenerator`` from the
perspective of ``TerrainImporter``: they expose ``terrain_mesh``,
``terrain_origins``, and ``flat_patches``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Any

import numpy as np
import trimesh

from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


# ============================================================================
# v1 (kept for backward compat reference; not used by current cfgs)
# ============================================================================


class LinearCourseTerrainGenerator_v1_pyramidstyle(TerrainGenerator):
    """v1: dict-ordered sub_terrains as rows, fixed difficulty.

    Each row uses sub_terrains[r] (in dict order) at fixed
    difficulty=cfg.course_difficulty. num_cols must be 1.
    """

    def _generate_curriculum_terrains(self):
        if self.cfg.num_cols != 1:
            raise ValueError(
                f"LinearCourseTerrainGenerator requires num_cols=1, got {self.cfg.num_cols}."
            )
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        if self.cfg.num_rows != len(sub_terrains_cfgs):
            raise ValueError(
                f"LinearCourseTerrainGenerator requires num_rows == len(sub_terrains); "
                f"got num_rows={self.cfg.num_rows}, num_sub_terrains={len(sub_terrains_cfgs)}."
            )
        difficulty = float(self.cfg.course_difficulty)
        for sub_row in range(self.cfg.num_rows):
            sub_cfg = sub_terrains_cfgs[sub_row]
            mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
            self._add_sub_terrain(mesh, origin, sub_row, 0, sub_cfg)

    def _generate_random_terrains(self):
        self._generate_curriculum_terrains()


# ============================================================================
# v2: MeshLinearObstacleCourseGenerator
# ============================================================================
#
# Constants for the v2 course geometry. The corridor width MUST be ≥ 3m so the
# 1.5m-wide eval lane (|y - env_origin.y| < 1.5) fits with margin. The track
# surface sits at z=TRACK_Z above the world origin so pits (negative z drop)
# don't immediately fall through the world.
# ----------------------------------------------------------------------------

CORRIDOR_WIDTH = 3.0          # m, full y extent of the obstacle mesh
LEAD_IN_LENGTH = 2.0          # m, flat spawn pad in front of the first obstacle
TAIL_OUT_LENGTH = 2.0         # m, flat goal pad after the last obstacle
BUFFER_LENGTH = 1.2           # m, flat platform after EACH obstacle
TRACK_Z = 0.5                 # m, height of the track top surface above world z=0
# Robot spawns this far into the lead-in (object-frame x). The whole footprint
# must sit on the track slab. This is the SINGLE source of truth — referenced
# by the terrain origin_local, the eval env_origins override, and the progress
# measurement (disp is measured in OBJECT frame = world_x - env_origin_x +
# SPAWN_INSET, so disp ∈ [SPAWN_INSET, COURSE_LENGTH] and the reach_goal
# threshold COURSE_LENGTH is the mesh end — physically reachable).
SPAWN_INSET = 0.5
FLOOR_THICKNESS = 0.05        # m, thickness of the thin under-track floor slab
TRACK_THICKNESS = 0.5         # m, thickness of the full track slab (matches TRACK_Z)


# Per-obstacle slot lengths (along x), excluding the trailing BUFFER_LENGTH
# buffer. These determine total course length. Budgeted to land the full
# 12-obstacle course at ~30 m:
#     2 cycles × (0.5+1.6+1.5+1.0+1.2+1.0) = 2 × 6.8 = 13.6 m obstacles
#     + 12 × 1.2 buffer + 2 + 2 lead/tail   = 14.4 + 4.0 = 18.4 m flat
#     = 32.0 m total                       (within ±10% of 30 m).
OBSTACLE_LENGTHS = {
    "hurdle":  0.5,
    "slope":  1.6,   # 0.6 ramp_up + 0.4 platform + 0.6 ramp_down
    "stairs": 1.5,   # 3 steps × 0.2 up + 0.3 platform + 3 × 0.2 down
    "rail":   1.0,
    # Stones must accommodate `num_stones × stone_len + (num_stones-1) × gap`.
    # With num_stones=3, stone_len=0.3, gap up to 0.60 → 0.9 + 1.2 = 2.1 max.
    # Pick 1.8 so even Extreme (gap≈0.575) fits without saturating to a stone
    # min (prev 1.2 caused Hard/Extreme to clamp to identical gap=0.375).
    "stones": 1.8,
    "step_up": 1.0,  # raised platform (renamed from `pit`; geometry: step-up box)
}

OBSTACLE_ORDER = ["hurdle", "slope", "stairs", "rail", "stones", "step_up"]
NUM_CYCLES = 2


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    """Create an axis-aligned box mesh centred at `center`."""
    mesh = trimesh.creation.box(extents=list(extents))
    mesh.apply_translation(list(center))
    return mesh


def _flat_section(length_x: float, top_z: float = TRACK_Z) -> trimesh.Trimesh:
    """A flat segment of the track: solid slab from z=0 to z=top_z.

    Centred at x=length_x/2 (callers translate later).
    """
    return _box(
        extents=(length_x, CORRIDOR_WIDTH, top_z),
        center=(length_x / 2.0, 0.0, top_z / 2.0),
    )


@dataclass
class ObstacleSpec:
    """One entry in the course sequence."""

    kind: str                          # one of OBSTACLE_ORDER
    params: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Per-obstacle mesh builders. Each returns (mesh, length_x).
# ----------------------------------------------------------------------------
#
# Geometry convention: builders place obstacles between x=0 and x=length_x, with
# y=0 the corridor centre. Z=0 is the world floor; the track top is at TRACK_Z.
# The caller translates the whole obstacle to the correct course-frame x.

def _make_hurdle(difficulty: float, hurdle_height: float, bar_thickness: float = 0.08) -> tuple[trimesh.Trimesh, float]:
    """Floating horizontal bar with two side posts (no base box — robot crosses on
    the flat buffers before/after). length_x along x ≈ 0.5m.

    The corridor floor is NOT included here (the lead-in / buffer slabs cover
    the full x extent so the robot can walk under).
    """
    length_x = OBSTACLE_LENGTHS["hurdle"]
    H = float(hurdle_height)
    bt = float(bar_thickness)          # x-direction (travel-axis thickness)
    bar_vert = 0.30                    # z-direction (vertical board height) —
                                       # widened from `bt` so the elevation
                                       # scan / lidar has more vertical signal
                                       # to detect the bar.
    meshes = []
    # Floor slab spanning the hurdle gap (so robot floor is continuous).
    meshes.append(_flat_section(length_x))

    cx = length_x / 2.0
    # Bar BOTTOM stays at TRACK_Z + H (so clearance = H is preserved).
    # Bar centre therefore at TRACK_Z + H + bar_vert/2.
    bar_z = TRACK_Z + H + bar_vert / 2.0
    # Horizontal bar across the corridor (full y).
    meshes.append(_box((bt, CORRIDOR_WIDTH, bar_vert), (cx, 0.0, bar_z)))
    # Two side posts (inside corridor edges so they don't add to width).
    post_y = CORRIDOR_WIDTH / 2.0 - bt / 2.0
    meshes.append(_box((bt, bt, H), (cx, +post_y, TRACK_Z + H / 2.0)))
    meshes.append(_box((bt, bt, H), (cx, -post_y, TRACK_Z + H / 2.0)))

    return trimesh.util.concatenate(meshes), length_x


def _make_slope(difficulty: float, slope: float) -> tuple[trimesh.Trimesh, float]:
    """Two short ramps + middle platform. Translation of MuJoCo add_slope but
    sized to fit the v2 budget.

    Mesh approach: instead of tilted boxes (which need rotation), we build the
    ramp as a wedge prism using vertices/faces directly. This keeps the mesh
    water-tight to the track surface.

    length_x = 1.6 m: 0.6m ramp up + 0.4m platform + 0.6m ramp down.
    """
    ramp_len = 0.6
    plat_len = 0.4
    length_x = 2 * ramp_len + plat_len  # = 1.6
    rise = ramp_len * float(slope)
    plat_top_z = TRACK_Z + rise

    meshes = []
    # Up ramp wedge (x in [0, ramp_len]): bottom at z=0, top surface tilts up.
    # We approximate the wedge as a tilted box (simpler than custom prism).
    # Use a thin box rotated about y-axis. Then add the track slab below it as
    # the floor block so it's water-tight.
    # Actually, the cleanest robust approach: use the prism via convex hull of
    # 8 vertices that form a wedge (rectangular base of [0..ramp_len, -W/2..W/2]
    # × z=0, top sloped from TRACK_Z to plat_top_z).
    W = CORRIDOR_WIDTH

    def _wedge(x0, x1, z0_low, z1_low, z0_high, z1_high):
        """Quadrilateral prism between x0/x1, with bottom z(x0)=z0_low,
        z(x1)=z1_low and top z(x0)=z0_high, z(x1)=z1_high; full width in y.

        Vertex layout (CCW when viewed from outside):
          0..3 bottom (z=*_low),  4..7 top (z=*_high).
          0=(x0,-y), 1=(x1,-y), 2=(x1,+y), 3=(x0,+y).
        Each face winding chosen so the outward normal points OUTSIDE the prism.
        """
        verts = np.array([
            [x0, -W/2, z0_low],   # 0
            [x1, -W/2, z1_low],   # 1
            [x1,  W/2, z1_low],   # 2
            [x0,  W/2, z0_low],   # 3
            [x0, -W/2, z0_high],  # 4
            [x1, -W/2, z1_high],  # 5
            [x1,  W/2, z1_high],  # 6
            [x0,  W/2, z0_high],  # 7
        ])
        # Outward normals: bottom face normal -z, top normal +z, -y side normal -y,
        # +y side normal +y, x=x0 side normal -x, x=x1 side normal +x.
        faces = np.array([
            [0, 2, 1], [0, 3, 2],   # bottom: normal -z (CW from above = CCW from below)
            [4, 5, 6], [4, 6, 7],   # top: normal +z
            [0, 1, 5], [0, 5, 4],   # -y side: normal -y
            [3, 6, 2], [3, 7, 6],   # +y side: normal +y
            [0, 4, 7], [0, 7, 3],   # x=x0 side: normal -x
            [1, 2, 6], [1, 6, 5],   # x=x1 side: normal +x
        ])
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Up ramp (x in [0, ramp_len])
    meshes.append(_wedge(0.0, ramp_len, 0.0, 0.0, TRACK_Z, plat_top_z))
    # Middle flat top (x in [ramp_len, ramp_len+plat_len])
    meshes.append(_box((plat_len, W, plat_top_z),
                       (ramp_len + plat_len/2.0, 0.0, plat_top_z/2.0)))
    # Down ramp (x in [ramp_len+plat_len, length_x])
    x0 = ramp_len + plat_len
    x1 = length_x
    meshes.append(_wedge(x0, x1, 0.0, 0.0, plat_top_z, TRACK_Z))

    return trimesh.util.concatenate(meshes), length_x


def _make_stairs(difficulty: float, step_rise: float, step_run: float = 0.2,
                 num_steps: int = 3) -> tuple[trimesh.Trimesh, float]:
    """Short up-stairs + middle platform + down-stairs.

    Each step extends from z=0 (track base) up to track top + per-step rise.
    Mimics MuJoCo add_stairs but with smaller defaults so the obstacle fits
    OBSTACLE_LENGTHS["stairs"] = 1.5m (3 steps × 0.2 up + 0.3 platform + 3
    steps × 0.2 down).
    """
    plat_len = 0.3
    # Make total length match OBSTACLE_LENGTHS["stairs"]
    total_steps_len = OBSTACLE_LENGTHS["stairs"] - plat_len  # 1.2
    actual_run = total_steps_len / (2 * num_steps)  # = 0.2m per step
    length_x = OBSTACLE_LENGTHS["stairs"]
    W = CORRIDOR_WIDTH

    meshes = []
    # Up stairs: each step's top surface is at TRACK_Z + (i+1)*step_rise.
    for i in range(num_steps):
        top_z = TRACK_Z + (i + 1) * step_rise
        x_lo = i * actual_run
        x_hi = (i + 1) * actual_run
        meshes.append(_box(
            extents=(x_hi - x_lo, W, top_z),
            center=((x_lo + x_hi)/2.0, 0.0, top_z/2.0),
        ))
    # Middle platform.
    plat_top_z = TRACK_Z + num_steps * step_rise
    plat_x_lo = num_steps * actual_run
    plat_x_hi = plat_x_lo + plat_len
    meshes.append(_box(
        extents=(plat_len, W, plat_top_z),
        center=((plat_x_lo + plat_x_hi)/2.0, 0.0, plat_top_z/2.0),
    ))
    # Down stairs.
    for i in range(num_steps):
        top_z = TRACK_Z + (num_steps - i - 1) * step_rise + step_rise  # i=0 → num_steps*step_rise
        # Simpler: descending levels
        top_z = TRACK_Z + (num_steps - 1 - i) * step_rise + step_rise
        # Even simpler: down stairs height at step j (0..num_steps-1) = (num_steps-j)*step_rise
        top_z = TRACK_Z + (num_steps - i) * step_rise
        x_lo = plat_x_hi + i * actual_run
        x_hi = x_lo + actual_run
        # Last step should end at TRACK_Z, so top_z at i=num_steps-1 = TRACK_Z+step_rise
        meshes.append(_box(
            extents=(x_hi - x_lo, W, top_z),
            center=((x_lo + x_hi)/2.0, 0.0, top_z/2.0),
        ))

    return trimesh.util.concatenate(meshes), length_x


def _make_rail(difficulty: float, rail_height: float, rail_thickness: float = 0.1) -> tuple[trimesh.Trimesh, float]:
    """Thin centred rail along x.

    Floor slab spans the full obstacle length so the robot has somewhere to
    walk beside the rail. The rail itself is a thin box centred on y=0.

    length_x = 1.5m.
    """
    length_x = OBSTACLE_LENGTHS["rail"]
    meshes = []
    meshes.append(_flat_section(length_x))
    # Rail: single PERPENDICULAR bar (across full corridor) at the centre of the
    # obstacle slot. Robot must step / jump over. Previously oriented along x —
    # which let the robot just walk past it on either side without engaging.
    rh = float(rail_height)
    rt = float(rail_thickness)
    meshes.append(_box(
        extents=(rt, CORRIDOR_WIDTH, rh),
        center=(length_x/2.0, 0.0, TRACK_Z + rh/2.0),
    ))
    return trimesh.util.concatenate(meshes), length_x


def _make_stones(difficulty: float, stone_gap: float, stone_len: float = 0.3,
                 num_stones: int = 3) -> tuple[trimesh.Trimesh, float]:
    """Sequence of stepping stones separated by gaps.

    The stones each sit at TRACK_Z; gaps between them are EMPTY (the under-floor
    safety slab catches any robot that falls in). The first stone starts at
    x=stone_len/2 (so there's a small gap from the lead-in buffer too).

    length_x ≈ num_stones*stone_len + (num_stones-1)*stone_gap. We size to
    OBSTACLE_LENGTHS["stones"] = 1.8m by computing the actual stone_len.
    """
    target_len = OBSTACLE_LENGTHS["stones"]
    # Solve: num_stones*stone_len + (num_stones-1)*stone_gap = target_len
    # → stone_len = (target_len - (num_stones-1)*stone_gap) / num_stones
    sl = (target_len - (num_stones - 1) * stone_gap) / num_stones
    # Guard against very large gaps that would make stones too small.
    # In that case we shrink the gap instead so length_x stays = target_len.
    min_sl = 0.15
    if sl < min_sl:
        sl = min_sl
        stone_gap = max(0.05, (target_len - num_stones * sl) / (num_stones - 1))
    length_x = target_len  # always exactly OBSTACLE_LENGTHS["stones"]
    W = CORRIDOR_WIDTH

    meshes = []
    # Under-floor catch slab at z=-0.5 (so falls register course_below_ground
    # when really deep, but here we want them to recover; keep the slab as a
    # safety net so the robot doesn't escape the world).
    meshes.append(_box(
        extents=(length_x, W, FLOOR_THICKNESS),
        center=(length_x/2.0, 0.0, FLOOR_THICKNESS/2.0),
    ))
    x_cur = 0.0
    for i in range(num_stones):
        meshes.append(_box(
            extents=(sl, W, TRACK_Z - FLOOR_THICKNESS),
            center=(x_cur + sl/2.0, 0.0, (FLOOR_THICKNESS + TRACK_Z)/2.0),
        ))
        x_cur += sl
        if i < num_stones - 1:
            x_cur += stone_gap

    return trimesh.util.concatenate(meshes), length_x


def _make_pit(difficulty: float, pit_depth: float, gap_length: float = 0.4) -> tuple[trimesh.Trimesh, float]:
    """Single raised platform after a small gap.

    Mimics MuJoCo add_pit with double_pit=False, num_pits=1: small gap
    (gap_length, no track surface, only the under-floor slab) + 1m raised
    platform of total height pit_depth (positive = obstacle on top of the
    track, negative is not supported here).

    length_x = gap_length + 1.0 m platform (matches OBSTACLE_LENGTHS["pit"] when
    gap_length is small). To keep total length budget, we tune gap_length so
    gap + plat = OBSTACLE_LENGTHS["pit"] = 1.0; default gap_length=0.0 (no gap,
    just a high step). The MuJoCo "pit" was actually a raised box on the floor,
    so we mirror that: it's a step-up obstacle.
    """
    plat_len = 1.0
    length_x = plat_len  # ignore gap_length in v2 budget (gap covered by buffer)
    W = CORRIDOR_WIDTH
    meshes = []
    # Thin under-floor (so the gap before platform is still walkable surface).
    meshes.append(_box(
        extents=(length_x, W, FLOOR_THICKNESS),
        center=(length_x/2.0, 0.0, FLOOR_THICKNESS/2.0),
    ))
    # Pit/platform: solid box sitting on the floor slab, of TOTAL height
    # (TRACK_Z - FLOOR_THICKNESS) + pit_depth. Top surface at TRACK_Z + pit_depth.
    pd = float(pit_depth)
    h = (TRACK_Z - FLOOR_THICKNESS) + pd
    meshes.append(_box(
        extents=(plat_len, W, h),
        center=(plat_len/2.0, 0.0, FLOOR_THICKNESS + h/2.0),
    ))
    return trimesh.util.concatenate(meshes), length_x


# ----------------------------------------------------------------------------
# Difficulty-parameter mapping. Linear interp between EASY (d=0) and EXTREME
# (d=1) bounds; passed obstacle specs select named keys per obstacle kind.
# ----------------------------------------------------------------------------


def _interp(d: float, lo: float, hi: float) -> float:
    return float(lo + d * (hi - lo))


def _build_obstacle(spec: ObstacleSpec, difficulty: float) -> tuple[trimesh.Trimesh, float]:
    """Dispatch one ObstacleSpec to the matching builder."""
    kind = spec.kind
    p = spec.params
    if kind == "hurdle":
        # Crawl-mode hurdle: bar height = clearance under the bar. Higher bar =
        # easier to crouch under. Invert the difficulty so d=0 → tall (easy),
        # d=1 → short (hard). The min cap of 0.20 m matches M20's physical
        # squat limit so even Extreme stays physically passable.
        h = _interp(1.0 - difficulty, p["hurdle_height_min"], p["hurdle_height_max"])
        return _make_hurdle(difficulty, h, bar_thickness=p.get("bar_thickness", 0.08))
    elif kind == "slope":
        s = _interp(difficulty, p["slope_min"], p["slope_max"])
        return _make_slope(difficulty, s)
    elif kind == "stairs":
        r = _interp(difficulty, p["step_rise_min"], p["step_rise_max"])
        return _make_stairs(difficulty, r,
                            step_run=p.get("step_run", 0.2),
                            num_steps=p.get("num_steps", 3))
    elif kind == "rail":
        rh = _interp(difficulty, p["rail_height_min"], p["rail_height_max"])
        rt = p.get("rail_thickness", 0.10)
        return _make_rail(difficulty, rh, rail_thickness=rt)
    elif kind == "stones":
        g = _interp(difficulty, p["stone_gap_min"], p["stone_gap_max"])
        return _make_stones(difficulty, g,
                            stone_len=p.get("stone_len", 0.3),
                            num_stones=p.get("num_stones", 3))
    elif kind == "step_up":
        d = _interp(difficulty, p["step_height_min"], p["step_height_max"])
        return _make_pit(difficulty, d, gap_length=p.get("gap_length", 0.0))
    else:
        raise ValueError(f"Unknown obstacle kind: {kind}")


# ----------------------------------------------------------------------------
# Top-level course builder
# ----------------------------------------------------------------------------


def build_course_mesh(specs: list[ObstacleSpec], difficulty: float
                      ) -> tuple[trimesh.Trimesh, list[float], float]:
    """Assemble the full course mesh and per-obstacle end-x positions.

    Layout along x (in the OBJECT frame, not yet centred):
        [lead_in 2m] + for each obstacle: [obs_len] + [buffer 1.5m] + [tail 2m]

    Returns:
        mesh: concatenated trimesh of the entire course (corridor surface)
        end_x: cumulative x at the end of each obstacle slot (obstacle_end +
               buffer), measured from x=0 = course start. len == len(specs).
        total_len: total course length along x.
    """
    meshes: list[trimesh.Trimesh] = []
    # Lead-in flat
    meshes.append(_flat_section(LEAD_IN_LENGTH))
    x_cur = LEAD_IN_LENGTH

    end_x: list[float] = []

    for spec in specs:
        obs_mesh, obs_len = _build_obstacle(spec, difficulty)
        obs_mesh = obs_mesh.copy()
        obs_mesh.apply_translation([x_cur, 0.0, 0.0])
        meshes.append(obs_mesh)
        x_cur += obs_len

        # Buffer flat after each obstacle
        buf_mesh = _flat_section(BUFFER_LENGTH)
        buf_mesh.apply_translation([x_cur, 0.0, 0.0])
        meshes.append(buf_mesh)
        x_cur += BUFFER_LENGTH

        end_x.append(x_cur)

    # Tail flat (goal pad)
    tail = _flat_section(TAIL_OUT_LENGTH)
    tail.apply_translation([x_cur, 0.0, 0.0])
    meshes.append(tail)
    x_cur += TAIL_OUT_LENGTH

    full = trimesh.util.concatenate(meshes)
    return full, end_x, x_cur


# ----------------------------------------------------------------------------
# Isaac Lab TerrainGenerator adapter
# ----------------------------------------------------------------------------


class MeshLinearObstacleCourseGenerator(TerrainGenerator):
    """Isaac Lab TerrainGenerator that builds a single contiguous course mesh.

    Overrides ``__init__`` to bypass the TerrainGenerator sub_terrains loop
    (we don't have row-tileable sub-terrains here; the course is a single
    custom mesh that already spans the full course length).
    """

    def __init__(self, cfg: "MeshLinearObstacleCourseGeneratorCfg", device: str = "cpu"):
        # Don't call super().__init__: we build the mesh and origins from scratch.
        self.cfg = cfg
        self.device = device
        # Use the cfg seed (or numpy state seed) for reproducibility
        seed = cfg.seed if cfg.seed is not None else np.random.get_state()[1][0]
        self.np_rng = np.random.default_rng(seed)
        # Required attributes for TerrainImporter
        self.flat_patches: dict = {}
        self.terrain_meshes: list = []  # unused but kept for API compat
        self.terrain_origins = np.zeros((1, 1, 3))

        # Build the course mesh
        mesh, end_x, total_len = build_course_mesh(
            specs=cfg.obstacle_specs,
            difficulty=float(cfg.course_difficulty),
        )

        # Cache computed quantities so cfgs / eval scripts can read them.
        self._course_length: float = float(total_len)
        self._obstacle_end_x: list[float] = list(end_x)

        # Validate that the configured size matches: TerrainImporter centres
        # using cfg.size[0]*num_rows/2 — we want the mesh centred at world x=0
        # so the course spans x ∈ [-total_len/2, +total_len/2]. We override
        # cfg.size to (total_len, CORRIDOR_WIDTH) so the centring math works
        # out without the caller having to know total_len in advance.
        cfg.size = (total_len, CORRIDOR_WIDTH)
        cfg.num_rows = 1
        cfg.num_cols = 1

        # Origin (in the course mesh frame): 0.5m into the lead-in. Spawning
        # exactly at x=0 puts the robot's body straddling the lead-in/border
        # boundary (half on track, half off). Shifting by SPAWN_INSET keeps the
        # whole footprint on the lead-in slab. COURSE_LENGTH measured downstream
        # already excludes the inset, since `x_disp = root_pos_w.x - env_origin.x`.
        origin_local = np.array([SPAWN_INSET, 0.0, TRACK_Z])
        self.terrain_origins[0, 0] = origin_local

        # Apply the standard TerrainGenerator centring transform: shift mesh
        # by -size[0]*num_rows/2 in x (and y/2 in y) so the mesh is centred.
        transform = np.eye(4)
        transform[0, 3] = -total_len * 0.5
        # We already built corridor centred at y=0, so no y shift needed; but
        # to mimic Isaac Lab behaviour the standard transform is -size[1]/2.
        # Our build places y in [-W/2, +W/2] (already centred), so set y shift
        # so it stays centred. With num_cols=1 and size[1]=W, Isaac Lab would
        # shift by -W/2. To compensate we pre-add +W/2 then let the standard
        # transform subtract it. Simpler: just translate mesh by -total_len/2
        # in x and leave y as-is. We then assign terrain_origins to the
        # post-transform position.
        mesh.apply_transform(transform)
        # terrain_origins must be in post-transform coordinates too
        self.terrain_origins[0, 0, 0] += transform[0, 3]

        # Add the surrounding border slab (so the mesh has a clear ground
        # outside the corridor — prevents robots that get pushed sideways from
        # falling out of the world). Built in post-transform coordinates so it
        # sits flush with the centred corridor mesh.
        border_mesh = self._build_border_v2(total_len)
        if border_mesh is not None:
            mesh = trimesh.util.concatenate([mesh, border_mesh])

        # Color scheme handling (mimic parent for compatibility)
        self.terrain_mesh = mesh
        if cfg.color_scheme == "height":
            from isaaclab.terrains.utils import color_meshes_by_height
            self.terrain_mesh = color_meshes_by_height(self.terrain_mesh)
        elif cfg.color_scheme == "random":
            self.terrain_mesh.visual.vertex_colors = self.np_rng.choice(
                range(256), size=(len(self.terrain_mesh.vertices), 4)
            )

    def _build_border_v2(self, total_len: float) -> trimesh.Trimesh | None:
        """Build a flat ground around the corridor (so the world has a floor
        outside the obstacle course). Returns the border mesh in
        post-centring (world) coordinates: corridor centred at x=0, y=0.
        """
        bw = self.cfg.border_width
        if bw <= 0.0:
            return None
        bh = self.cfg.border_height
        W = CORRIDOR_WIDTH

        # Border is at world z in [0, bh] (sits at world floor, well below
        # track top at z=TRACK_Z=0.5 — so robots that fall off the corridor
        # land on the border).
        full_x = total_len + 2 * bw
        half_x = total_len / 2.0  # corridor extends from -half_x to +half_x
        meshes = []
        # +y strip
        meshes.append(_box(
            extents=(full_x, bw, bh),
            center=(0.0, W/2.0 + bw/2.0, bh/2.0),
        ))
        # -y strip
        meshes.append(_box(
            extents=(full_x, bw, bh),
            center=(0.0, -W/2.0 - bw/2.0, bh/2.0),
        ))
        # +x strip (in front of goal)
        meshes.append(_box(
            extents=(bw, W, bh),
            center=(half_x + bw/2.0, 0.0, bh/2.0),
        ))
        # -x strip (behind spawn)
        meshes.append(_box(
            extents=(bw, W, bh),
            center=(-half_x - bw/2.0, 0.0, bh/2.0),
        ))
        return trimesh.util.concatenate(meshes)

    @property
    def course_length(self) -> float:
        return self._course_length

    @property
    def obstacle_end_x(self) -> list[float]:
        return list(self._obstacle_end_x)


@configclass
class MeshLinearObstacleCourseGeneratorCfg(TerrainGeneratorCfg):
    """Cfg for the v2 obstacle course generator.

    Note ``size`` and ``num_rows`` / ``num_cols`` are FILLED IN automatically by
    ``MeshLinearObstacleCourseGenerator.__init__`` (after the mesh is built and
    the total length is known). Callers should not rely on them at cfg time.
    """

    class_type: type = MeshLinearObstacleCourseGenerator

    course_difficulty: float = MISSING
    """Difficulty in [0, 1] used to interpolate every obstacle's parameter range."""

    obstacle_specs: list = MISSING
    """Ordered list of ObstacleSpec describing the course sequence."""

    # Required by parent but unused here. Set to a dummy non-empty dict so the
    # base class's MISSING check doesn't trip if we ever call super().__init__.
    sub_terrains: dict = field(default_factory=lambda: {"course": None})

    size: tuple[float, float] = (1.0, CORRIDOR_WIDTH)  # overwritten by generator
    num_rows: int = 1
    num_cols: int = 1
    border_width: float = 5.0
    border_height: float = 0.05
    curriculum: bool = True

    # Disable use_cache by default (the mesh is cheap to rebuild and the cache
    # key wouldn't capture the obstacle list properly).
    use_cache: bool = False


# Backward-compat alias: existing v1 cfg name still importable
LinearCourseTerrainGenerator = LinearCourseTerrainGenerator_v1_pyramidstyle


@configclass
class LinearCourseTerrainGeneratorCfg(TerrainGeneratorCfg):
    """v1 cfg (kept for backward compat)."""

    class_type: type = LinearCourseTerrainGenerator_v1_pyramidstyle
    course_difficulty: float = MISSING
    curriculum: bool = True
    num_cols: int = 1
