"""Obstacle-course terrain configs for the M20 v2 eval.

The v2 course is a single shared corridor mesh (Isaac Lab replicates physics
across all envs) built from short obstacle primitives that mimic the MuJoCo
deploy terrain (``sdk_deploy/.../generate_mujoco_terrain.py``).

Obstacle sequence (fixed order, 2 cycles = 12 obstacles total):
    hurdle → slope → stairs → rail → stones → step_up

For each obstacle: obstacle geometry (≤2 m along x, 3 m corridor width in y),
followed by a 1.2 m flat platform buffer. Plus a 2 m lead-in (spawn pad) and
a 2 m tail-out (goal pad). Total = 32 m. The course is centred at x=0 so the
robot spawn (overridden in eval_course.py to env_origins.x = -16.0) sits at
the start of the lead-in.

Four difficulty levels: difficulty d ∈ {0.30, 0.50, 0.70, 0.95} linearly
interpolates each obstacle parameter between its EASY and EXTREME bounds.
"""

from __future__ import annotations

from rl_training.terrains.course_terrain_generator import (
    BUFFER_LENGTH,
    CORRIDOR_WIDTH,
    LEAD_IN_LENGTH,
    NUM_CYCLES,
    OBSTACLE_LENGTHS,
    OBSTACLE_ORDER,
    ObstacleSpec,
    TAIL_OUT_LENGTH,
    MeshLinearObstacleCourseGeneratorCfg,
)


# ---------------------------------------------------------------------------
# Difficulty parameter ranges (EASY @ d=0 → EXTREME @ d=1).
# These are the global min/max bounds; each level cfg below picks a single
# difficulty value (0.30, 0.50, 0.70, 0.95) that interpolates inside the range.
# ---------------------------------------------------------------------------

_DIFF_RANGES = {
    "hurdle": {
        # CRAWL-mode floating bar. v2.2 range widened back to 0.35-0.55 per
        # user request (Easy 49 cm clearance for visual generosity). Tested
        # together with spawn-inset fix + rail fix + stones-1-gap to see if
        # the bar geometry itself is the dead-zone issue or if it was confounded
        # with other course-design bugs. Inverted mapping: d=0 → 55 cm (easy),
        # d=1 → 35 cm (hard).
        "hurdle_height_min": 0.35,
        "hurdle_height_max": 0.55,
        "bar_thickness": 0.08,
    },
    "slope": {
        "slope_min": 0.10,           # @ d=0:  10% rise/run
        "slope_max": 0.55,           # @ d=1:  55% rise/run
    },
    "stairs": {
        "step_rise_min": 0.05,       # @ d=0:  5 cm steps
        "step_rise_max": 0.24,       # @ d=1:  24 cm steps
        "step_run": 0.2,
        "num_steps": 3,
    },
    "rail": {
        "rail_height_min": 0.05,     # @ d=0:  5 cm rail
        "rail_height_max": 0.40,     # @ d=1:  40 cm rail
        "rail_thickness": 0.10,
    },
    "stones": {
        # v2.2: num_stones 3 → 2 (single gap instead of 2 consecutive gaps),
        # matching MuJoCo deploy-style single-jump. Gap range unchanged so
        # difficulty scaling stays the same — robot now faces one wider jump
        # instead of two narrower ones per stones obstacle.
        # v2.2: gap_min 0.10 → 0.20 so Easy (d=0.30) gives 30 cm gap (was 22 cm),
        # which is wider and more wheel-leg-friendly. Max adjusted so Extreme
        # still hits ~53 cm (near M20 single-step limit but reachable).
        "stone_gap_min": 0.20,       # @ d=0:  20 cm gap   (Easy = 30 cm)
        "stone_gap_max": 0.55,       # @ d=1:  55 cm gap   (Extreme = 53 cm)
        "stone_len": 0.3,
        "num_stones": 2,
    },
    # Renamed from "pit" — the geometry is a RAISED platform on top of the
    # corridor (robot must climb up), not a depression. The previous "pit"
    # naming was misleading and the course_below_ground(z<-2) termination
    # could not fire on this geometry.
    "step_up": {
        # Max tightened 0.70 → 0.60: a 70cm step exceeds M20 standing height
        # (≈50cm) so it was physically unclimbable. 60cm is right at the limit
        # — still hard but reachable in principle for an aggressive jump.
        "step_height_min": 0.05,     # @ d=0:  5 cm step up
        "step_height_max": 0.60,     # @ d=1:  60 cm step up  (Extreme = 57 cm)
    },
}


def _build_specs() -> list[ObstacleSpec]:
    """Build the 12-obstacle list (2 cycles × 6 obstacles)."""
    specs: list[ObstacleSpec] = []
    for _cycle in range(NUM_CYCLES):
        for kind in OBSTACLE_ORDER:
            specs.append(ObstacleSpec(kind=kind, params=dict(_DIFF_RANGES[kind])))
    return specs


# ---------------------------------------------------------------------------
# Course layout constants (consumed by env_cfg and eval_course)
# ---------------------------------------------------------------------------

# Total course length (m): lead-in + Σ(obstacle + buffer) + tail-out.
# = 2 + 2 × (Σ obstacle_lengths + 6 × buffer) + 2
# = 4 + 2 × (0.5+1.6+1.5+1.0+1.2+1.0) + 12 × 1.2
# = 4 + 2 × 6.8 + 14.4
# = 32.0  m
COURSE_LENGTH: float = (
    LEAD_IN_LENGTH
    + TAIL_OUT_LENGTH
    + NUM_CYCLES * sum(OBSTACLE_LENGTHS[k] for k in OBSTACLE_ORDER)
    + NUM_CYCLES * len(OBSTACLE_ORDER) * BUFFER_LENGTH
)

COURSE_NUM_PATCHES: int = NUM_CYCLES * len(OBSTACLE_ORDER)  # = 12

# Per-obstacle end-x (cumulative end of each obstacle's slot = obs_end + buffer),
# measured from the START of the lead-in (object-frame x=0). eval_course measures
# OBJECT-FRAME displacement disp_x = root_pos.x - env_origin.x + SPAWN_INSET,
# which spans [SPAWN_INSET, COURSE_LENGTH] and is directly comparable to these
# thresholds (also object-frame). The robot spawns SPAWN_INSET into the lead-in.
def _compute_patch_end_x() -> list[float]:
    xs: list[float] = []
    cur = LEAD_IN_LENGTH
    for _ in range(NUM_CYCLES):
        for k in OBSTACLE_ORDER:
            cur += OBSTACLE_LENGTHS[k] + BUFFER_LENGTH
            xs.append(round(cur, 6))
    return xs

COURSE_PATCH_END_X: list[float] = _compute_patch_end_x()

# Human-readable patch labels (for plotting / summary.json sub_terrain_names).
# Each cycle appends "_2" so the eval scripts can distinguish them.
PATCH_NAMES: list[str] = []
for cycle in range(NUM_CYCLES):
    suffix = "" if cycle == 0 else "_2"
    for k in OBSTACLE_ORDER:
        PATCH_NAMES.append(f"{k}{suffix}")


# ---------------------------------------------------------------------------
# Cfg builders
# ---------------------------------------------------------------------------


def _make_course_cfg(difficulty: float) -> MeshLinearObstacleCourseGeneratorCfg:
    return MeshLinearObstacleCourseGeneratorCfg(
        # size + num_rows + num_cols are overwritten by the generator's __init__.
        size=(COURSE_LENGTH, CORRIDOR_WIDTH),
        border_width=5.0,
        border_height=0.05,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        course_difficulty=float(difficulty),
        obstacle_specs=_build_specs(),
        # sub_terrains is required by parent TerrainGeneratorCfg but not used.
        # We register one stub entry per patch so eval_course / summary.json
        # downstream see meaningful names in sub_terrain_names.
        sub_terrains={name: None for name in PATCH_NAMES},
    )


COURSE_EASY_CFG = _make_course_cfg(0.40)
COURSE_MED_CFG = _make_course_cfg(0.50)
COURSE_HARD_CFG = _make_course_cfg(0.70)
COURSE_EXTREME_CFG = _make_course_cfg(0.95)
