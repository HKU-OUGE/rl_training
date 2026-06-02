"""Obstacle-course env cfgs for the M20 MoE Teacher eval (v2 short-obstacle course).

Wraps DeeproboticsM20MoETeacherEnvCfg with:
  - MeshLinearObstacleCourseGeneratorCfg (one of 4 difficulty levels), 32 m total,
    12 obstacles (hurdle/slope/stairs/rail/stones/step_up × 2 cycles), 3 m corridor.
  - cmd_vx fixed at +1.0 m/s with heading_command stiffness=1.0
  - illegal_contact disabled, bad_orientation_2 stays None
  - course_oob / course_below_ground / course_reached_goal terminations
  - all curricula off, all domain-randomization events off
  - episode_length_s = 105.0 (5250 steps @ 50Hz) — 3.3× headroom for 32 m at 1 m/s

The four levels (easy / med / hard / extreme) use course_difficulty values
0.30 / 0.50 / 0.70 / 0.95, mapping to obstacle severities of roughly:
  easy:    hurdle 49 cm, slope 0.24, stair 11 cm, rail 16 cm, stone gap 30 cm, step_up 22 cm
  med:     hurdle 45 cm, slope 0.33, stair 15 cm, rail 23 cm, stone gap 37 cm, step_up 33 cm
  hard:    hurdle 41 cm, slope 0.42, stair 18 cm, rail 30 cm, stone gap 45 cm, step_up 44 cm
  extreme: hurdle 36 cm, slope 0.53, stair 23 cm, rail 38 cm, stone gap 53 cm, step_up 57 cm
(NB: hurdle is crawl-mode — bar height = clearance, so LOW bar is HARD.
 step_up replaces v2-initial "pit": it's a raised platform robot climbs onto,
 not a depression. Other obstacle params scale the conventional way.)
"""

from __future__ import annotations

import torch

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from rl_training.terrains.config.course import (
    COURSE_EASY_CFG,
    COURSE_EXTREME_CFG,
    COURSE_HARD_CFG,
    COURSE_LENGTH,
    COURSE_MED_CFG,
)
from rl_training.terrains.course_terrain_generator import SPAWN_INSET

from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg


# ---------------------------------------------------------------------------
# Custom termination functions
# ---------------------------------------------------------------------------


def course_oob(env) -> torch.Tensor:
    """Out-of-bounds in y: leaving the 1.5m-wide centre lane around the course."""
    y_disp = env.scene["robot"].data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return y_disp.abs() > 1.5


def course_below_ground(env) -> torch.Tensor:
    """Sim safety net: robot ended up below z=-0.5 (corridor track is at z=0+).

    The v2 course has no real depressions (step_up obstacles only rise from the
    floor), so this fires only on physics glitches / falling off the side ramp
    at start. Threshold lowered from -2.0 (never fired) to -0.5 to catch real
    failures.
    """
    return env.scene["robot"].data.root_pos_w[:, 2] < -0.5


def course_reached_goal(env) -> torch.Tensor:
    """Robot reached the mesh end (object-frame x >= COURSE_LENGTH).

    disp is measured in OBJECT frame: world_x - env_origin_x + SPAWN_INSET.
    The robot spawns SPAWN_INSET into the lead-in, so world_x - env_origin_x
    maxes out at COURSE_LENGTH - SPAWN_INSET at the mesh end; adding SPAWN_INSET
    back makes the reachable object-frame disp span [SPAWN_INSET, COURSE_LENGTH],
    so this threshold fires exactly at the physical mesh end.
    """
    x_disp = (env.scene["robot"].data.root_pos_w[:, 0]
              - env.scene.env_origins[:, 0] + SPAWN_INSET)
    return x_disp >= COURSE_LENGTH


def course_tipover(env) -> torch.Tensor:
    """Lenient tip-over: terminate only when the robot is clearly falling over
    (tilt past ~72° from vertical, or fully past horizontal).

    Catches the "fall + slide" measurement cheat (a fallen robot sliding forward
    can otherwise fake course_reached_goal) WITHOUT false-firing on the large
    transient pitch of descending Extreme stairs (~45-50°), which is why the
    earlier 45°/|xy|>0.7 threshold (mdp.bad_orientation_2) was too sensitive.

    projected_gravity_b is the gravity direction in body frame: [0,0,-1] upright.
    Horizontal component magnitude = sin(tilt); sin(72°) ≈ 0.95. pg_z > 0 means
    the body is tilted past 90° (definitely fallen).
    """
    pg = env.scene["robot"].data.projected_gravity_b
    tilt_horiz = torch.linalg.norm(pg[:, :2], dim=1)  # = sin(tilt angle)
    return (tilt_horiz > 0.95) | (pg[:, 2] > 0.0)


# ---------------------------------------------------------------------------
# Shared override logic
# ---------------------------------------------------------------------------


def _apply_course_overrides(cfg: DeeproboticsM20MoETeacherEnvCfg, course_terrain_cfg) -> None:
    """Patch a parent MoE env cfg into a course-eval cfg, in-place.

    Mirrors eval_moe.apply_eval_overrides but tailored to the linear course
    (no per-row terrain curriculum to disable, no double_pit/cap to apply).
    """

    # ---- 1) Terrain ----
    cfg.scene.terrain.terrain_generator = course_terrain_cfg
    cfg.scene.terrain.max_init_terrain_level = 0

    # ---- 2) Episode length: 50 s (single-attempt budget per env) ----
    # 50 s = 2500 control steps at dt=0.02. An env that never terminates earlier
    # times out at 50 s. eval_course scores SINGLE-attempt (progress frozen at
    # first termination), so this is the max time a robot has to reach the goal.
    cfg.episode_length_s = 50.0

    # ---- 3) Commands: heading-controlled +1.0 m/s forward ----
    # heading_control_stiffness=0.5 (was 1.0): stiffness=1.0 railroaded the yaw
    # so hard that even an unstable gait stayed perfectly on the centerline,
    # masking gait quality. 0.5 gives realistic heading GUIDANCE (like a
    # waypoint/teleop follower) while still requiring the policy to self-stabilize
    # — so gait stability shows up in the score. Verified: at 0.5 full dominates
    # A2 (full hard/extreme binary 0.95/0.93 vs A2 0.63/0.23); at 1.0 A2's
    # instability was hidden (A2 looked SOTA). Matches user's manual play.
    cmds = cfg.commands.base_velocity
    cmds.heading_command = True
    cmds.heading_control_stiffness = 0.5
    cmds.rel_heading_envs = 1.0
    cmds.rel_standing_envs = 0.0
    cmds.resampling_time_range = (1e9, 1e9)
    cmds.ranges.heading = (0.0, 0.0)
    cmds.ranges.lin_vel_x = (1.0, 1.0)
    cmds.ranges.lin_vel_y = (0.0, 0.0)
    cmds.ranges.ang_vel_z = (-1.0, 1.0)
    cmds.debug_vis = False

    # ---- 4) Terminations (v2.6 production policy) ----
    #   * illegal_contact OFF — false-fires when base grazes step_up edges.
    #   * bad_orientation_2 OFF — replaced by the lenient course_tipover below.
    #   * course_tipover ON — catches the "fall + slide" cheat (a tipped robot
    #     sliding forward could otherwise fake reach_goal) using a 72° threshold
    #     that does NOT false-fire on Extreme-stairs transient pitch.
    #   * course_oob / below_ground / reached_goal / time_out as before.
    t = cfg.terminations
    t.illegal_contact = None
    t.bad_orientation_2 = None
    t.terrain_out_of_bounds = None
    t.course_oob = DoneTerm(func=course_oob)
    t.course_below_ground = DoneTerm(func=course_below_ground)
    t.course_reached_goal = DoneTerm(func=course_reached_goal)
    t.course_tipover = DoneTerm(func=course_tipover)
    # time_out stays as-is from base cfg

    # ---- 5) Disable all domain randomization events ----
    events_to_disable = [
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_actuator_gains",
        "randomize_push_robot",
    ]
    for name in events_to_disable:
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)

    # Zero reset noise so all envs spawn identically.
    if hasattr(cfg.events, "randomize_reset_base") and cfg.events.randomize_reset_base is not None:
        cfg.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        cfg.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

    # ---- 6) Curricula off ----
    if hasattr(cfg, "curriculum"):
        for name in ["terrain_levels", "command_levels_lin_vel", "command_levels_ang_vel"]:
            if hasattr(cfg.curriculum, name):
                setattr(cfg.curriculum, name, None)


# ---------------------------------------------------------------------------
# Per-level cfg classes
# ---------------------------------------------------------------------------


@configclass
class DeeproboticsM20CourseEasyEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_course_overrides(self, COURSE_EASY_CFG)
        if self.__class__.__name__ == "DeeproboticsM20CourseEasyEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class DeeproboticsM20CourseMedEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_course_overrides(self, COURSE_MED_CFG)
        if self.__class__.__name__ == "DeeproboticsM20CourseMedEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class DeeproboticsM20CourseHardEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_course_overrides(self, COURSE_HARD_CFG)
        if self.__class__.__name__ == "DeeproboticsM20CourseHardEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class DeeproboticsM20CourseExtremeEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_course_overrides(self, COURSE_EXTREME_CFG)
        if self.__class__.__name__ == "DeeproboticsM20CourseExtremeEnvCfg":
            self.disable_zero_weight_rewards()
