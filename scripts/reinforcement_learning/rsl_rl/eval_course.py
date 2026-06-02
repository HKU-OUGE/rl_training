"""Evaluate a SplitMoE policy on the M20 obstacle course (v2: 12 patches, fixed order).

CLI:
    python eval_course.py --level easy --ablation full --num_envs 400 --num_steps 2500

Scoring is SINGLE-ATTEMPT: each env's progress is frozen at its FIRST
termination (fall / oob / below-ground / reached_goal / 50 s time_out). A
robot that falls and slides, or that auto-resets and wanders, does not inflate
its score. progress_ratio = max_x_reached / COURSE_LENGTH (object-frame disp).
binary_complete = reached the mesh end (course_reached_goal fired).

Reuses eval_moe's install_hooks / handle_dones / build_and_load_runner /
resolve_checkpoint helpers. Adds course-specific buffers:
    max_x_reached       (N,) float32   — best object-frame disp_x before first_done
    per_patch_pass      (N, 12) bool   — whether env crossed each patch end before first_done
    first_fail_patch    (N,) int8      — index of patch where env first failed (-1 if reached_goal)
    time_to_complete    (N,) int32     — step at which reached_goal fired (-1 if never)

The 12 patches are 2 cycles of [hurdle, slope, stairs, rail, stones, step_up]:
    indices  0..5: cycle 1 (hurdle, slope, stairs, rail, stones, step_up)
    indices  6..11: cycle 2 (hurdle_2, slope_2, stairs_2, rail_2, stones_2, step_up_2)

After env construction (and before first reset() rollout) we override env_origins
to (x_start, 0, spawn_z) for every env, where x_start = -size[0] * num_rows / 2
is the start of the lead-in pad (course-frame x=0).
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate MoE policy on the M20 obstacle course.")
parser.add_argument("--level", type=str, default="easy",
                    choices=["easy", "med", "hard", "extreme"],
                    help="Course difficulty preset.")
parser.add_argument("--ablation", type=str, default="full",
                    choices=["full", "A1", "A2", "A3", "B1", "B2", "locomoe", "mlp_baseline"],
                    help="Ablation variant whose ckpt to eval.")
parser.add_argument("--num_envs", type=int, default=400)
parser.add_argument("--num_steps", type=int, default=2500,
                    help="Control steps (= 50 s at dt=0.02), matching episode_length_s=50.")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run dir name or absolute path; default = latest.")
parser.add_argument("--checkpoint", type=str, default="model_*.pt",
                    help="Checkpoint glob.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Override output dir.")
parser.add_argument("--heading_stiffness", type=float, default=None,
                    help="Override course heading_control_stiffness. Lower = weaker "
                         "yaw auto-correction (robot must self-stabilize heading). "
                         "0 disables heading rescue entirely (cmd_wz fixed at 0).")
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)
parser.add_argument("--latent_sample_envs", type=int, default=200)
parser.add_argument("--latent_sample_stride", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==============================================================================
# Heavy imports (AFTER AppLauncher)
# ==============================================================================
import numpy as np
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Inject SplitMoE classes into rsl_rl namespace (mirrors eval_moe.py boot)
sys.path.append(os.getcwd())
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import (
    SplitMoEActorCritic,
    SplitMoEPPO,
)
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module
rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEPPO = SplitMoEPPO

# Reuse eval_moe machinery. eval_moe.py creates its own AppLauncher at module
# import time, which would shut down ours. Monkey-patch AppLauncher to a no-op
# subclass that hands back the already-running app, then import.
import isaaclab.app as _isaac_app  # noqa: E402

_real_AppLauncher = _isaac_app.AppLauncher


class _StubAppLauncher:
    """No-op shim: hands back the already-running SimulationApp."""

    def __init__(self, *_args, **_kwargs):
        pass

    @property
    def app(self):
        return simulation_app

    @staticmethod
    def add_app_launcher_args(parser):
        # already added once by us; calling twice on the same parser raises
        return None


_isaac_app.AppLauncher = _StubAppLauncher
try:
    from eval_moe import (  # noqa: E402
        DEVICE,
        install_hooks,
        handle_dones,
        build_and_load_runner,
        resolve_checkpoint,
        _iter_num,
    )
finally:
    _isaac_app.AppLauncher = _real_AppLauncher
from rl_training.terrains.config.course import (  # noqa: E402
    COURSE_PATCH_END_X,
    COURSE_LENGTH,
    COURSE_NUM_PATCHES,
)
from rl_training.terrains.course_terrain_generator import SPAWN_INSET  # noqa: E402


TASK_PER_LEVEL = {
    "easy": "Course-MoE-Teacher-Deeprobotics-M20-easy-v0",
    "med": "Course-MoE-Teacher-Deeprobotics-M20-med-v0",
    "hard": "Course-MoE-Teacher-Deeprobotics-M20-hard-v0",
    "extreme": "Course-MoE-Teacher-Deeprobotics-M20-extreme-v0",
}


def allocate_course_buffers(N, T, model, args):
    """Course buffers + the standard eval_moe per-step traces."""
    # SplitMoE has num_leg/wheel_experts; LocoMoE has num_experts (unified);
    # MlpBaseline has no experts at all. Coalesce to a single 'nL' and 'nW'
    # split so the rest of the pipeline stays unchanged.
    if hasattr(model, "num_leg_experts"):
        nL = model.num_leg_experts
        nW = model.num_wheel_experts
    elif hasattr(model, "num_experts"):  # LocoMoE
        nL = int(model.num_experts)
        nW = 0
    else:  # MlpBaseline — no gate
        nL = 0
        nW = 0
    L = getattr(model, "latent_dim", 256)

    n_latent = min(args.latent_sample_envs, N)
    latent_env_idx = torch.linspace(0, N - 1, n_latent, device=DEVICE).long()
    t_stride = max(1, args.latent_sample_stride)
    T_latent = (T + t_stride - 1) // t_stride

    return {
        # constants (filled after reset; not really meaningful here since all envs
        # share row 0 col 0, but kept for plot_moe_eval_v2 compatibility)
        "terrain_types": torch.zeros(N, dtype=torch.int32, device=DEVICE),
        "terrain_levels": torch.zeros(N, dtype=torch.int32, device=DEVICE),
        # episode tracking
        "term_cause": torch.full((N,), -1, dtype=torch.int8, device=DEVICE),
        "term_step": torch.full((N,), -1, dtype=torch.int32, device=DEVICE),
        "first_done": torch.zeros(N, dtype=torch.bool, device=DEVICE),
        # per-step traces
        "root_pos_xy": torch.zeros(N, T, 2, dtype=torch.float32, device=DEVICE),
        "cmd": torch.zeros(N, T, 3, dtype=torch.float32, device=DEVICE),
        "actual_vel": torch.zeros(N, T, 3, dtype=torch.float32, device=DEVICE),
        "gate_leg": torch.zeros(N, T, nL, dtype=torch.float16, device=DEVICE),
        "gate_wheel": torch.zeros(N, T, nW, dtype=torch.float16, device=DEVICE),
        "gru_latent_sample": torch.zeros(n_latent, T_latent, L, dtype=torch.float16, device=DEVICE),
        "latent_env_idx": latent_env_idx,
        "t_stride": t_stride,
        # course-specific (SINGLE-ATTEMPT scoring: progress is frozen at the
        # FIRST termination per env via the ~first_done gate in
        # update_course_progress, so a fall/oob/timeout ends scoring and any
        # post-reset wandering does not count).
        # max_x_reached = best object-frame disp_x reached before first_done.
        "max_x_reached": torch.zeros(N, dtype=torch.float32, device=DEVICE),
        # diagnostics: per-env max body tilt (sin of tilt angle) and min base z
        # reached BEFORE first_done — lets us see how close to "fallen" an env
        # got without yet triggering course_tipover.
        "max_tilt_sin": torch.zeros(N, dtype=torch.float32, device=DEVICE),
        "min_base_z": torch.full((N,), 9.9, dtype=torch.float32, device=DEVICE),
        "per_patch_pass": torch.zeros(N, COURSE_NUM_PATCHES, dtype=torch.bool, device=DEVICE),
        "first_fail_patch": torch.full((N,), -1, dtype=torch.int8, device=DEVICE),
        "time_to_complete": torch.full((N,), -1, dtype=torch.int32, device=DEVICE),
        # placeholder so handle_dones can lazy-init it
        "reward_terms": None,
    }


def update_course_progress(bufs, base_env, t, patch_end_x_t):
    """Per-step bookkeeping for max_x_reached + per_patch_pass (single-attempt).

    disp_x is measured in OBJECT frame (= world_x - env_origin_x + SPAWN_INSET)
    so it spans [SPAWN_INSET, COURSE_LENGTH] and is directly comparable to
    COURSE_PATCH_END_X (object frame) and the COURSE_LENGTH reach threshold.

    Only envs that have NOT yet terminated (~first_done) accumulate progress —
    once a robot falls / goes OOB / times out, its score is frozen and any
    post-auto-reset wandering does not inflate max_x_reached.
    """
    robot = base_env.scene["robot"]
    disp_x = (robot.data.root_pos_w[:, 0]
              - base_env.scene.env_origins[:, 0] + SPAWN_INSET)
    active = ~bufs["first_done"]
    if active.any():
        bufs["max_x_reached"][active] = torch.maximum(
            bufs["max_x_reached"][active], disp_x[active]
        )
        pass_now = disp_x.unsqueeze(1) >= patch_end_x_t  # (N, P)
        bufs["per_patch_pass"][active] = bufs["per_patch_pass"][active] | pass_now[active]
        # tilt diagnostic: sin(tilt) = horizontal gravity magnitude in body frame
        tilt_sin = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
        bufs["max_tilt_sin"][active] = torch.maximum(bufs["max_tilt_sin"][active], tilt_sin[active])
        base_z = robot.data.root_pos_w[:, 2]
        bufs["min_base_z"][active] = torch.minimum(bufs["min_base_z"][active], base_z[active])


def finalize_first_fail(bufs):
    """Set first_fail_patch for envs that terminated without reaching goal.

    first_fail_patch = index of the first patch they failed to pass (0..11).
    For envs that completed the course (reached_goal fired), leave at -1.
    """
    done_mask = bufs["first_done"]
    # We use term_cause-based detection: reached_goal envs have term_cause = 10
    # (our enum below). All other done envs failed somewhere along the course.
    reached_goal = bufs["term_cause"] == COURSE_TERM_ENUM["course_reached_goal"]
    failed = done_mask & ~reached_goal
    if failed.any():
        # First patch they failed = first patch where per_patch_pass is False
        passed = bufs["per_patch_pass"][failed]               # (Nf, P)
        # argmax of (~passed) along P returns the first False; if all True,
        # argmax returns 0 — guard that with all-pass check.
        first_fail = (~passed).int().argmax(dim=1)            # (Nf,)
        all_passed_anyway = passed.all(dim=1)
        first_fail = torch.where(all_passed_anyway,
                                 torch.full_like(first_fail, COURSE_NUM_PATCHES - 1),
                                 first_fail)
        bufs["first_fail_patch"][failed] = first_fail.to(torch.int8)


def finalize_time_to_complete(bufs):
    """Copy term_step into time_to_complete for envs that hit course_reached_goal."""
    reached_goal = bufs["term_cause"] == COURSE_TERM_ENUM["course_reached_goal"]
    bufs["time_to_complete"][reached_goal] = bufs["term_step"][reached_goal]


def finalize_reached_goal_progress(bufs):
    """Clamp progress markers for envs that fired course_reached_goal.

    update_course_progress runs BEFORE env.step() because Isaac Lab auto-resets
    inside step() — so the disp_x at the step the robot crosses COURSE_LENGTH is
    never captured (the pre-step disp_x is the closest we got, typically
    COURSE_LENGTH - epsilon). For envs whose terminations include
    course_reached_goal, we set max_x_reached = COURSE_LENGTH and
    per_patch_pass[-1] = True so progress_ratio = 1.0 and binary_complete
    catches them.
    """
    reached_goal = bufs["term_cause"] == COURSE_TERM_ENUM["course_reached_goal"]
    if reached_goal.any():
        bufs["max_x_reached"][reached_goal] = COURSE_LENGTH
        # All COURSE_NUM_PATCHES patches are by definition cleared if the robot
        # crossed the course_reached_goal threshold.
        bufs["per_patch_pass"][reached_goal, :] = True


# Termination enum for the course (distinct from eval_moe's enum to avoid
# colliding with terrain_out_of_bounds etc., though we keep the same values
# for time_out / bad_orientation that survive from the parent env cfg).
COURSE_TERM_ENUM = {
    "time_out": 0,
    "bad_orientation": 3,
    "course_oob": 5,
    "course_below_ground": 6,
    "course_reached_goal": 10,
    "course_tipover": 11,
}


# Patch eval_moe.handle_dones to use OUR enum, since it does name->int mapping
# via TERM_NAME_TO_ENUM. We monkeypatch the module-level dict.
import eval_moe as _eval_moe_mod  # noqa: E402
_eval_moe_mod.TERM_NAME_TO_ENUM = COURSE_TERM_ENUM


def save_outputs(bufs, env_cfg, ckpt_path, experiment_name, iter_num, level, ablation):
    """Write raw.npz + summary.json into logs/moe_eval/<exp>_course_<level>/<ts>_iter<N>/."""
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(
            "logs", "moe_eval",
            f"{experiment_name}_course_{level}",
            f"{ts}_iter{iter_num}",
        )
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    # raw.npz
    np_buf = {}
    for k, v in bufs.items():
        if isinstance(v, torch.Tensor):
            np_buf[k] = v.cpu().numpy()
        elif k == "t_stride":
            np_buf[k] = np.int32(v)
    raw_path = os.path.join(out_dir, "raw.npz")
    np.savez_compressed(raw_path, **np_buf)
    print(f"[eval_course] raw.npz written ({os.path.getsize(raw_path)/1e6:.1f} MB) → {raw_path}")

    tgen = env_cfg.scene.terrain.terrain_generator
    sub_terrain_names = list(tgen.sub_terrains.keys())

    summary = {
        "task": TASK_PER_LEVEL[level],
        "level": level,
        "ablation": ablation,
        "course_difficulty": float(tgen.course_difficulty),
        "experiment_name": experiment_name,
        "ckpt_path": ckpt_path,
        "iter": iter_num,
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "sub_terrain_names": sub_terrain_names,
        # plot_moe_eval_v2.env_subterrain_name expects col_to_subterrain. With
        # num_cols=1 in the course, every env's terrain_types index is 0, so we
        # only need a 1-element list pointing to *something*. We pick the first
        # patch by convention (callers that care about per-patch slicing should
        # use per_patch_pass / max_x_reached instead).
        "col_to_subterrain": [sub_terrain_names[0]],
        "num_rows": int(tgen.num_rows),
        "num_cols": int(tgen.num_cols),
        "patch_end_x": COURSE_PATCH_END_X,
        "course_length": float(COURSE_LENGTH),
        "term_enum": {v: k for k, v in COURSE_TERM_ENUM.items()},
        "reward_term_names": bufs.get("_reward_term_names", []),
        "model": {
            "num_leg_experts": int(bufs["gate_leg"].shape[-1]),
            "num_wheel_experts": int(bufs["gate_wheel"].shape[-1]),
            "latent_dim": int(bufs["gru_latent_sample"].shape[-1]),
        },
        "tstamp": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval_course] summary.json written → {out_dir}/summary.json")

    return out_dir


def main():
    task = TASK_PER_LEVEL[args.level]
    print(f"[eval_course] task={task} ablation={args.ablation} num_envs={args.num_envs} "
          f"num_steps={args.num_steps}")

    env_cfg = parse_env_cfg(task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.seed = args.seed

    # Optional heading-rescue override: the course default railroads yaw to +x
    # with stiffness=1.0. Lowering it forces the robot to self-stabilize heading;
    # 0 disables yaw command entirely (tests gait stability without rescue).
    if args.heading_stiffness is not None:
        cmds = env_cfg.commands.base_velocity
        cmds.heading_control_stiffness = float(args.heading_stiffness)
        if args.heading_stiffness == 0.0:
            cmds.heading_command = False
            cmds.rel_heading_envs = 0.0
            cmds.ranges.ang_vel_z = (0.0, 0.0)
        print(f"[eval_course] heading_control_stiffness overridden -> {args.heading_stiffness}")

    # train cfg — branches by ablation. LocoMoE / MlpBaseline are SEPARATE
    # architectures (not just cfg overrides on SplitMoE), so we load their
    # own PPOCfg and inject the policy/algo classes into rsl_rl's namespace
    # (mirrors train_locomoe.py / train_moe.py for MlpBaseline).
    if args.ablation == "locomoe":
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.locomoe_terrain import (
            LocoMoEActorCritic, LocoMoEPPO, LocoMoEPPOCfg,
        )
        import rsl_rl.modules as rsl_modules
        rsl_modules.LocoMoEActorCritic = LocoMoEActorCritic
        runner_module.LocoMoEActorCritic = LocoMoEActorCritic
        runner_module.LocoMoEPPO = LocoMoEPPO
        train_cfg = LocoMoEPPOCfg()
        train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
        train_cfg_dict["policy"]["class_name"] = "LocoMoEActorCritic"
        train_cfg_dict["algorithm"]["class_name"] = "LocoMoEPPO"
        train_cfg_dict["experiment_name"] = "locomoe_teacher_parallel"
    elif args.ablation == "mlp_baseline":
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import (
            MlpHeadActorCritic, MlpBaselinePPOCfg,
        )
        import rsl_rl.modules as rsl_modules
        rsl_modules.MlpHeadActorCritic = MlpHeadActorCritic
        runner_module.MlpHeadActorCritic = MlpHeadActorCritic
        train_cfg = MlpBaselinePPOCfg()
        train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
        train_cfg_dict["policy"]["class_name"] = "MlpHeadActorCritic"
        train_cfg_dict["experiment_name"] = "mlp_baseline_teacher_parallel"
    else:
        train_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
        train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
        if args.num_wheel_experts is not None:
            train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
        if args.num_leg_experts is not None:
            train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
        for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
            train_cfg_dict["policy"].pop(k, None)
        # SplitMoE ablation overrides
        _ABL_POLICY_OVERRIDES = {
            "A1": {"single_gate": True},
            "B2": {"blind_vision": True},
        }
        if args.ablation != "full":
            for k, v in _ABL_POLICY_OVERRIDES.get(args.ablation, {}).items():
                train_cfg_dict["policy"][k] = v
            train_cfg_dict["experiment_name"] = f"split_moe_teacher_parallel_abl_{args.ablation}"
            print(f"[eval_course][ablation={args.ablation}] policy overrides: "
                  f"{_ABL_POLICY_OVERRIDES.get(args.ablation, {})}")

    train_cfg_dict["device"] = DEVICE
    train_cfg_dict["logger"] = "tensorboard"
    experiment_name = train_cfg_dict.get("experiment_name", "split_moe_teacher_parallel")

    # checkpoint
    ckpt_path, run_dir = resolve_checkpoint(experiment_name, args.load_run, args.checkpoint)
    print(f"[eval_course] using ckpt: {ckpt_path}")
    iter_num = _iter_num(ckpt_path)

    # env + runner
    env_gym = gym.make(task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env_gym, clip_actions=train_cfg_dict.get("clip_actions", True))
    runner, model = build_and_load_runner(env_wrapped, train_cfg_dict, ckpt_path)
    nL_print = getattr(model, "num_leg_experts", getattr(model, "num_experts", 0))
    nW_print = getattr(model, "num_wheel_experts", 0)
    L_print = getattr(model, "latent_dim", 0)
    print(f"[eval_course] model loaded. num_leg_experts={nL_print} "
          f"num_wheel_experts={nW_print} latent_dim={L_print}")

    # install_hooks reads SplitMoE/A1 gate attrs; skip for LocoMoE/MlpBaseline
    # since we don't analyse gate routing on the course (per-patch metrics only).
    if hasattr(model, "leg_gate") or hasattr(model, "unified_gate"):
        sink = install_hooks(model)
    else:
        sink = {}
        print("[eval_course] gate hooks skipped (no leg_gate/unified_gate — LocoMoE/Mlp baseline)")

    # Override env_origins to the START of patch 0 BEFORE the first reset rollout.
    # Isaac Lab centres the mesh at x=0 with size[0]*num_rows/2 = 22.5 m of
    # half-extent in +x. We want env_origin.x = -22.5 so disp_x measures distance
    # from course start. Keep z = current value (terrain spawn height).
    base_env_for_origin = env_wrapped.unwrapped
    while hasattr(base_env_for_origin, "env"):
        base_env_for_origin = base_env_for_origin.env
    if hasattr(base_env_for_origin, "unwrapped"):
        base_env_for_origin = base_env_for_origin.unwrapped

    tgen = env_cfg.scene.terrain.terrain_generator
    # v2.2: shift spawn 0.5m into the lead-in so the robot's full footprint
    # sits on the track slab (was straddling the lead-in/border boundary).
    SPAWN_INSET = 0.5
    x_start = -float(tgen.size[0]) * float(tgen.num_rows) / 2.0 + SPAWN_INSET
    cur_z = base_env_for_origin.scene.env_origins[:, 2].clone()
    new_origin = torch.zeros_like(base_env_for_origin.scene.env_origins)
    new_origin[:, 0] = x_start
    new_origin[:, 1] = 0.0
    new_origin[:, 2] = cur_z
    base_env_for_origin.scene.env_origins.copy_(new_origin)
    # Mirror into terrain importer too (the reset events use it).
    if hasattr(base_env_for_origin.scene, "terrain"):
        ti = base_env_for_origin.scene.terrain
        if hasattr(ti, "env_origins"):
            ti.env_origins.copy_(new_origin)
    print(f"[eval_course] env_origins overridden to x={x_start} (inset={SPAWN_INSET}m) for all {args.num_envs} envs")

    obs, _ = env_wrapped.reset()

    N = env_wrapped.num_envs
    T = args.num_steps
    bufs = allocate_course_buffers(N, T, model, args)

    base_env = base_env_for_origin
    bufs["terrain_types"].copy_(base_env.scene.terrain.terrain_types.to(torch.int32))
    bufs["terrain_levels"].copy_(base_env.scene.terrain.terrain_levels.to(torch.int32))

    patch_end_x_t = torch.tensor(COURSE_PATCH_END_X, device=DEVICE,
                                 dtype=torch.float32).unsqueeze(0)  # (1, P)

    print(f"[eval_course] buffers allocated. N={N} T={T} "
          f"latent_sample shape={tuple(bufs['gru_latent_sample'].shape)}")

    # =========================================================================
    # Collection loop
    # =========================================================================
    policy = runner.get_inference_policy(device=DEVICE)
    robot = base_env.scene["robot"]
    cmd_mgr = base_env.command_manager
    t_stride = bufs["t_stride"]
    latent_env_idx = bufs["latent_env_idx"]

    print(f"[eval_course] starting rollout: N={N} T={T} ...")
    with torch.inference_mode():
        for t in range(T):
            actions = policy(obs)

            active = ~bufs["first_done"]
            bufs["root_pos_xy"][active, t] = robot.data.root_pos_w[active, :2]
            bufs["cmd"][active, t] = cmd_mgr.get_command("base_velocity")[active]
            actual = torch.cat([robot.data.root_lin_vel_b[:, :2],
                                robot.data.root_ang_vel_b[:, 2:3]], dim=-1)
            bufs["actual_vel"][active, t] = actual[active]
            # Gate logging only when hooks present (skip for LocoMoE/Mlp baseline).
            if "leg_logits" in sink:
                gate_leg = torch.softmax(sink["leg_logits"], dim=-1)
                gate_wheel = torch.softmax(sink["wheel_logits"], dim=-1)
                bufs["gate_leg"][active, t] = gate_leg[active].to(torch.float16)
                bufs["gate_wheel"][active, t] = gate_wheel[active].to(torch.float16)
                if t % t_stride == 0:
                    t_l = t // t_stride
                    bufs["gru_latent_sample"][:, t_l] = sink["rnn_out"][latent_env_idx].to(torch.float16)

            # Course progress BEFORE step — root_pos_w reflects this step's
            # outcome, not the post-reset spawn. Isaac Lab auto-resets terminated
            # envs INSIDE step(), so the goal-crossing position would be lost
            # if we recorded after step. finalize_course_progress() below clamps
            # max_x_reached and per_patch_pass for reached_goal envs to COURSE_LENGTH.
            update_course_progress(bufs, base_env, t, patch_end_x_t)

            obs, _, dones, info = env_wrapped.step(actions)

            # GRU hidden-state reset for just-terminated envs. Isaac Lab
            # auto-resets the env's physics state on done but does NOT touch the
            # policy's RNN hidden state. Without this, every env that hits
            # time_out / oob / reach_goal continues with stale GRU state across
            # the implicit episode boundary, polluting downstream metrics.
            # This mirrors the per-step model.reset(dones) call in rsl_rl's
            # rollout collector during training.
            if dones.any() and hasattr(model, "reset"):
                model.reset(dones.bool())

            handle_dones(t, dones, info, bufs, base_env)

            if (t + 1) % 100 == 0:
                done_frac = bufs["first_done"].float().mean().item()
                prog_med = float(bufs["max_x_reached"].median().item())
                print(f"[eval_course] step {t + 1}/{T}  done = {100*done_frac:.1f}%  "
                      f"median max_x = {prog_med:.2f} m")

            if bufs["first_done"].all():
                print(f"[eval_course] all envs done at step {t + 1} — early exit")
                break

        # Drain steps to flush any pending time_out (mirror eval_moe behaviour)
        n_drain = 2
        for d in range(n_drain):
            if bufs["first_done"].all():
                break
            actions = policy(obs)
            obs, _, dones, info = env_wrapped.step(actions)
            if dones.any() and hasattr(model, "reset"):
                model.reset(dones.bool())
            handle_dones(T + d, dones, info, bufs, base_env)

    # Safety: envs that STILL never died → mark as time_out
    never_done = ~bufs["first_done"]
    if never_done.any():
        bufs["term_cause"][never_done] = COURSE_TERM_ENUM["time_out"]
        bufs["term_step"][never_done] = T - 1
        print(f"[eval_course] {never_done.sum().item()} envs never terminated — marked time_out")
        bufs["first_done"][never_done] = True

    # Final per-env attribution. finalize_reached_goal_progress must run BEFORE
    # finalize_first_fail so reached_goal envs are excluded from the fail logic.
    finalize_reached_goal_progress(bufs)
    finalize_first_fail(bufs)
    finalize_time_to_complete(bufs)

    # Headline stats — single-attempt: max_x_reached is frozen at first_done.
    progress = (bufs["max_x_reached"] / COURSE_LENGTH).clamp(0.0, 1.0)
    print(f"[eval_course] mean progress_ratio = {float(progress.mean()):.3f}")
    print(f"[eval_course] binary_complete     = {float((progress >= 1.0).float().mean()):.3f}")

    out_dir = save_outputs(bufs, env_cfg, ckpt_path, experiment_name, iter_num,
                           args.level, args.ablation)
    print(f"[eval_course] ALL DONE. results at {out_dir}")
    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
