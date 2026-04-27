"""Benchmark multi-GPU scaling using the SAME moe_terrain configs as production training.

Runs a small fixed number of iterations and logs per-iter wall time + reward
mean to a JSON file. Designed to be invoked under torchrun for multi-GPU.

Pairs with benchmark_scaling.sh which loops over 1/4/8 GPU configs.

Usage:
    # single GPU
    python scripts/reinforcement_learning/rsl_rl/benchmark_scaling.py \
        --task MoE-Scan-Teacher-Deeprobotics-M20-v0 \
        --num_envs 2000 --max_iter 50 --output /tmp/bench_1gpu.json

    # 4 GPU
    torchrun --nproc_per_node=4 scripts/reinforcement_learning/rsl_rl/benchmark_scaling.py \
        --task MoE-Scan-Teacher-Deeprobotics-M20-v0 \
        --num_envs 2000 --max_iter 50 --output /tmp/bench_4gpu.json

The script writes (master rank only):
{
  "task":, "num_envs_per_gpu":, "world_size":, "max_iter":,
  "iter_wall_times":  [t1, t2, ...],   # seconds per iter
  "iter_mean_rewards":[r1, r2, ...],   # rew_total mean per iter (or None)
  "total_wall_time":,
  "warmup_wall_time":  # iter 0 (includes JIT etc.)
}
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-GPU scaling benchmark using moe_terrain cfg.")
parser.add_argument("--task", type=str, default="MoE-Scan-Teacher-Deeprobotics-M20-v0",
                    help="Must be a task whose rsl_rl_cfg_entry_point lives in moe_terrain.py")
parser.add_argument("--num_envs", type=int, default=2000, help="Per-GPU env count")
parser.add_argument("--max_iter", type=int, default=50, help="Iterations to time (keep small)")
parser.add_argument("--output", type=str, required=True, help="JSON output path (written only by master rank)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--local_rank", type=int, default=0)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
is_master = (local_rank == 0)

if world_size > 1:
    args.distributed = True
args.headless = True
args.device = f"cuda:{local_rank}"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- post-launch imports ----
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

# Inject MoE classes — same as train_moe.py
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import (  # noqa: E402
    SplitMoEActorCritic, SplitMoEPPO, SplitMoEStudentTeacher,
)
import rsl_rl.modules as rsl_modules  # noqa: E402
import rsl_rl.runners.on_policy_runner as runner_module  # noqa: E402

rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEPPO = SplitMoEPPO


def main():
    device = f"cuda:{local_rank}"
    if is_master:
        print(f"[bench] task={args.task} num_envs/gpu={args.num_envs} world_size={world_size} max_iter={args.max_iter}")

    # ---- Build env ----
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed + local_rank
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)

    # ---- Build train cfg (use the moe_terrain cfg unchanged, just override max_iter & disable logger) ----
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    train_cfg_dict = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg)
    train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
    train_cfg_dict["device"] = device
    train_cfg_dict["max_iterations"] = args.max_iter
    train_cfg_dict["save_interval"] = 10**9       # never save during bench
    train_cfg_dict["logger"] = "tensorboard"      # avoid wandb auth/network
    train_cfg_dict["empirical_normalization"] = train_cfg_dict.get("empirical_normalization", False)
    for k in ("checkpoint_wheel", "checkpoint_leg", "freeze_experts"):
        train_cfg_dict["policy"].pop(k, None)

    log_dir = f"/tmp/bench_run_rank{local_rank}_ws{world_size}"
    os.makedirs(log_dir, exist_ok=True)

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=train_cfg_dict.get("clip_actions", True))
    runner = OnPolicyRunner(env_wrapped, train_cfg_dict, log_dir=log_dir, device=device)

    if is_master:
        print(f"[bench] runner ready. starting {args.max_iter} iters...")

    # ---- Hand-rolled per-iter loop with timing ----
    iter_times = []
    iter_rewards = []  # mean episode return when available

    # warm: do iter 0 separately so its outsize cost (JIT, kernel build) doesn't dominate later stats
    torch.cuda.synchronize(device=device)
    overall_t0 = time.time()
    iter_t0 = time.time()
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    torch.cuda.synchronize(device=device)
    warmup_t = time.time() - iter_t0
    iter_times.append(warmup_t)

    # remaining iters
    for i in range(1, args.max_iter):
        torch.cuda.synchronize(device=device)
        iter_t0 = time.time()
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        torch.cuda.synchronize(device=device)
        iter_times.append(time.time() - iter_t0)

        # try to fish a reward number out of the runner if it exposes one
        rew = None
        try:
            if hasattr(runner, "tot_rewbuffer") and len(runner.tot_rewbuffer) > 0:
                rew = float(sum(runner.tot_rewbuffer) / len(runner.tot_rewbuffer))
            elif hasattr(runner, "rewbuffer") and len(runner.rewbuffer) > 0:
                rew = float(sum(runner.rewbuffer) / len(runner.rewbuffer))
        except Exception:
            rew = None
        iter_rewards.append(rew)

        if is_master and (i + 1) % 5 == 0:
            mean_recent = sum(iter_times[-5:]) / 5.0
            print(f"[bench] iter {i+1}/{args.max_iter}  last_5_mean_iter_t={mean_recent:.3f}s  "
                  f"reward~{rew if rew is not None else 'n/a'}")

    total_t = time.time() - overall_t0

    # ---- Write report (master only) ----
    if is_master:
        report = {
            "task": args.task,
            "num_envs_per_gpu": args.num_envs,
            "world_size": world_size,
            "max_iter": args.max_iter,
            "iter_wall_times": iter_times,
            "iter_mean_rewards": iter_rewards,
            "total_wall_time": total_t,
            "warmup_wall_time": warmup_t,
            "steady_iter_mean": sum(iter_times[1:]) / max(len(iter_times) - 1, 1),
            "steady_iter_p50": sorted(iter_times[1:])[(len(iter_times) - 1) // 2] if len(iter_times) > 1 else None,
            "throughput_steps_per_sec": (
                args.num_envs * world_size * train_cfg_dict.get("num_steps_per_env", 24)
                / (sum(iter_times[1:]) / max(len(iter_times) - 1, 1))
            ),
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[bench] wrote {args.output}")
        print(f"[bench] total={total_t:.1f}s  warmup={warmup_t:.1f}s  "
              f"steady_mean={report['steady_iter_mean']:.3f}s/iter  "
              f"throughput={report['throughput_steps_per_sec']:.0f} steps/s")

    simulation_app.close()


if __name__ == "__main__":
    main()
