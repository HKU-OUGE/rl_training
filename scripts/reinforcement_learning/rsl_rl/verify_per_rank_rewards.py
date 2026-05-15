"""Pre-flight 验证: apply_platform_rewards / apply_scan_rewards 能否构建有效 env.

单 GPU 即可跑 (不需要 8 卡 torchrun)。验证链:
  parse_env_cfg → apply_*_rewards → gym.make (构建 RewardManager) → reset + step
任何 reward term 名字错 / mdp 函数签名错 / RewardManager 构建失败都会在这里暴露。

用法 (在 102 上, env_isaaclab 环境):
  python /tmp/verify_per_rank_rewards.py --mode platform --headless
  python /tmp/verify_per_rank_rewards.py --mode scan --headless
  python /tmp/verify_per_rank_rewards.py --mode baseline --headless   # 对照
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["platform", "scan", "baseline"], default="platform")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import rl_training.tasks  # noqa: F401  注册任务

TASK = "Rough-MoE-Teacher-Deeprobotics-M20-v0"
DEVICE = "cuda:0"

print(f"\n{'='*60}\n[verify] mode={args.mode}  task={TASK}  num_envs={args.num_envs}\n{'='*60}")

env_cfg = parse_env_cfg(TASK, device=DEVICE, num_envs=args.num_envs)
print("[verify] parse_env_cfg OK (含 __post_init__ + disable_zero_weight_rewards)")

if args.mode == "platform":
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.teacher_per_rank_rewards import (
        apply_platform_rewards,
    )
    apply_platform_rewards(env_cfg)
    print("[verify] apply_platform_rewards(env_cfg) OK")
elif args.mode == "scan":
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.teacher_per_rank_rewards import (
        apply_scan_rewards,
    )
    apply_scan_rewards(env_cfg)
    print("[verify] apply_scan_rewards(env_cfg) OK")
else:
    print("[verify] baseline — 不改 reward")

# 打印最终生效的 reward term + weight
print(f"\n[verify] 最终 reward terms ({args.mode}):")
n_active = 0
for attr in sorted(dir(env_cfg.rewards)):
    if attr.startswith("_"):
        continue
    v = getattr(env_cfg.rewards, attr)
    if v is not None and not callable(v):
        fn = getattr(v.func, "__name__", str(v.func))
        print(f"    {attr:32s} weight={v.weight:>10}  func={fn}")
        n_active += 1
print(f"[verify] active reward terms: {n_active}")

# 关键测试: gym.make 构建 RewardManager
print("\n[verify] gym.make ... (构建 RewardManager)")
env = gym.make(TASK, cfg=env_cfg)
print("[verify] gym.make OK — RewardManager 构建成功")

# reset + step: 验证 reward 函数实际能算
obs, _ = env.reset()
print("[verify] env.reset() OK")
act_shape = env.action_space.shape
for i in range(args.steps):
    act = torch.zeros(act_shape, device=DEVICE)
    obs, rew, term, trunc, info = env.step(act)
    print(f"[verify] step {i:2d}: reward mean={rew.mean().item():+.4f}  "
          f"min={rew.min().item():+.4f}  max={rew.max().item():+.4f}")

print(f"\n{'='*60}\n[verify] PASS — {args.mode}: env 构建 + step 全程无异常\n{'='*60}\n")
env.close()
simulation_app.close()
