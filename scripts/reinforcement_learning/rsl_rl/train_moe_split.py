# train_moe_split.py

import argparse
import sys
import os
from datetime import datetime

# 1. 启动 App
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Train Cross-Observation MoE Policy (Dual Backbone)")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoE-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")

# MoE 参数 (覆盖 Config)
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)

# Resume 参数
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename (e.g. model_1000.pt)")

# 分布式参数
parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. 导入依赖
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# ==============================================================================
# 性能优化配置
# ==============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
# ==============================================================================

# === 关键：导入自定义模块 (moe_split_cross) ===
# 假设 moe_split_cross.py 和 moe_terrain.py 在同一目录下
try:
    sys.path.append(os.getcwd())
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_split_cross import CrossMoEActorCritic, SplitMoEPPO
    print("[Info] Imported Cross-MoE classes from current directory.")
except ImportError:
    try:
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_split_cross import CrossMoEActorCritic, SplitMoEPPO
        print("[Info] Imported Cross-MoE classes from project path.")
    except ImportError:
        raise ImportError("Could not import CrossMoEActorCritic/SplitMoEPPO from moe_split_cross.py")

# === 注入到 RSL-RL ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

# 1. 注入 Policy
# 注意：我们将 CrossMoEActorCritic 注入进去，覆盖配置中可能存在的引用
rsl_modules.CrossMoEActorCritic = CrossMoEActorCritic
runner_module.CrossMoEActorCritic = CrossMoEActorCritic

# 为了兼容性，如果 config 写错了名字，也可以在这里做别名映射
rsl_modules.SplitMoEActorCritic = CrossMoEActorCritic 

# 2. 注入 Algorithm (使用 mo_split_cross.py 中修复了 Aux Loss 的 PPO)
runner_module.SplitMoEPPO = SplitMoEPPO

def main():
    # 获取设备信息
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}"
    print(f"[Info] Using device: {device}")

    # 解析环境配置
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    
    # 设置 Seed
    if args.seed is not None:
        env_cfg.seed = args.seed + local_rank
    
    env = gym.make(args.task, cfg=env_cfg)

    # 加载训练配置
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"): 
        train_cfg_dict = train_cfg.to_dict()
    else: 
        train_cfg_dict = train_cfg

    # === 配置覆写 (Config Overrides) ===
    print(f"\n[Info] Switching Policy Class to: CrossMoEActorCritic")
    # 强制指定 Policy 类名为 CrossMoEActorCritic
    train_cfg_dict["policy"]["class_name"] = "CrossMoEActorCritic"

    # 注入 device
    train_cfg_dict["device"] = device

    # 覆盖专家数量参数
    if args.num_wheel_experts is not None:
        train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts is not None:
        train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    
    # Resume 设置
    train_cfg_dict["resume"] = args.resume
    train_cfg_dict["load_run"] = args.load_run
    train_cfg_dict["load_checkpoint"] = args.checkpoint

    # 清理旧参数 (防止冲突)
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
        train_cfg_dict["policy"].pop(k, None)

    # === 路径设置 ===
    # 修改默认实验名为 cross_moe，方便区分日志
    experiment_name = train_cfg_dict.get("experiment_name", "cross_moe_end2end")
    log_root_path = os.path.abspath(os.path.join("logs", "moe_training", experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # 1. 解析 Resume 路径
    resume_path = None
    if args.resume:
        try:
            resume_path = get_checkpoint_path(log_root_path, args.load_run, args.checkpoint)
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
        except Exception as e:
            print(f"[Error] Failed to resolve checkpoint path from root {log_root_path}: {e}")
            sys.exit(1)

    # 2. 创建本次运行的新日志目录
    log_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.load_run:
        log_dir += f"_resume_{args.load_run}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Current run directory: {log_dir}")
    
    # === 环境包装 ===
    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    # === 初始化 Runner ===
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device=device)

    # === 加载 Checkpoint ===
    if resume_path:
        runner.load(resume_path)

    # [Debug] 打印架构，确认使用的是 CrossMoE
    print("\n" + "="*80)
    print("[Debug] Policy Architecture:")
    try:
        model = getattr(runner.alg, "actor_critic", getattr(runner.alg, "policy", None))
        if model:
            print(model)
            # 简单检查是否为 CrossMoE
            if hasattr(model, "rnn_leg") and hasattr(model, "rnn_wheel"):
                print("\n[Check] Verified: Dual RNN structure detected (CrossMoE).")
            else:
                print("\n[Warning] Dual RNN structure NOT detected!")
    except: pass
    print("="*80 + "\n")

    # 开始训练
    runner.learn(num_learning_iterations=train_cfg_dict["max_iterations"], init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()