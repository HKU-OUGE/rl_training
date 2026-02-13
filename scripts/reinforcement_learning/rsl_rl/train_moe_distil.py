import argparse
import sys
import os
from datetime import datetime

# 1. 启动 App
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Train H-MoE Policy (End-to-End or Distillation)")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoE-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")

# H-MoE 参数
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)

# Resume 参数
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename (e.g. model_1000.pt)")

# [New] 蒸馏参数
parser.add_argument("--distill", action="store_true", default=False, help="Enable Distillation (Student-Teacher) training")
parser.add_argument("--teacher_ckpt", type=str, default=None, help="Path to teacher checkpoint (required for distillation)")

# [New] 添加分布式参数支持 (AppLauncher 需要)
parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. 导入依赖
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
# [Mod] 引入 DistillationRunner
from rsl_rl.runners import OnPolicyRunner, DistillationRunner
# [Fix] 导入 DistillationRunner 模块以进行注入
import rsl_rl.runners.distillation_runner as distillation_runner_module
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# ==============================================================================
# [New] 性能优化配置 (移植自 train.py)
# ==============================================================================
# 开启 TF32 加速 (针对 RTX 30xx, 40xx, A100, H100 等 Ampere+ 架构)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 关闭确定性算法以换取速度
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
# ==============================================================================

# === 关键：导入自定义模块 ===
# 尝试导入 SplitMoE 相关类，包括新的 StudentTeacher 适配器
try:
    sys.path.append(os.getcwd())
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO, SplitMoEStudentTeacher
    print("[Info] Imported H-MoE classes from current directory.")
except ImportError:
    # 尝试从项目路径导入
    try:
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO, SplitMoEStudentTeacher
        print("[Info] Imported H-MoE classes from project path.")
    except ImportError:
        raise ImportError("Could not import SplitMoEActorCritic/PPO/StudentTeacher from moe_terrain.py")

# === 注入到 RSL-RL ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

# 1. 注入 Policy 到 RSL-RL 模块
rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SplitMoEStudentTeacher = SplitMoEStudentTeacher # [New] 注入适配器类
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic 

# [Fix] 关键步骤：将自定义类注入到 DistillationRunner 模块的全局命名空间
# 这样 distillation_runner.py 内部的 eval("SplitMoEStudentTeacher") 才能找到该类
distillation_runner_module.SplitMoEStudentTeacher = SplitMoEStudentTeacher

# 2. 注入 Algorithm
runner_module.SplitMoEPPO = SplitMoEPPO

def main():
    # [New] 获取 Local Rank 以支持动态设备选择
    # 如果通过 torchrun 启动，os.environ 会有 LOCAL_RANK；否则默认为 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}"
    print(f"[Info] Using device: {device}")

    # 解析环境配置
    # [Mod] 将 device 改为动态变量，而不是硬编码 "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    
    # [New] 如果是分布式环境，设置 seed 偏移，防止不同 GPU 采样相同数据
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
    # [New] 确保 device 正确传入 agent 配置
    train_cfg_dict["device"] = device

    # 覆盖专家数量参数
    if args.num_wheel_experts is not None:
        train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts is not None:
        train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    
    # 将 CLI 的 Resume 参数同步到 config 中
    train_cfg_dict["resume"] = args.resume
    train_cfg_dict["load_run"] = args.load_run
    train_cfg_dict["load_checkpoint"] = args.checkpoint

    # 清理旧参数
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
        train_cfg_dict["policy"].pop(k, None)

    # ==========================================================================
    # [Mod] 蒸馏模式与 PPO 模式切换逻辑
    # ==========================================================================
    if args.distill:
        print(f"\n[Info] Mode: Distillation (Student-Teacher)")
        
        # 1. 切换策略类为适配器
        train_cfg_dict["policy"]["class_name"] = "SplitMoEStudentTeacher"
        # 蒸馏时 Student 不需要 Estimator，Teacher 也不需要跑训练
        train_cfg_dict["policy"]["estimator_output_dim"] = 0 

        # 2. 切换算法配置为 Distillation
        # 参数可以根据需要微调，这里使用典型值
        train_cfg_dict["algorithm"] = {
            "class_name": "Distillation",
            "num_learning_epochs": 5,
            # "num_mini_batches": 4, # [Fix] rsl_rl Distillation does not use this parameter
            "gradient_length": 15, # Distillation 特定参数
            "learning_rate": 1.0e-4, # 模仿学习通常使用较小 LR
            "loss_type": "mse",
            "optimizer": "adam",
            "max_grad_norm": 1.0
        }

        # 3. 重新映射观测组
        # policy -> Student Input (blind_student_policy defined in EnvCfg)
        # teacher -> Teacher Input (policy defined in EnvCfg, containing priv info)
        # [Fix] 根据您的环境报错信息，学生观测组应为 'blind_student_policy'
        train_cfg_dict["obs_groups"] = {"policy": ["blind_student_policy"], "teacher": ["policy"]}
        
        # 4. 修改实验名称
        train_cfg_dict["experiment_name"] = "split_moe_distill"
        
        # 5. 确保 Teacher Checkpoint 存在
        if not args.teacher_ckpt and not args.resume:
            raise ValueError("Distillation mode requires --teacher_ckpt to load the teacher policy!")

    else:
        print(f"\n[Info] Mode: PPO Training")
        train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"

    # === 路径设置 (Logging & Resume) ===
    experiment_name = train_cfg_dict.get("experiment_name", "h_moe_end2end")
    log_root_path = os.path.abspath(os.path.join("logs", "moe_training", experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # 1. 解析 Resume 路径 (PPO Resume 或 Distill Resume)
    resume_path = None
    if args.resume:
        try:
            resume_path = get_checkpoint_path(log_root_path, args.load_run, args.checkpoint)
            print(f"[INFO] Resuming from checkpoint: {resume_path}")
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
    if args.distill:
        # 使用 DistillationRunner
        runner = DistillationRunner(env, train_cfg_dict, log_dir=log_dir, device=device)
        
        # 加载权重逻辑
        if args.resume and resume_path:
            # Case A: 恢复中断的蒸馏训练 (加载 Student 和 Teacher)
            runner.load(resume_path)
        elif args.teacher_ckpt:
            # Case B: 开始新的蒸馏 (加载 Teacher 权重)
            # 注意：SplitMoEStudentTeacher.load_state_dict 有逻辑处理仅加载 Teacher
            print(f"[INFO] Loading Teacher Policy from: {args.teacher_ckpt}")
            # [Fix] load_optimizer=False because we are starting fresh distillation with a new optimizer
            runner.load(args.teacher_ckpt, load_optimizer=False)
            
    else:
        # 使用 OnPolicyRunner (PPO)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device=device)
        if resume_path:
            runner.load(resume_path)

    # [Debug] 打印架构
    print("\n" + "="*80)
    print(f"[Debug] Runner Type: {type(runner).__name__}")
    print("[Debug] Policy Architecture:")
    try:
        model = getattr(runner.alg, "actor_critic", getattr(runner.alg, "policy", None))
        if model:
            print(model)
    except: pass
    print("="*80 + "\n")

    # 开始训练
    runner.learn(num_learning_iterations=train_cfg_dict["max_iterations"], init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()