"""eval_train.py — SplitMoE ablation training suite.

Derived from train_moe.py. Adds a ``--ablation`` flag selecting one variant from
the built-in ABLATIONS registry; everything else (per-rank terrain dispatch,
wandb shared-mode logging, per-rank critic, etc.) is identical to train_moe.py,
so every variant trains on the exact same code path as the main run.

Reference ("full") is v0 / SplitMoEPPOCfg (6 leg + 3 wheel experts). Run it
exactly like train_moe.py, under torchrun, with an extra --ablation:

    PER_RANK_TERRAIN=1 PER_RANK_NO_ILLEGAL_CONTACT=1 \\
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
    torchrun --nnodes=1 --nproc_per_node=8 \\
        scripts/reinforcement_learning/rsl_rl/eval_train.py \\
        --task=Rough-MoE-Teacher-Deeprobotics-M20-v0 --headless \\
        --num_envs 3000 --logger wandb --distributed \\
        --max_init_terrain_level 1 --ablation A1

Variants: full, A1 (single gate), A2 (shared critic), A3 (no L_sym),
          B1 (no L_bal), B2 (blind). See the ABLATIONS dict below.
"""

import argparse
import sys
import os
from datetime import datetime
from typing import Callable

# ==============================================================================
# 1. 启动 App
# ==============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import cli_args
except ImportError:
    pass

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Train SplitMoE ablation variants (eval suite)")
parser.add_argument("--task", type=str, default="Rough-MoE-Teacher-Deeprobotics-M20-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--max_init_terrain_level", type=int, default=None,
                    help="Override env_cfg.scene.terrain.max_init_terrain_level for ALL ranks. "
                         "Default None = use env cfg value (typically 0). "
                         "Use e.g. 18 to spawn robots on high-level terrain immediately "
                         "(useful when resuming from a trained policy).")

# H-MoE 参数
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)

# 消融实验变体 (见下方 ABLATIONS 注册表)
parser.add_argument("--ablation", type=str, default="full",
                    choices=["full", "A1", "A2", "A3", "B1", "B2"],
                    help="Ablation variant: full=reference; A1=single merged gate; "
                         "A2=shared critic; A3=no L_sym; B1=no L_bal; B2=blind.")

parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")

# 分布式参数
parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

# 视频录制参数
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1800, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=20000, help="Interval between video recordings (in steps).")
parser.add_argument("--tour_interval", type=int, default=100, help="How many steps to stay on one sub-terrain before moving camera.")

if 'cli_args' in sys.modules:
    cli_args.add_rsl_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
is_master = (local_rank == 0)

# 检测 torchrun 环境，自动启用 distributed
if os.environ.get("WORLD_SIZE", None) is not None and int(os.environ["WORLD_SIZE"]) > 1:
    args.distributed = True

if args.video and is_master:
    args.enable_cameras = True
args.device = f"cuda:{local_rank}"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==============================================================================
# 2. 导入依赖
# ==============================================================================
import torch
import gymnasium as gym
from gymnasium.wrappers.rendering import RecordVideo
from gymnasium import logger as gym_logger
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab.utils.math as math_utils  # [New] 用于视角坐标系转换

try:
    import wandb
except ImportError:
    wandb = None
try:
    import av 
except ImportError:
    av = None

# ==============================================================================
# 性能优化配置 
# ==============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # 启用 cudnn 自动选 conv 算法 (port 5260c3d, 通用提升 RNN/CNN 训练吞吐)

# ==============================================================================
# [New] 动态视角巡视 Wrapper (修正 KeyError Bug)
# ==============================================================================
class CameraTourWrapper(gym.Wrapper):
    """
    定期遍历各个子地形区域，并自动跟随离地形中心最近的机器人。
    """
    def __init__(self, env, tour_interval=50):
        super().__init__(env)
        self.tour_interval = tour_interval
        self.step_count = 0
        self.cur_row = 0
        self.cur_col = 0
        self.camera_history = []
        self.OFFSET = [-2.5, 0.0, 1.5]  # 跟随镜头的相对坐标 (X 后方, Z 上方)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        # 提取 IsaacLab 的原生环境
        base_env = self.unwrapped
        if hasattr(base_env, "scene"):
            # 【修复点】：安全地获取 robot 实体，避免触发 __getitem__ 遍历
            try:
                robot = base_env.scene["robot"]
            except KeyError:
                robot = None
                
            # 尝试获取地形原点网格
            origins = None
            if hasattr(base_env.scene, "terrain") and hasattr(base_env.scene.terrain, "terrain_origins"):
                origins = base_env.scene.terrain.terrain_origins
            
            if robot is not None and origins is not None:
                num_rows, num_cols = origins.shape[0], origins.shape[1]
                
                # 到达切换间隔时，移动到下一个子地形
                if self.step_count > 0 and self.step_count % self.tour_interval == 0:
                    self.cur_col += 1
                    if self.cur_col >= num_cols:
                        self.cur_col = 0
                        self.cur_row = (self.cur_row + 1) % num_rows
                    self.camera_history.clear() # 清空平滑缓存防止镜头瞬移拖尾
                    
                target_origin = origins[self.cur_row, self.cur_col]
                
                # 寻找距离当前地形中心最近的机器人
                root_pos = robot.data.root_pos_w
                dists = torch.norm(root_pos[:, :2] - target_origin[:2], dim=-1)
                closest_idx = torch.argmin(dists).item()
                
                r_pos = root_pos[closest_idx]
                r_quat = robot.data.root_quat_w[closest_idx]
                
                # 使用 math_utils 计算世界坐标系下的镜头坐标 (unsqueeze以支持batch运算)
                offset_local = torch.tensor(self.OFFSET, device=r_pos.device).unsqueeze(0)
                offset_world = math_utils.quat_apply(r_quat.unsqueeze(0), offset_local)[0]
                eye = r_pos + offset_world
                
                # 平滑处理
                self.camera_history.append(eye)
                if len(self.camera_history) > 30:
                    self.camera_history.pop(0)
                smooth_eye = torch.stack(self.camera_history).mean(dim=0)
                
                # 更新仿真器视角
                base_env.sim.set_camera_view(smooth_eye.cpu().numpy(), r_pos.cpu().numpy())
        
        self.step_count += 1
        return obs, rew, terminated, truncated, info

# ==============================================================================
# 支持 PyAV (h264) 高效编码和 W&B 自动上传
# ==============================================================================
class CustomRecordVideo(RecordVideo):
    def __init__(self, env: gym.Env, video_folder: str, episode_trigger: Callable[[int], bool] | None = None, step_trigger: Callable[[int], bool] | None = None, video_length: int = 0, name_prefix: str = "rl-video", fps: int | None = None, disable_logger: bool = True, enable_wandb: bool = True, wandb_key: str = "train/video", video_resolution: tuple[int, int] = (1280, 720), video_crf: int = 30):
        super().__init__(env=env, video_folder=video_folder, episode_trigger=episode_trigger, step_trigger=step_trigger, video_length=video_length, name_prefix=name_prefix, disable_logger=disable_logger)
        if fps is not None: self.frames_per_sec = fps  
        self.enable_wandb = bool(enable_wandb and (wandb is not None))
        self.wandb_key = wandb_key
        self.video_resolution = tuple(video_resolution)
        self.video_crf = int(video_crf)

    def _write_with_pyav(self, frames, path):
        if av is None: raise RuntimeError("PyAV (av) not available. Please pip install av.")
        container = av.open(path, "w")
        stream = container.add_stream("libx264", rate=round(float(self.frames_per_sec)))
        stream.width, stream.height = self.video_resolution
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(self.video_crf), "preset": "ultrafast"}
        for fr in frames:
            vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
            if fr.shape[1] != self.video_resolution[0] or fr.shape[0] != self.video_resolution[1]:
                vf = vf.reformat(width=self.video_resolution[0], height=self.video_resolution[1])
            packet = stream.encode(vf)
            if packet: container.mux(packet)
        packet = stream.encode(None)
        if packet: container.mux(packet)
        container.close()

    def stop_recording(self):
        assert self.recording, "stop_recording was called, but no recording was started"
        path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
        if len(self.recorded_frames) == 0:
            gym_logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            os.makedirs(self.video_folder, exist_ok=True)
            try:
                self._write_with_pyav(self.recorded_frames, path)
            except Exception as e:
                gym_logger.warn(f"Failed to write video with PyAV, falling back to moviepy: {e}")
                super().stop_recording()
            else:
                self.recorded_frames = []
                self.recording = False
                self._video_name = None

            if self.enable_wandb and os.path.exists(path) and (wandb is not None):
                try:
                    wandb.log({self.wandb_key: wandb.Video(path, format="mp4")}, commit=False)
                    if not self.disable_logger: print(f"[W&B] Logged video: {path}")
                except Exception as e:
                    print(f"[WARN] wandb video log failed: {e}")

# === 导入自定义模块 ===
try:
    sys.path.append(os.getcwd())
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO, MlpHeadActorCritic
    print("[Info] Imported H-MoE classes from current directory.")
except ImportError:
    try:
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO, MlpHeadActorCritic
        print("[Info] Imported H-MoE classes from project path.")
    except ImportError:
        raise ImportError("Could not import SplitMoEActorCritic/PPO/MlpHeadActorCritic from moe_terrain.py")

# === 注入到 RSL-RL ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEPPO = SplitMoEPPO
rsl_modules.MlpHeadActorCritic = MlpHeadActorCritic
runner_module.MlpHeadActorCritic = MlpHeadActorCritic

# ==============================================================================
# 消融实验注册表 (Ablation registry)
# ==============================================================================
# Each variant isolates ONE design from the SplitMoE paper. "full" is the
# reference (v0 / SplitMoEPPOCfg, 6 leg + 3 wheel experts). A variant is applied
# by (a) overriding keys in the policy cfg dict and/or (b) setting env vars that
# SplitMoEPPO reads at construction time:
#   A1  single merged gate           policy.single_gate=True   (9 experts, 16-D each)
#   A2  shared (all-reduced) critic   env DEBUG_NO_PERRANK_CRITIC=1
#   A3  drop symmetry loss L_sym      env ABLATE_NO_SYM_LOSS=1
#   B1  drop load-balancing loss      policy.aux_loss_coef=0.0
#   B2  blind (exteroception zeroed)  policy.blind_vision=True
ABLATIONS = {
    "full": {"desc": "reference: all designs on"},
    "A1":   {"desc": "single merged gate (leg+wheel experts unified, 16-D output)",
             "policy": {"single_gate": True}},
    "A2":   {"desc": "shared critic (critic re-joins all-reduce; per-rank critic off)",
             "env": {"DEBUG_NO_PERRANK_CRITIC": "1"}},
    "A3":   {"desc": "drop symmetry loss L_sym",
             "env": {"ABLATE_NO_SYM_LOSS": "1"}},
    "B1":   {"desc": "drop load-balancing loss L_bal (aux_loss_coef=0)",
             "policy": {"aux_loss_coef": 0.0}},
    "B2":   {"desc": "blind: exteroception features zeroed",
             "policy": {"blind_vision": True}},
}


def apply_ablation(train_cfg_dict, ablation):
    """Apply one ablation variant in place.

    Mutates ``train_cfg_dict["policy"]`` and sets the env vars that SplitMoEPPO
    reads in __init__. Must be called before the OnPolicyRunner is built.
    Returns the variant spec dict (for logging).
    """
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation '{ablation}'; choices = {list(ABLATIONS)}")
    spec = ABLATIONS[ablation]
    for key, val in spec.get("policy", {}).items():
        train_cfg_dict["policy"][key] = val
    for key, val in spec.get("env", {}).items():
        os.environ[key] = val
    return spec


def main():
    device = f"cuda:{local_rank}"
    
    if not is_master and wandb is not None:
        os.environ["WANDB_MODE"] = "disabled"

    if is_master:
        print(f"[Info] Using device: {device}")

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)

    if args.seed is not None:
        env_cfg.seed = args.seed + local_rank

    # ---- max_init_terrain_level override (applies to all ranks, before per-rank terrain swap) ----
    if args.max_init_terrain_level is not None:
        env_cfg.scene.terrain.max_init_terrain_level = args.max_init_terrain_level
        if local_rank == 0:
            print(f"[INFO] max_init_terrain_level override: {args.max_init_terrain_level} (applies to all ranks)")

    # =========================================================================
    # Per-rank terrain dispatch (异构地形多卡训练) — port cfb9e3b
    # 每张卡跑不同 terrain, 共享同一 reward/network, DDP 平均 actor 梯度;
    # 配合 PER_RANK_TERRAIN=1 + 多 GPU + Rough-MoE-Teacher 任务一起用。
    # rank → modality:
    #   0: FLAT             — 纯平地
    #   1: STAIR_SLOPE      — 楼梯 + 斜坡
    #   2: PLATFORM         — pit + boxes (高台爬降)
    #   3: SCAN             — hurdle (跨栏)
    #   4: STEPPING_STONES  — baseline 的 stepping_stones
    #   5: RAIL             — rail bars
    #   6: NOISE            — random_rough
    #   7: GRID             — discrete grid boxes
    # 不改 reward / command / yaw / func — 这些维持 baseline 共享
    # =========================================================================
    if args.distributed and os.environ.get("PER_RANK_TERRAIN", "0") == "1":
        from rl_training.terrains.config.rough import (
            STAIR_ONLY_TEACHER_TERRAINS_CFG,
            FLAT_SLOPE_TEACHER_TERRAINS_CFG,
            PLATFORM_TEACHER_TERRAINS_CFG,
            SCAN_TEACHER_TERRAINS_CFG,
            STEPPING_STONES_TEACHER_TERRAINS_CFG,
            RAIL_TEACHER_TERRAINS_CFG,
            NOISE_TEACHER_TERRAINS_CFG,
            MOE_ROUGH_TERRAINS_CFG,
        )
        _RANK_TERRAIN_MAP = [
            MOE_ROUGH_TERRAINS_CFG,              # rank 0: MIXED (master rank, baseline-like 全局 anchor)
            STAIR_ONLY_TEACHER_TERRAINS_CFG,     # rank 1: STAIR (纯台阶 50/50 ±), 配 T3 reward
            PLATFORM_TEACHER_TERRAINS_CFG,       # rank 2: PLATFORM (pit/box)
            SCAN_TEACHER_TERRAINS_CFG,           # rank 3: SCAN (hurdle)
            STEPPING_STONES_TEACHER_TERRAINS_CFG, # rank 4: STEPPING_STONES (替代 GAP)
            RAIL_TEACHER_TERRAINS_CFG,           # rank 5: RAIL
            NOISE_TEACHER_TERRAINS_CFG,          # rank 6: NOISE
            FLAT_SLOPE_TEACHER_TERRAINS_CFG,     # rank 7: 50% plane + 25/25 slope±, 配原 2 项温和 reward
        ]
        if local_rank < len(_RANK_TERRAIN_MAP):
            chosen = _RANK_TERRAIN_MAP[local_rank]
            env_cfg.scene.terrain.terrain_generator = chosen
            print(f"[rank={local_rank}] terrain → "
                  f"{list(chosen.sub_terrains.keys())} "
                  f"(num_rows={chosen.num_rows}, curriculum={chosen.curriculum})")
        else:
            chosen = None
            print(f"[rank={local_rank}] WARN: out of RANK_TERRAIN_MAP range, "
                  f"using default terrain from task cfg")

        # =====================================================================
        # Per-rank command override
        # y 命令全局关闭 (env default lin_vel_y=(0,0)), 所有 rank 都不发 y 命令;
        # x 全局 ±1.0 (env default), 所有 rank 一致, 不再做 per-rank x override.
        # 此前的 FLAT 纯侧移 / NOISE 宽 x 都已取消.
        # =====================================================================

        # =====================================================================
        # Per-rank reward override
        #   PLATFORM → 高台攀爬 reward (port main); SCAN → 钻栏 reward (port main).
        #   其他 rank (FLAT/MIXED/STAIR_SLOPE/STONES/RAIL/NOISE) 用共享 baseline reward.
        # DEBUG: 设 DEBUG_NO_PERRANK_REWARD=1 跳过 (退回全 rank 共享 baseline reward)
        # =====================================================================
        if os.environ.get("DEBUG_NO_PERRANK_REWARD", "0") != "1":
            if chosen is PLATFORM_TEACHER_TERRAINS_CFG:
                from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.teacher_per_rank_rewards import apply_platform_rewards
                apply_platform_rewards(env_cfg)
                print(f"[rank={local_rank}] reward → PLATFORM (高台攀爬, port main)")
            elif chosen is SCAN_TEACHER_TERRAINS_CFG:
                from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.teacher_per_rank_rewards import apply_scan_rewards
                apply_scan_rewards(env_cfg)
                print(f"[rank={local_rank}] reward → SCAN (钻栏, port main)")
            elif chosen is STAIR_ONLY_TEACHER_TERRAINS_CFG:
                from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.teacher_per_rank_rewards import apply_stair_rewards
                apply_stair_rewards(env_cfg)
                print(f"[rank={local_rank}] reward → STAIR (T3: lin_vel_z+undesired + joint_mirror/hipy/knee/joint_acc/action_rate 放宽)")
            elif chosen is FLAT_SLOPE_TEACHER_TERRAINS_CFG:
                from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.teacher_per_rank_rewards import apply_stair_slope_rewards
                apply_stair_slope_rewards(env_cfg)
                print(f"[rank={local_rank}] reward → FLAT_SLOPE (T1: 仅 lin_vel_z 放宽 + undesired_contacts -0.5)")

        # =====================================================================
        # Per-rank termination override
        #   硬地形 rank (STAIR/PLATFORM/SCAN/STONES/RAIL) 关掉 illegal_contact,
        #   让 base_link 撞击不致死, 鼓励"撞了再爬"探索更激进的通过策略.
        #   FLAT/SLOPE/NOISE/MIXED 仍保留 base-die, 维持基础步态学习信号.
        #   bad_orientation_2 全局保持 None (env_cfg 默认), 不在此恢复.
        # 启用: PER_RANK_NO_ILLEGAL_CONTACT=1 (与 PER_RANK_TERRAIN=1 一起用)
        # =====================================================================
        if os.environ.get("PER_RANK_NO_ILLEGAL_CONTACT", "0") == "1":
            HARD_RANKS = {1, 2, 3, 4, 5}  # STAIR, PLATFORM, SCAN, STONES, RAIL
            if local_rank in HARD_RANKS:
                env_cfg.terminations.illegal_contact = None
                print(f"[rank={local_rank}] termination → illegal_contact DISABLED (hard terrain)")
            else:
                print(f"[rank={local_rank}] termination → illegal_contact KEPT (base-die on flat/slope/noise/mixed)")


    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)

    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"): 
        train_cfg_dict = train_cfg.to_dict()
    else: 
        train_cfg_dict = train_cfg

    if 'cli_args' in sys.modules and hasattr(cli_args, 'update_rsl_rl_cfg'):
        if hasattr(train_cfg, 'seed'):
            train_cfg = cli_args.update_rsl_rl_cfg(train_cfg, args)
            if hasattr(train_cfg, "to_dict"): 
                train_cfg_dict = train_cfg.to_dict()
            else: 
                train_cfg_dict = train_cfg

    # Honor the cfg's policy class_name (set by task registry). E.g.
    # SplitMoEPPOCfg → "SplitMoEActorCritic", MlpBaselinePPOCfg → "MlpHeadActorCritic".
    policy_class = train_cfg_dict["policy"].get("class_name", "SplitMoEActorCritic")
    train_cfg_dict["policy"]["class_name"] = policy_class
    if is_master:
        print(f"\n[Info] Policy Class: {policy_class}")
    train_cfg_dict["device"] = device

    train_cfg_dict["logger"] = getattr(args, "logger", "wandb") or "wandb"
    train_cfg_dict["wandb_project"] = args.task

    if args.num_wheel_experts is not None: train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts is not None: train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    
    if hasattr(args, "resume") and args.resume:
        train_cfg_dict["resume"] = args.resume
        train_cfg_dict["load_run"] = getattr(args, "load_run", None)
        train_cfg_dict["load_checkpoint"] = getattr(args, "checkpoint", None)

    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
        train_cfg_dict["policy"].pop(k, None)

    # ---- apply ablation variant (eval suite) ----
    # Mutates policy cfg + sets env vars; suffix experiment_name so each variant
    # logs to its own dir / wandb run and never collides with the main run.
    _abl_spec = apply_ablation(train_cfg_dict, args.ablation)
    _abl_base_name = train_cfg_dict.get("experiment_name", "h_moe_end2end")
    train_cfg_dict["experiment_name"] = f"{_abl_base_name}_abl_{args.ablation}"
    if is_master:
        print(f"\n[Ablation] variant = {args.ablation} :: {_abl_spec['desc']}")
        print(f"[Ablation] experiment_name -> {train_cfg_dict['experiment_name']}")
        if _abl_spec.get("policy"):
            print(f"[Ablation] policy cfg overrides: {_abl_spec['policy']}")
        if _abl_spec.get("env"):
            print(f"[Ablation] env vars set: {_abl_spec['env']}")

    experiment_name = getattr(args, "experiment_name", train_cfg_dict.get("experiment_name", "h_moe_end2end"))
    if not experiment_name: experiment_name = train_cfg_dict.get("experiment_name", "h_moe_end2end")
    
    log_root_path = os.path.abspath(os.path.join("logs", "moe_training", experiment_name))
    
    if is_master: print(f"[INFO] Logging experiment in directory: {log_root_path}")

    resume_path = None
    if getattr(args, "resume", False) or train_cfg_dict.get("resume", False):
        checkpoint = getattr(args, "checkpoint", None) or train_cfg_dict.get("load_checkpoint")
        load_run = getattr(args, "load_run", None) or train_cfg_dict.get("load_run")
        if checkpoint and os.path.exists(checkpoint):
            resume_path = checkpoint
            if is_master: print(f"[INFO] Directly using exact checkpoint path: {resume_path}")
        else:
            try:
                load_run_str = load_run if load_run is not None else ".*"
                ckpt_str = checkpoint if checkpoint is not None else ".*"
                resume_path = get_checkpoint_path(log_root_path, load_run_str, ckpt_str)
                if is_master: print(f"[INFO] Loading model checkpoint from: {resume_path}")
            except Exception as e:
                if is_master: print(f"[Error] Failed to resolve checkpoint path from root {log_root_path}: {e}")
                sys.exit(1)

    if args.distributed and local_rank != 0:
        # 方案 C: 非 master rank 的 log_dir 重定向到 /tmp, 避免在项目 logs/ 里
        # 生成 7 个无 ckpt 的空 dir (rank0 的 timestamp + 7 个不同 timestamp = 8 dir).
        # ckpt 只在 rank0 写 (runner.save 已 patch 为 master-only), wandb 走 shared mode,
        # 这里的 SummaryWriter 只是把 TB events 倒进 /tmp 不污染项目 logs.
        log_dir = os.path.join("/tmp", "rl_training_noop_logs", f"rank_{local_rank}")
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if getattr(args, "run_name", None): log_dir += f"_{args.run_name}"
        elif getattr(args, "load_run", None): log_dir += f"_resume_{args.load_run}"

        log_dir = os.path.join(log_root_path, log_dir)
    
    if is_master: print(f"[INFO] Current run directory: {log_dir}")
    
    train_cfg_dict["run_name"] = os.path.basename(log_dir)

    # === [MODIFIED] 环境包装 (Video Recording with Tour) ===
    if args.video and is_master:
        # 1. 挂载视角遍历插件
        env = CameraTourWrapper(env, tour_interval=args.tour_interval)
        
        # 2. 挂载视频录制插件
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_video_interval == 0 if args_video_interval > 0 else False,
            "video_length": args.video_length,
            "disable_logger": True,
            "enable_wandb": (train_cfg_dict.get("logger") == "wandb"),
            "wandb_key": "train/video",
            "video_resolution": (640, 360), 
            "video_crf": 30,
        }
        args_video_interval = args.video_interval 
        
        print(f"[INFO] Recording videos during training every {args.video_interval} steps.")
        print(f"[INFO] Camera will tour terrains every {args.tour_interval} steps.")
        env = CustomRecordVideo(env, **video_kwargs)

    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device=device)

    # =========================================================================
    # Per-rank logging hook (monkey-patch runner.log) — jsonl 文件路径, 无 NCCL
    #
    # 设计:
    # - 每 rank 各自把本 iter 的 metrics 写一行 JSON 到自己的文件
    #   /tmp/per_rank_train_{name}_rank{N}.jsonl
    # - 不做 all_gather / all_reduce → 杜绝 NCCL 死锁可能 (之前的根因)
    # - 离线用 scripts/.../plot_per_rank.py 读 jsonl 画曲线 / 上传 wandb
    #
    # 历史: 之前用 dist.all_gather hook 直接写 wandb scalar, 但 iter 0
    # 之后 all_gather 死锁 (即使做了 deterministic keys 修复也未解决, 真因
    # 可能是 CUDA stream / NCCL 内部状态污染). jsonl 方案完全绕开 NCCL 路径.
    #
    # DEBUG: 设 DEBUG_NO_PERRANK_LOG=1 跳过 hook (理论上没必要了, 保留作 escape)
    # =========================================================================
    if args.distributed and os.environ.get("PER_RANK_TERRAIN", "0") == "1" \
            and os.environ.get("DEBUG_NO_PERRANK_LOG", "0") != "1":
        import json as _json
        RANK_NAMES = os.environ.get(
            "PER_RANK_NAMES",
            "MIXED,STAIR_SLOPE,PLATFORM,SCAN,STONES,RAIL,NOISE,FLAT",
        ).split(",")
        _rank_name = RANK_NAMES[local_rank] if local_rank < len(RANK_NAMES) else f"rank{local_rank}"
        _jsonl_path = f"/tmp/per_rank_train_{_rank_name}_rank{local_rank}.jsonl"

        _base_env = env
        while hasattr(_base_env, "env"):
            _base_env = _base_env.env
        if hasattr(_base_env, "unwrapped"):
            _base_env = _base_env.unwrapped

        # 清空 (truncate) 自己的 jsonl, 让本次 run 数据干净
        try:
            with open(_jsonl_path, "w") as _f:
                pass
        except Exception as _e:
            if is_master:
                print(f"[per_rank_log] could not init {_jsonl_path}: {_e}")

        _original_log = runner.log

        def _per_rank_log(locs, *a, **kw):
            ret = _original_log(locs, *a, **kw)
            try:
                it = locs.get("it", runner.current_learning_iteration)
                metrics = {
                    "iter": int(it),
                    "rank": int(local_rank),
                    "name": _rank_name,
                }
                if hasattr(_base_env, "scene") and hasattr(_base_env.scene, "terrain") \
                   and hasattr(_base_env.scene.terrain, "terrain_levels"):
                    tl = _base_env.scene.terrain.terrain_levels.float()
                    metrics["terrain_level_mean"] = float(tl.mean().item())
                    metrics["terrain_level_max"] = float(tl.max().item())
                if locs.get("rewbuffer") and len(locs["rewbuffer"]) > 0:
                    metrics["ep_reward"] = float(sum(locs["rewbuffer"]) / len(locs["rewbuffer"]))
                if locs.get("lenbuffer") and len(locs["lenbuffer"]) > 0:
                    metrics["ep_length"] = float(sum(locs["lenbuffer"]) / len(locs["lenbuffer"]))
                if "mean_value_loss" in locs:
                    metrics["value_function"] = float(locs["mean_value_loss"])
                if "mean_surrogate_loss" in locs:
                    metrics["surrogate"] = float(locs["mean_surrogate_loss"])
                if "mean_entropy" in locs:
                    metrics["entropy"] = float(locs["mean_entropy"])
                if locs.get("ep_infos"):
                    for fixed_key in ["time_out", "illegal_contact", "terrain_out_of_bounds"]:
                        full_key = f"Episode_Termination/{fixed_key}"
                        vals = [ep[full_key] for ep in locs["ep_infos"] if full_key in ep]
                        if vals:
                            avg = sum(v.item() if hasattr(v, "item") else float(v)
                                      for v in vals) / len(vals)
                            metrics[f"term_{fixed_key}"] = float(avg)
                # Append 一行 (no NCCL)
                with open(_jsonl_path, "a") as _f:
                    _f.write(_json.dumps(metrics) + "\n")
            except Exception as e:
                if is_master:
                    print(f"[per_rank_log] write failed: {e}")
            return ret

        runner.log = _per_rank_log
        if is_master:
            print(f"[INFO] PerRank logging enabled (jsonl mode). RANK_NAMES = {RANK_NAMES}")
            print(f"[INFO] Each rank writes to /tmp/per_rank_train_<name>_rank<N>.jsonl")
            print(f"[INFO] Visualize: scripts/reinforcement_learning/rsl_rl/plot_per_rank.py --remote <host>")

    if resume_path:
        loaded_dict = torch.load(resume_path, map_location=device)
        state_dict = loaded_dict.get("model_state_dict", loaded_dict)
        
        if any(k.startswith("student.") for k in state_dict.keys()):
            if is_master: print("[INFO] Detected Distilled Checkpoint. Stripping 'student.' prefix...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("student."):
                    new_key = k.replace("student.", "", 1)
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            runner.alg.policy.load_state_dict(new_state_dict, strict=False)
            if "iter" in loaded_dict: runner.current_learning_iteration = loaded_dict["iter"]
                
        else:
            if is_master: print("[INFO] Standard PPO checkpoint. Loading natively...")
            runner.load(resume_path)

    if is_master:
        print("\n" + "="*80)
        print("[Debug] Policy Architecture:")
        try:
            model = getattr(runner.alg, "actor_critic", getattr(runner.alg, "policy", None))
            if model: print(model)
        except: pass
        print("="*80 + "\n")

    # =========================================================================
    # Shared-mode wandb: 8 个 rank 共写 1 个 wandb run, 数据用 x_label 区分。
    # wandb 0.22 已 hard-block "同机多进程独立 init" (非 master rank 的 init
    # 返回 None), 官方多进程方案是 mode='shared': rank 0 创建 run, 其他 rank
    # 用同一 run.id 加入 (worker), wandb 自动用 x_label 标记 metric 来源。
    # 文档要求 SDK >= 0.19.9 (我们 0.22.3 ✓)。
    #
    # 流程:
    #   1. rank 0 wandb.init(mode='shared', x_primary=True), 拿 run.id
    #   2. dist.broadcast_object_list 把 run.id 同步到其他 rank
    #   3. 其他 rank wandb.init(id=run.id, mode='shared', x_primary=False)
    #   4. 替换 runner.writer 为 wandb.log wrapper; 强制 disable_logs=False
    #      让所有 rank 都调 runner.log() → 每 rank 写自己 x_label 标记的 metric
    #   5. 非 master rank: save=no-op + print 静音 (避免 8× 重复)
    # =========================================================================
    if args.distributed and train_cfg_dict.get("logger") == "wandb":
        try:
            import wandb as _wandb
            import torch.distributed as _dist
            from torch.utils.tensorboard.writer import SummaryWriter as _SW

            _rank_names = os.environ.get(
                "PER_RANK_NAMES",
                "MIXED,STAIR_SLOPE,PLATFORM,SCAN,STONES,RAIL,NOISE,FLAT",
            ).split(",")
            _rank_tag = _rank_names[local_rank] if local_rank < len(_rank_names) else f"r{local_rank}"
            _x_label = f"rank{local_rank}_{_rank_tag}"
            # group 跨多次启动用 TORCHELASTIC_RUN_ID 区分, 也是跨 rank 一致的
            _run_id_env = os.environ.get("TORCHELASTIC_RUN_ID", "")
            _wandb_group = f"{experiment_name}_{_run_id_env}" if _run_id_env else experiment_name

            try:
                _project = train_cfg_dict["wandb_project"]
            except KeyError:
                _project = "rl_training"
            _entity = os.environ.get("WANDB_USERNAME")
            _run_name = experiment_name  # 共享 run 的人类可读名

            # --- step 1: rank 0 init primary ---
            if local_rank == 0:
                _wandb_run = _wandb.init(
                    project=_project, entity=_entity, group=_wandb_group, name=_run_name,
                    settings=_wandb.Settings(
                        mode="shared",
                        x_label=_x_label,
                        x_primary=True,
                    ),
                )
                _shared_run_id = _wandb_run.id
            else:
                _shared_run_id = None

            # --- step 2: broadcast run_id (rank 0 → others) ---
            _obj_list = [_shared_run_id]
            _dist.broadcast_object_list(_obj_list, src=0)
            _shared_run_id = _obj_list[0]

            # --- step 3: workers attach with same id, x_primary=False ---
            if local_rank != 0:
                _wandb_run = _wandb.init(
                    project=_project, entity=_entity, group=_wandb_group,
                    id=_shared_run_id,
                    settings=_wandb.Settings(
                        mode="shared",
                        x_label=_x_label,
                        x_primary=False,
                    ),
                )

            # --- step 4: writer wrapper that routes add_scalar → wandb.log ---
            # 关键 1: wandb 的 x_label 只 tag system metrics / 控制台日志, **不会** 自动
            # tag 用户 wandb.log({tag: value}) 的指标。8 个 rank 都写 'Train/mean_reward'
            # 会互相覆盖。官方建议: 手动在 metric key 加 rank 前缀, 让 wandb UI 按
            # rank 自动归到不同 folder。
            #
            # 关键 2: shared mode 下 wandb 忽略 wandb.log(..., step=N), auto-increment
            # 自己的 _step (8 rank × add_scalar/iter → step 暴涨到几万)。修复用
            # wandb.define_metric 把 'iter' 声明为 step_metric, 每 log 里带上 'iter'
            # 字段, wandb UI 自动用 iter 做 x 轴。
            _per_rank_prefix = _x_label  # 比如 "rank3_SCAN"
            try:
                _wandb.define_metric("iter")
                _wandb.define_metric("*", step_metric="iter")
            except Exception as _e:
                print(f"[wandb-shared rank={local_rank}] define_metric failed: {_e}")

            class _SharedWandbWriter(_SW):
                """Writer for shared-mode wandb: 写本进程的 wandb run, metric key 加
                rank 前缀避免 8 rank 同 key 互相覆盖, 用 iter 做 x 轴避免 _step 暴涨。"""
                def __init__(self, log_dir, flush_secs):
                    super().__init__(log_dir, flush_secs)

                def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
                    super().add_scalar(tag, scalar_value, global_step, walltime, new_style)
                    try:
                        # rank 前缀 + iter 字段一起 log。不传 step= (shared mode 会忽略,
                        # 且经实测会让 _step 失控)。wandb 用 define_metric 声明的 iter 做 x 轴。
                        payload = {f"{_per_rank_prefix}/{tag}": scalar_value}
                        if global_step is not None:
                            payload["iter"] = int(global_step)
                        _wandb.log(payload)
                    except Exception:
                        pass

                def stop(self):
                    try:
                        _wandb.finish()
                    except Exception:
                        pass

                def save_file(self, path):
                    # primary 才传文件
                    if local_rank == 0:
                        try:
                            _wandb.save(path)
                        except Exception:
                            pass

                def save_model(self, model_path, iter):
                    if local_rank == 0:
                        try:
                            _wandb.save(model_path)
                        except Exception:
                            pass

                def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
                    if local_rank == 0:
                        try:
                            _cfg = {"runner_cfg": runner_cfg, "alg_cfg": alg_cfg, "policy_cfg": policy_cfg}
                            try:
                                _cfg["env_cfg"] = env_cfg.to_dict()
                            except Exception:
                                pass
                            _wandb.config.update(_cfg)
                        except Exception:
                            pass

            runner.writer = _SharedWandbWriter(log_dir, 10)
            runner.logger_type = "wandb"
            runner.disable_logs = False

            # --- step 5: 非 master rank 不存 ckpt + 静音 print ---
            if local_rank != 0:
                runner.save = lambda *a, **kw: None
                import contextlib as _ctxlib
                _devnull = open(os.devnull, "w")
                _orig_log_for_silence = runner.log
                def _silent_log(locs, *a, **kw):
                    with _ctxlib.redirect_stdout(_devnull):
                        return _orig_log_for_silence(locs, *a, **kw)
                runner.log = _silent_log

            # 手动调 log_config (rsl_rl 的 _prepare_logging_writer 跳过了)
            try:
                runner.writer.log_config(
                    env.unwrapped.cfg if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "cfg") else {},
                    train_cfg_dict,
                    train_cfg_dict.get("algorithm", {}),
                    train_cfg_dict.get("policy", {}),
                )
            except Exception as _e:
                print(f"[wandb-shared rank={local_rank}] log_config skipped: {_e}")
            print(f"[wandb-shared rank={local_rank}] run_id={_shared_run_id} "
                  f"group={_wandb_group} x_label={_x_label} primary={local_rank==0}")
        except Exception as _e:
            print(f"[wandb-shared rank={local_rank}] init FAILED, fallback to master-only: {_e}")

    runner.learn(num_learning_iterations=train_cfg_dict["max_iterations"], init_at_random_ep_len=True)
    
    if is_master:
        try:
            if wandb and wandb.run is not None:
                wandb.log({}, commit=True)
                wandb.finish()
        except Exception as e:
            print(f"[WARN] Failed to gracefully close wandb: {e}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()