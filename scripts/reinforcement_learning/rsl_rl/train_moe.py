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
parser = argparse.ArgumentParser(description="Train H-MoE Policy (End-to-End)")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoE-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")

# H-MoE 参数
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)

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
torch.backends.cudnn.benchmark = True

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
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO
    print("[Info] Imported H-MoE classes from current directory.")
except ImportError:
    try:
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO
        print("[Info] Imported H-MoE classes from project path.")
    except ImportError:
        raise ImportError("Could not import SplitMoEActorCritic/PPO from moe_terrain.py")

# === 注入到 RSL-RL ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic 
runner_module.SplitMoEPPO = SplitMoEPPO

def main():
    device = f"cuda:{local_rank}"
    
    if not is_master and wandb is not None:
        os.environ["WANDB_MODE"] = "disabled"

    if is_master:
        print(f"[Info] Using device: {device}")

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)

    if args.seed is not None:
        env_cfg.seed = args.seed + local_rank

    # ===== Per-rank terrain dispatch (异构地形多卡训练) =====
    # 每张卡跑不同 terrain, 共享同一 reward/network, DDP 平均梯度 → 通用 generalist policy
    # 须配合 Rough-MoE-Teacher-Deeprobotics-M20-v0 这种统一 reward 的 task 用
    # 用户指定 rank 映射 (1-indexed in user's spec, 这里 0-indexed):
    #   rank 0 (1st): FLAT             — 纯平地
    #   rank 1 (2nd): STAIR_SLOPE      — 楼梯+斜坡 (上下两向, 最高 30cm 阶 / 45° 坡)
    #   rank 2 (3rd): PLATFORM (PIT/BOX) — 高台爬降
    #   rank 3 (4th): SCAN (HURDLE)    — 跨栏/挡板
    #   rank 4 (5th): GAP              — 跨沟
    #   rank 5 (6th): RAIL             — 跳栏 (0-40cm)
    #   rank 6 (7th): NOISE            — 随机起伏 (官方 ROUGH 默认参数)
    #   rank 7 (8th): GRID             — Mesh 离散方格 (官方 ROUGH 默认参数)
    if args.distributed and os.environ.get("PER_RANK_TERRAIN", "0") == "1":
        import isaaclab.terrains as _terrain_gen
        from isaaclab.terrains import TerrainGeneratorCfg as _TGC
        from rl_training.terrains.config.rough import (
            STAIR_SLOPE_TEACHER_TERRAINS_CFG,
            PLATFORM_TEACHER_TERRAINS_CFG,
            SCAN_TEACHER_TERRAINS_CFG,
            GAP_TEACHER_TERRAINS_CFG,
            STEPPING_STONES_TEACHER_TERRAINS_CFG,
            GAP_STONES_MIX_TEACHER_TERRAINS_CFG,
            RAIL_TEACHER_TERRAINS_CFG,
            NOISE_TEACHER_TERRAINS_CFG,
            GRID_TEACHER_TERRAINS_CFG,
        )
        # 平地 cfg (rank 0 专属): curriculum=False, 单 sub_terrain
        _FLAT_TERRAINS_CFG = _TGC(
            size=(12.0, 12.0), border_width=20.0,
            num_rows=10, num_cols=10, curriculum=False,
            sub_terrains={"flat": _terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
        )
        # rank → terrain 映射 (跟 PER_RANK_NAMES 默认顺序对齐)
        _RANK_TERRAIN_MAP = [
            _FLAT_TERRAINS_CFG,                  # rank 0: FLAT
            STAIR_SLOPE_TEACHER_TERRAINS_CFG,    # rank 1: STAIR_SLOPE
            PLATFORM_TEACHER_TERRAINS_CFG,       # rank 2: PLATFORM (pit/box)
            SCAN_TEACHER_TERRAINS_CFG,           # rank 3: SCAN (hurdle)
            GAP_STONES_MIX_TEACHER_TERRAINS_CFG, # rank 4: GAP+STONES 50/50 (MeshGap 单缝 + HfSteppingStones 多 hole 互补信号)
            RAIL_TEACHER_TERRAINS_CFG,           # rank 5: RAIL (0-40cm)
            NOISE_TEACHER_TERRAINS_CFG,          # rank 6: NOISE
            GRID_TEACHER_TERRAINS_CFG,           # rank 7: GRID
        ]
        # rank → {reward_term_name: new_weight}
        # =====================================================================
        # Per-rank reward weight overrides
        # =====================================================================
        # 每个 rank 跑不同地形, reward landscape 应该不同. 共享 critic 时这
        # 不可行 (V* 偏); SplitMoEPPO 现在已经做 per-rank critic
        # (commit eb2145d), 所以放心在这里给每个 rank 设独立 reward weight.
        #
        # 下面 rank 2 / rank 3 的 weight 直接镜像自专才 env_cfg:
        #   - rank 2 (PLATFORM)  ← teacher_platform_env_cfg.py
        #   - rank 3 (SCAN/hurdle) ← teacher_scan_env_cfg.py
        # 仅 mirror "与通才 base cfg 不同的 weight". params / functions 不动.
        #
        # rank 4 (STONES) 沿用之前 commit 46238ac 的 is_terminated=0; 后续可
        # 按 teacher_gap_env_cfg.py (commit d5ae4f4) 把 base_roll_l2 +
        # feet_height_body 也归零, 当前先保守.
        #
        # 注意: 只能 override 已经存在的 reward term. 如果某 term 被
        # disable_zero_weight_rewards() 在 base cfg 里已经删了, getattr
        # 拿不到, 会被 dispatch log 警告并跳过.
        # =====================================================================
        # =====================================================================
        # Per-rank Command Range / Reset Yaw overrides
        # =====================================================================
        # FLAT (rank 0) 是唯一允许 y 速度命令 + 全向随机朝向的 rank, 因为只有
        # 平地有侧移+原地转向训练价值. 其它 rank 都锁定:
        #   - lin_vel_y = (0, 0)  → 不发 y 速度命令
        #   - heading   = (0, 0)  → heading 命令永远朝前
        #   - reset yaw = (0, 0)  → spawn 时机器人朝前 (不随机)
        # 让其它地形的 policy 专注前进, 不被侧移/转向干扰. FLAT 反过来用全向
        # 训练 actor 学到通用机动性, 通过共享 actor 让其它地形也"会转弯"
        # (但 critic 仍 per-rank, 见 SplitMoEPPO).
        # =====================================================================
        import math as _math
        _LATERAL_ON  = {"lin_vel_y": (-1.0, 1.0), "heading": (-_math.pi, _math.pi)}
        _LATERAL_OFF = {"lin_vel_y": ( 0.0, 0.0), "heading": ( 0.0,       0.0)}
        _RANK_COMMAND_RANGE_OVERRIDES = {
            0: _LATERAL_ON,
            **{r: _LATERAL_OFF for r in range(1, 8)},
        }
        _YAW_RANDOM = {"yaw": (-_math.pi, _math.pi)}
        _YAW_FIXED  = {"yaw": ( 0.0,       0.0)}
        _RANK_RESET_POSE_OVERRIDES = {
            0: _YAW_RANDOM,
            **{r: _YAW_FIXED for r in range(1, 8)},
        }

        # =====================================================================
        # Per-rank Reward function (not weight) overrides
        # =====================================================================
        # Some rewards use a curriculum-gated function that returns 0 on flat
        # terrain (e.g. feet_air_time_curriculum has scale = clamp(level/5, 0, 1)
        # which is 0 at terrain level 0). On rank 0 (FLAT, level always 0), this
        # silently neuters the feet_air_time reward → robot learns pure skid-steer
        # turning. For pure-yaw command, we want the robot to *lift feet* and step
        # around, not roll. Swap rank 0's feet_air_time func to
        # feet_air_time_including_ang_z which:
        #   - has NO curriculum scale (works on flat)
        #   - triggers on ang_vel commands too (not just lin_vel)
        # =====================================================================
        import rl_training.tasks.manager_based.locomotion.velocity.mdp as _mdp
        _RANK_REWARD_FUNC_OVERRIDES = {
            0: {  # FLAT: enable step-turn instead of skid-steer
                "feet_air_time": _mdp.feet_air_time_including_ang_z,
            },
        }

        _RANK_REWARD_WEIGHT_OVERRIDES = {
            # rank 2: PLATFORM (pit/box) — 攀爬大落差, 需要放宽 roll/pitch、
            # 关掉 base_height_l2、抑制蹲走 (feet_height_body 加重)、关掉 upward
            2: {
                "ang_vel_xy_l2":          -0.01,   # base -0.05 → 放宽
                "base_height_l2":          0.0,   # base -0.3  → 攀爬时身高变化是必要的
                "feet_air_time":           1.5,   # base 1.0   → 鼓励大幅抬腿
                "feet_height_body":       -0.5,   # base -0.2  → 抑制蹲走 (was -0.2 here -0.5)
                "upward":                  0.0,   # base 0.05  → 攀爬必然 pitch, 这个反向信号
            },
            # rank 3: SCAN (hurdle) — 跨栏需要明确"先跳后蹲"信号, 关闭 air_time
            # 让 policy 学站姿冲撞而非反复跳, 同时强化 z 速度惩罚 (跨栏不能弹)
            3: {
                "lin_vel_z_l2":           -2.0,   # base -0.03 → SCAN: 跳栏不要"弹跳", 严罚 z 速度
                "base_height_l2":         -0.5,   # base -0.3  → 强化身高 (站直冲栏)
                "hipx_joint_pos_penalty": -0.6,   # base -0.5  → 略紧 hipx
                "hipy_joint_pos_penalty": -0.3,   # base -0.25 → 略紧 hipy
                "feet_air_time":           0.0,   # base 1.0   → 不奖励抬腿 (蹲伏冲栏)
                "feet_height_body":        0.0,   # base -0.2  → SCAN 蹲伏过栏杆是任务本身
                "upward":                  0.08,  # base 0.05  → 鼓励保持竖直
                "undesired_contacts":     -0.1,   # base -0.3  → 放宽; 栏杆轻触 ok
            },
            # rank 4: STONES (gap-crossing)
            4: {
                "is_terminated":           0.0,   # base -100  → 跳跃失败不毒打 (commit 46238ac 原 dict 那项)
            },
        }
        # 每 rank 把自己的 dispatch info append 到共享文件 (不 print 到 stdout, 避免被 8x 刷屏)
        _DISPATCH_FILE = "/tmp/per_rank_dispatch.txt"
        if local_rank == 0:
            # rank 0 启动时清空文件 (其他 rank 接着 append)
            with open(_DISPATCH_FILE, "w") as _f:
                _f.write(f"# Per-rank terrain dispatch ({int(os.environ.get('WORLD_SIZE', '1'))} ranks)\n")
        if local_rank < len(_RANK_TERRAIN_MAP):
            chosen = _RANK_TERRAIN_MAP[local_rank]
            env_cfg.scene.terrain.terrain_generator = chosen
            # Apply per-rank reward weight overrides (if any for this rank)
            _reward_overrides = _RANK_REWARD_WEIGHT_OVERRIDES.get(local_rank, {})
            _applied = []    # list of (name, prev_w, new_w)
            _skipped = []    # list of names that don't exist in env_cfg.rewards
            for _term_name, _new_w in _reward_overrides.items():
                _term = getattr(env_cfg.rewards, _term_name, None)
                if _term is None or not hasattr(_term, "weight"):
                    _skipped.append(_term_name)
                    continue
                _prev_w = _term.weight
                _term.weight = _new_w
                _applied.append((_term_name, _prev_w, _new_w))
            # Apply per-rank reward FUNCTION overrides (e.g. switch curriculum-gated
            # feet_air_time to ang-vel-aware variant on FLAT rank)
            _func_overrides = _RANK_REWARD_FUNC_OVERRIDES.get(local_rank, {})
            _func_applied = []
            _func_skipped = []
            for _term_name, _new_func in _func_overrides.items():
                _term = getattr(env_cfg.rewards, _term_name, None)
                if _term is None or not hasattr(_term, "func"):
                    _func_skipped.append(_term_name)
                    continue
                _prev_func_name = getattr(_term.func, "__name__", str(_term.func))
                _new_func_name  = getattr(_new_func, "__name__", str(_new_func))
                _term.func = _new_func
                _func_applied.append((_term_name, _prev_func_name, _new_func_name))
            # Apply per-rank command range overrides
            _cmd_overrides = _RANK_COMMAND_RANGE_OVERRIDES.get(local_rank, {})
            _cmd_applied = []
            _cmd_skipped = []
            if env_cfg.commands.base_velocity is not None and _cmd_overrides:
                _ranges = env_cfg.commands.base_velocity.ranges
                for _k, _v in _cmd_overrides.items():
                    if hasattr(_ranges, _k):
                        _prev = getattr(_ranges, _k)
                        setattr(_ranges, _k, _v)
                        _cmd_applied.append((_k, _prev, _v))
                    else:
                        _cmd_skipped.append(_k)
            # Apply per-rank reset pose_range overrides (only yaw for now)
            _reset_overrides = _RANK_RESET_POSE_OVERRIDES.get(local_rank, {})
            _reset_applied = []
            _reset_skipped = []
            if _reset_overrides:
                _pose_range = env_cfg.events.randomize_reset_base.params.get("pose_range", {})
                for _k, _v in _reset_overrides.items():
                    if _k in _pose_range:
                        _prev = _pose_range[_k]
                        _pose_range[_k] = _v
                        _reset_applied.append((_k, _prev, _v))
                    else:
                        _reset_skipped.append(_k)
            with open(_DISPATCH_FILE, "a") as _f:
                _f.write(f"[rank={local_rank}] terrain → "
                         f"{list(chosen.sub_terrains.keys())} "
                         f"(size={chosen.size}, num_rows={chosen.num_rows}, "
                         f"curriculum={chosen.curriculum})\n")
                for _name, _prev, _new in _applied:
                    _f.write(f"[rank={local_rank}] rewards.{_name}.weight: {_prev} → {_new}\n")
                for _name in _skipped:
                    _f.write(f"[rank={local_rank}] WARN: rewards.{_name} not found, override skipped\n")
                for _name, _prev, _new in _func_applied:
                    _f.write(f"[rank={local_rank}] rewards.{_name}.func: {_prev} → {_new}\n")
                for _name in _func_skipped:
                    _f.write(f"[rank={local_rank}] WARN: rewards.{_name}.func override skipped (no .func attr)\n")
                for _name, _prev, _new in _cmd_applied:
                    _f.write(f"[rank={local_rank}] commands.base_velocity.ranges.{_name}: {_prev} → {_new}\n")
                for _name in _cmd_skipped:
                    _f.write(f"[rank={local_rank}] WARN: commands.base_velocity.ranges.{_name} not found\n")
                for _name, _prev, _new in _reset_applied:
                    _f.write(f"[rank={local_rank}] reset.pose_range.{_name}: {_prev} → {_new}\n")
                for _name in _reset_skipped:
                    _f.write(f"[rank={local_rank}] WARN: reset.pose_range.{_name} not found\n")
        else:
            with open(_DISPATCH_FILE, "a") as _f:
                _f.write(f"[rank={local_rank}] WARN: out of RANK_TERRAIN_MAP range\n")
        # rank 0 等其他 rank 写完, 一次性打印汇总
        if is_master:
            import time as _time
            _time.sleep(2)  # 等其他 rank 完成 dispatch
            print("\n" + "=" * 80)
            print("PER-RANK TERRAIN DISPATCH (8 卡映射汇总)")
            print("=" * 80)
            try:
                with open(_DISPATCH_FILE, "r") as _f:
                    print(_f.read(), end="")
            except Exception as _e:
                print(f"  (failed to read {_DISPATCH_FILE}: {_e})")
            print("=" * 80 + "\n", flush=True)

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

    if is_master:
        print(f"\n[Info] Switching Policy Class to: SplitMoEActorCritic")
    train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
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

    # ===== Per-rank wandb logging via 双层 hook =====
    # 1. 每 rank hook alg.update() (所有 rank 都调) → 写自己的 jsonl 文件
    # 2. master 在 runner.log() 里读所有 rank 的文件 → 写 wandb
    # 不用 dist.all_gather, 避免 master-only log 导致的 collective 死锁
    if args.distributed and os.environ.get("PER_RANK_TERRAIN", "0") == "1":
        import json as _json
        RANK_NAMES = os.environ.get(
            "PER_RANK_NAMES",
            "FLAT,STAIR_SLOPE,PLATFORM,SCAN,STONES,RAIL,NOISE,GRID",
        ).split(",")
        _world_size = int(os.environ.get("WORLD_SIZE", "1"))
        _rank_name = RANK_NAMES[local_rank] if local_rank < len(RANK_NAMES) else f"rank{local_rank}"
        _per_rank_path = lambda r: (
            f"/tmp/per_rank_train_"
            f"{RANK_NAMES[r] if r < len(RANK_NAMES) else f'rank{r}'}"
            f"_rank{r}.jsonl"
        )
        # 每 rank 启动时清空自己的文件
        with open(_per_rank_path(local_rank), "w") as _f:
            pass

        # 找到 base env (穿过 wrappers)
        _base_env = env
        while hasattr(_base_env, "env"):
            _base_env = _base_env.env
        if hasattr(_base_env, "unwrapped"):
            _base_env = _base_env.unwrapped

        # ----- Hook 1: alg.update (每 rank 都调) → 写自己 jsonl -----
        # 用 closure 维护 iter 计数 (alg.update 没有 it 参数)
        _iter_counter = [0]
        _original_update = runner.alg.update

        def _update_with_dump(*a, **kw):
            result = _original_update(*a, **kw)
            try:
                it = _iter_counter[0]
                _iter_counter[0] += 1
                m = {"iter": it, "rank": local_rank, "name": _rank_name}
                if hasattr(_base_env, "scene") and hasattr(_base_env.scene, "terrain") \
                   and hasattr(_base_env.scene.terrain, "terrain_levels"):
                    tl = _base_env.scene.terrain.terrain_levels.float()
                    m["terrain_level_mean"] = float(tl.mean().item())
                    m["terrain_level_max"] = float(tl.max().item())
                # Per-rank PPO loss components (return dict from alg.update).
                # 主要用来看 per-rank critic 是否真的让各 rank V̂ 分化:
                #   - value_function: critic MSE (low + diverging across ranks = healthy)
                #   - surrogate: PPO actor loss
                #   - entropy: action distribution entropy
                if isinstance(result, dict):
                    for k in ("value_function", "surrogate", "entropy"):
                        if k in result and isinstance(result[k], (int, float)):
                            m[k] = float(result[k])
                # episode 长度 / 奖励直接读 env (alg.update 此时还有 rollout 数据)
                # episode_sums / lenbuffer 数据保存在 runner 自己的 deque, 不能从 alg 拿到
                # → 这里只写 terrain_level + ppo loss. ep_reward / ep_length 走 hook 2 (master 自己有)
                with open(_per_rank_path(local_rank), "a") as f:
                    f.write(_json.dumps(m) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                print(f"[per_rank dump rank {local_rank}] failed: {e}", flush=True)
            return result

        runner.alg.update = _update_with_dump

        # ----- Hook 2: master 的 log() → 读所有 rank 的 jsonl, 写 wandb -----
        if is_master:
            _original_log = runner.log

            def _log_with_per_rank(locs, *a, **kw):
                ret = _original_log(locs, *a, **kw)
                try:
                    it = locs.get("it", runner.current_learning_iteration)
                    # 读每 rank 最新一行
                    for r in range(_world_size):
                        name = RANK_NAMES[r] if r < len(RANK_NAMES) else f"rank{r}"
                        path = _per_rank_path(r)
                        try:
                            with open(path, "r") as f:
                                lines = f.readlines()
                            if not lines:
                                continue
                            last = _json.loads(lines[-1])
                            for k, v in last.items():
                                if k in ("iter", "rank", "name"):
                                    continue
                                if isinstance(v, (int, float)) and runner.writer is not None:
                                    runner.writer.add_scalar(f"PerRank/{k}/{name}", v, it)
                        except Exception:
                            continue
                    # master 自己还能 dump 一些只它有的 metrics (rewbuffer/lenbuffer/ep_infos)
                    # 这部分作为 master rank 的"代理 per-rank metric", 写到 PerRank/<master_name>/
                    master_name = RANK_NAMES[0] if 0 < len(RANK_NAMES) else "rank0"
                    if locs.get("rewbuffer") and len(locs["rewbuffer"]) > 0:
                        v = float(sum(locs["rewbuffer"]) / len(locs["rewbuffer"]))
                        if runner.writer is not None:
                            runner.writer.add_scalar(f"PerRank/ep_reward/{master_name}", v, it)
                    if locs.get("lenbuffer") and len(locs["lenbuffer"]) > 0:
                        v = float(sum(locs["lenbuffer"]) / len(locs["lenbuffer"]))
                        if runner.writer is not None:
                            runner.writer.add_scalar(f"PerRank/ep_length/{master_name}", v, it)
                except Exception as e:
                    print(f"[per_rank log master] failed: {e}", flush=True)
                return ret

            runner.log = _log_with_per_rank

        if is_master:
            print(f"[INFO] PerRank logging: each rank → /tmp/per_rank_train_<NAME>_rank<N>.jsonl")
            print(f"       master 读所有 → wandb 'PerRank/*/*' panels")

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