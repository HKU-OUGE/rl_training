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