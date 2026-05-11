# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import math
import os
import torch # Added torch for depth calculation
import torch.nn.functional as F

# ==============================================================================
# Scan sim2real augmentation toggle
# ==============================================================================
# 控制 multi_layer_scan 的增强 (距离 ramp / 随机 dropout / latency / 噪声放大).
# bug 修复 (sensor._ray_starts_w 替代 data.pos_w) **始终生效** — 那是正确性修复, 不可关闭.
#   SCAN_AUG=1 (默认): 全部增强 + 噪声 ±0.02
#   SCAN_AUG=0       : 原始逻辑 (hard <0.3m 截断, 无 dropout/latency, 噪声 ±0.005)
# 用法:
#   python train_moe.py ...                    # 默认开增强
#   SCAN_AUG=0 python train_moe.py ...         # 控制变量, baseline
SCAN_AUG_ENABLED = os.environ.get("SCAN_AUG", "1") == "1"
_SCAN_NOISE_AMP = 0.02 if SCAN_AUG_ENABLED else 0.005
print(f"[moe_teacher_env_cfg] SCAN_AUG_ENABLED = {SCAN_AUG_ENABLED}  (noise=±{_SCAN_NOISE_AMP})")
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.sensors import RayCasterCfg, patterns, RayCasterCameraCfg
from rl_training.sensors import HemisphericalLidarPatternCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    MySceneCfg, # Import base scene config
)
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import MultiMeshRayCasterCfg
##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_CFG  # isort: skip
from rl_training.terrains.config.rough import *
from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import TerrainAwareVelocityCommandCfg
# ==============================================================================
# Helper Functions (Modified for Sim2Real & CNN)
# ==============================================================================

def euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cx = math.cos(roll * 0.5)
    sx = math.sin(roll * 0.5)
    cy = math.cos(pitch * 0.5)
    sy = math.sin(pitch * 0.5)
    cz = math.cos(yaw * 0.5)
    sz = math.sin(yaw * 0.5)
    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    return (w, x, y, z)

def process_lidar_data(depths: torch.Tensor, is_student: bool) -> torch.Tensor:
    """
    终极优化的 Lidar 数据处理管道（修复维度拼接报错）
    """
    # 1. 基础归一化 (Tanh)
    # 正常物理距离 [0, +inf) -> 被映射到 [0.0, 1.0)
    scale = 10.0
    normalized_depths = torch.tanh(depths / scale)
    
    # 2. 盲区处理 (仅Student)
    if is_student:
        blind_zone_fill = torch.tanh(torch.tensor(5.0 / scale))
        normalized_depths = torch.where(
            depths < 0.3,
            torch.full_like(normalized_depths, blind_zone_fill.item()),
            normalized_depths
        )
    return normalized_depths

def lidar_depth_scan_teacher(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    rel_vec = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    depths = torch.norm(rel_vec, dim=-1)
    return process_lidar_data(depths, is_student=False)

def lidar_depth_scan_student(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    rel_vec = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    depths = torch.norm(rel_vec, dim=-1)
    return process_lidar_data(depths, is_student=True)

def flattened_image(env, sensor_cfg: SceneEntityCfg, data_type: str, normalize: bool = False) -> torch.Tensor:
    img = mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize)
    return img.flatten(start_dim=1)

def teacher_camera_depth(env, sensor_cfg: SceneEntityCfg, data_type: str, normalize: bool = False) -> torch.Tensor:
    img = mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False)
    depths = img.flatten(start_dim=1)
    return process_lidar_data(depths, is_student=False)

def student_camera_depth(env, sensor_cfg: SceneEntityCfg, data_type: str, normalize: bool = False) -> torch.Tensor:
    img = mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False)
    depths = img.flatten(start_dim=1)
    return process_lidar_data(depths, is_student=True)
def multi_layer_scan(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """半球 LiDAR scan + sim2real 增强 (匹配真机 lidar_to_scan.cpp 行为).

    增强项:
      1) 距离渐变盲区: depth ≤ 0.1m P=1, 0.1-0.35m 线性概率, ≥ 0.35m P=0
      2) 随机 bin dropout: per-env [0.3, 0.5] 概率, 每 bin 独立采样
         (模拟 Robosense Airy Lissajous 扫描密度不均, 真机大量 bin 是 no-hit)
      3) per-episode 随机延迟 0-200ms (50Hz → 0-10 步), reset 重采样
      4) 归一化 [0, 1], 盲区/no-hit = 1.0 (= 真机 max_distance 2.5m)
    输出后 ObsTerm 还会叠加 Unoise (在归一化空间)。
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    # 用 _ray_starts_w (真实射线起点世界坐标, 含 OffsetCfg 偏移), 而不是 data.pos_w (= base_link).
    # 验证脚本 verify_lidar_extrinsics.py 证实 data.pos_w 报的是父 prim, 用它会让深度系统性大 ~0.32m,
    # 与真机 lidar_to_scan.cpp (从 sensor offset 计算) 之间产生 sim2real 偏差。
    rel_vec = sensor.data.ray_hits_w - sensor._ray_starts_w
    depths = torch.norm(rel_vec, dim=-1)
    depths = torch.nan_to_num(depths, posinf=2.5, neginf=2.5, nan=2.5)

    # ===== SCAN_AUG=0 baseline 路径: 原始逻辑 (硬 <0.3m 盲区, 无 dropout/latency) =====
    if not SCAN_AUG_ENABLED:
        normalized = torch.clip(depths / 2.5, 0.0, 1.0)
        normalized = torch.where(depths < 0.3, torch.ones_like(normalized), normalized)
        return normalized

    num_envs, num_rays = depths.shape
    device = depths.device

    # --- 1. 距离渐变盲区 (替代原 <0.3m 硬阈值) ---
    blind_prob_dist = torch.clamp((0.35 - depths) / (0.35 - 0.1), 0.0, 1.0)
    mask_dist = torch.rand_like(depths) < blind_prob_dist

    # --- 2. 随机 bin dropout (per-env 30-50% rate) ---
    rate = 0.3 + 0.2 * torch.rand((num_envs, 1), device=device)
    mask_random = torch.rand_like(depths) < rate

    # 合并盲区 → 填 max_distance (2.5m)
    blind_mask = mask_dist | mask_random
    depths = torch.where(blind_mask, torch.full_like(depths, 2.5), depths)

    # --- 3. 0-200ms per-episode 延迟 (50Hz → 0-10 步, head 指针式环形 buffer) ---
    MAX_LATENCY = 10  # 200ms @ step_dt=0.02s
    BUF_LEN = MAX_LATENCY + 1
    buf_attr = f"_scan_buf_{sensor_cfg.name}"
    head_attr = f"_scan_head_{sensor_cfg.name}"
    delay_attr = f"_scan_delay_{sensor_cfg.name}"

    if not hasattr(env, buf_attr):
        setattr(env, buf_attr, depths.unsqueeze(1).repeat(1, BUF_LEN, 1).clone())
        setattr(env, head_attr, 0)
        setattr(env, delay_attr, torch.randint(0, BUF_LEN, (num_envs,), device=device))

    buf = getattr(env, buf_attr)
    head = getattr(env, head_attr)
    delay = getattr(env, delay_attr)

    # 写入当前帧 → head 槽位
    buf[:, head, :] = depths

    # 读 (head - delay) mod BUF_LEN
    read_idx = (head - delay) % BUF_LEN
    env_arange = torch.arange(num_envs, device=device)
    delayed = buf[env_arange, read_idx, :].clone()

    # 推进 head
    setattr(env, head_attr, (head + 1) % BUF_LEN)

    # reset 处理: 清空对应行的 buffer + 重采样 delay + 输出当前帧
    if hasattr(env, "reset_buf"):
        reset_idx = torch.where(env.reset_buf > 0)[0]
        if reset_idx.numel() > 0:
            buf[reset_idx] = depths[reset_idx].unsqueeze(1).repeat(1, BUF_LEN, 1)
            delay[reset_idx] = torch.randint(0, BUF_LEN, (reset_idx.numel(),), device=device)
            delayed[reset_idx] = depths[reset_idx]

    # --- 4. 归一化到 [0, 1] ---
    return torch.clip(delayed / 2.5, 0.0, 1.0)

def height_scan_sim2real(
    env, 
    sensor_cfg: SceneEntityCfg, 
    offset: float = 0.5, 
    mask_prob: float = 0.15, 
    min_latency: int = 1, 
    max_latency: int = 3,
    smooth_kernel_size: int = 3, 
    max_drift_pixels: int = 2,
    grid_length: int = 17, 
    min_noise_amp: float = 0.1
) -> torch.Tensor:
    """
    终极 Sim-to-Real 高程图处理 (Batched Vectorized 版本)
    新增修复：
    1. 使用 O(1) 的指针式环形缓冲区替代低效的 torch.roll 内存拷贝。
    2. 修复 reset_buf 维度挤压隐患，使用更安全的 torch.where 提取索引。
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    num_envs = env.num_envs
    device = sensor.data.pos_w.device
    
    # 1. 获取当前完美的物理深度 (num_envs, num_rays)
    current_depths = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    num_rays = current_depths.shape[-1]

    # =========================================================================
    # 防崩溃与严格维度校验
    # =========================================================================
    if num_rays % grid_length != 0:
        raise ValueError(
            f"[Sim2Real Error] 射线总数 ({num_rays}) 无法被 grid_length ({grid_length}) 整除！"
            "请检查 RayCaster 的 pattern_cfg 分辨率设置。"
        )
    
    grid_width = num_rays // grid_length
    depths_2d = current_depths.view(num_envs, 1, grid_length, grid_width)

    # =========================================================================
    # B. 边缘倒角 (Edge Chamfering) - Replicate Padding
    # =========================================================================
    if smooth_kernel_size > 1:
        pad_size = smooth_kernel_size // 2
        padded_depths = F.pad(depths_2d, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
        depths_2d = F.avg_pool2d(padded_depths, kernel_size=smooth_kernel_size, stride=1, padding=0)

    # =========================================================================
    # C. 空间漂移 (Spatial Drift) - 全环境并行独立采样 + 高级索引切片
    # =========================================================================
    if max_drift_pixels > 0:
        shift_x = torch.randint(-max_drift_pixels, max_drift_pixels + 1, (num_envs,), device=device)
        shift_y = torch.randint(-max_drift_pixels, max_drift_pixels + 1, (num_envs,), device=device)
        
        padded_for_drift = F.pad(depths_2d, (max_drift_pixels, max_drift_pixels, max_drift_pixels, max_drift_pixels), mode='replicate')
        
        start_x = max_drift_pixels - shift_x
        start_y = max_drift_pixels - shift_y
        
        b = torch.arange(num_envs, device=device).view(-1, 1, 1)
        grid_x = torch.arange(grid_length, device=device).view(1, -1, 1) + start_x.view(-1, 1, 1)
        grid_y = torch.arange(grid_width, device=device).view(1, 1, -1) + start_y.view(-1, 1, 1)
        
        depths_2d = padded_for_drift[b, 0, grid_x, grid_y].unsqueeze(1)

    # =========================================================================
    # 展平回 1D 进行 Element-wise 处理
    # =========================================================================
    current_depths = depths_2d.reshape(num_envs, num_rays)

    # =========================================================================
    # A. 非线性椒盐噪声 (Nonlinear Salt-and-Pepper Noise)
    # =========================================================================
    if mask_prob > 0.0:
        p = mask_prob / 2.0 
        rand_tensor = torch.rand_like(current_depths)
        
        M = current_depths.amax(dim=-1, keepdim=True)
        m = current_depths.amin(dim=-1, keepdim=True)
        noise_amp = torch.clamp(M - m, min=min_noise_amp)
        
        mask_high = rand_tensor < p
        mask_low = (rand_tensor >= p) & (rand_tensor < mask_prob)
        
        noise_high = M + torch.rand_like(current_depths) * noise_amp
        noise_low = m - torch.rand_like(current_depths) * noise_amp
        
        current_depths = torch.where(mask_high, noise_high, current_depths)
        current_depths = torch.where(mask_low, noise_low, current_depths)
        
    if max_latency <= 0:
        return current_depths

    # =========================================================================
    # D. 纯 GPU 高效指针环形延迟 Buffer (修复核心)
    # =========================================================================
    buffer_len = max_latency + 1
    
    # 初始化 Buffer 和 指针
    if not hasattr(env, "_sim2real_height_buffer"):
        env._sim2real_height_buffer = current_depths.unsqueeze(1).repeat(1, buffer_len, 1)
        env._sim2real_buffer_head = 0  # 记录当前写入位置的指针

    # 【修复 D】使用 torch.where 替代 .nonzero().squeeze(-1)，避免维度变化导致的挤压崩溃
    if hasattr(env, "reset_buf"):
        reset_idx = torch.where(env.reset_buf)[0]
        if len(reset_idx) > 0:
            # 环境重置时，用当前无延迟的清晰帧填满该环境的所有历史 Buffer
            env._sim2real_height_buffer[reset_idx, :, :] = current_depths[reset_idx].unsqueeze(1)

    # 【修复 C】通过循环指针移动，替代 torch.roll 的暴力内存拷贝
    env._sim2real_buffer_head = (env._sim2real_buffer_head + 1) % buffer_len
    head_idx = env._sim2real_buffer_head

    # 将最新的一帧写入指针所指向的位置 (O(1) 复杂度)
    env._sim2real_height_buffer[:, head_idx, :] = current_depths

    # 为每个环境独立采样延迟帧数
    delays = torch.randint(
        min_latency, 
        max_latency + 1, 
        (num_envs,), 
        device=device
    )
    
    # 计算每个环境需要读取的历史索引 (处理负数取模，自动回绕)
    read_indices = (head_idx - delays) % buffer_len
    
    # 使用高级索引高效并行提取延迟帧
    batch_indices = torch.arange(num_envs, device=device)
    delayed_depths = env._sim2real_height_buffer[batch_indices, read_indices, :]

    return delayed_depths

@configclass
class DeeproboticsM20ActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=20.0, use_default_offset=True, clip=None, preserve_order=True
    )

@configclass
class DeeproboticsM20RewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    joint_mirror_lr = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,  # 由 actor sym_loss (LR 等变约束) 替代，避免双重约束抑制动态步态
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["fl_(hipy|knee).*", "fr_(hipy|knee).*"],
                ["hl_(hipy|knee).*", "hr_(hipy|knee).*"],
            ]
        }
    )
    action_mirror_lr = RewTerm(
        func=mdp.action_mirror,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["fl_(hipy|knee).*", "fr_(hipy|knee).*"],
                ["hl_(hipy|knee).*", "hr_(hipy|knee).*"],
            ]
        }
    )
    joint_mirror_diag = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
                ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
            ]
        }
    )
    joint_mirror_fb = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.0,  
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["fl_(hipy|knee).*", "hl_(hipy|knee).*"], 
                ["fr_(hipy|knee).*", "hr_(hipy|knee).*"], 
            ]
        }
    )
    feet_air_time_long = RewTerm(
        func=mdp.feet_air_time_curriculum,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )
    base_roll_l2 = RewTerm(
        func=mdp.base_roll_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    undesired_contacts_knee = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee"),
            "threshold": 1.0,
        },
    )
@configclass
class DeeproboticsM20SceneCfg(MySceneCfg):
    pass

# ==============================================================================
# 自定义观测配置类
# ==============================================================================
@configclass
class DeeproboticsM20ObservationsCfg:
    """Observation specifications for the M20 environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """纯粹的 Teacher 本体感知组 (含真实线速度, 无高程图)"""
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=0.25, 
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05, 
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        base_lin_vel = None
        def __post_init__(self):
            # Teacher Actor 使用完全无噪声的本体感觉，以追求性能上限
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoisyElevationCfg(ObsGroup):
        """提供给 ScanAE 的环境感知组（elevation map + LIDAR 双流）"""
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # --- 前后两个半球 LidarPattern sensor (16 ch × 31 az = 496/sensor, 共 992) ---
        # 噪声幅度由 SCAN_AUG 环境变量控制 (±0.02 增强 / ±0.005 baseline)
        forward_scan = ObsTerm(func=multi_layer_scan, params={"sensor_cfg": SceneEntityCfg("forward_lidar")}, noise=Unoise(n_min=-_SCAN_NOISE_AMP, n_max=_SCAN_NOISE_AMP))
        backward_scan = ObsTerm(func=multi_layer_scan, params={"sensor_cfg": SceneEntityCfg("backward_lidar")}, noise=Unoise(n_min=-_SCAN_NOISE_AMP, n_max=_SCAN_NOISE_AMP))
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class BlindStudentPolicyCfg(ObsGroup):
        """纯粹的 Student 本体感知组 (无真实线速度, 无高程图)"""
        base_lin_vel = None
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=0.25, 
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05, 
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = None  # blind by design — no elevation map

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StudentPolicyCfg(BlindStudentPolicyCfg):
        pass

    @configclass    
    class CriticCfg(PolicyCfg):
        """Critic 获取与 Teacher Actor 相同的本体感觉维度"""
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-100.0, 100.0),
            scale=1.0, 
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-100.0, 100.0),
            scale=0.25, 
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        terrain_level = ObsTerm(
            func=mdp.terrain_level_normalized,
            scale=1.0,
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EstimatorCfg(ObsGroup):
        history_length = 15
        flatten_history_dim = True
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=0.25, 
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05, 
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        def __post_init__(self):
            self.enable_corruption = True # VAE 获取包含噪声的历史信息
            self.concatenate_terms = True
    @configclass
    class PretrainCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = None  # blind by design — pretrain group has no elevation map

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
    blind_student_policy: BlindStudentPolicyCfg = BlindStudentPolicyCfg()
    student_policy: StudentPolicyCfg = StudentPolicyCfg() 
    critic: CriticCfg = CriticCfg()
    estimator: EstimatorCfg = EstimatorCfg()
    noisy_elevation: NoisyElevationCfg = NoisyElevationCfg()
    pretraincfg: PretrainCfg = PretrainCfg()

@configclass
class DeeproboticsM20MoETeacherEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: DeeproboticsM20ActionsCfg = DeeproboticsM20ActionsCfg()
    rewards: DeeproboticsM20RewardsCfg = DeeproboticsM20RewardsCfg()
    observations: DeeproboticsM20ObservationsCfg = DeeproboticsM20ObservationsCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_wheel"

    # fmt: off
    leg_joint_names = [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    ]
    wheel_joint_names = [
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    ]
    hipx_joint_names = [
        "fl_hipx_joint", "fr_hipx_joint", "hl_hipx_joint", "hr_hipx_joint",
    ]
    hipy_joint_names = [
        "fl_hipy_joint", "fr_hipy_joint", "hl_hipy_joint", "hr_hipy_joint",
    ]
    knee_joint_names = [
        "fl_knee_joint", "fr_knee_joint", "hl_knee_joint", "hr_knee_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.robot = DEEPROBOTICS_M20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # height_scanner: fix prim_path for M20 (base_link, not base)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # height_scanner_base (9-pt grid for base height reward) 仍保留
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        obs_groups_to_process = [
            self.observations.policy,
            self.observations.blind_student_policy,
            self.observations.student_policy,
            self.observations.critic,
            self.observations.estimator,
            self.observations.noisy_elevation,
            self.observations.pretraincfg,
        ]

        for obs_group in obs_groups_to_process:
            if obs_group is None:
                continue
            if hasattr(obs_group, "joint_pos") and obs_group.joint_pos is not None:
                obs_group.joint_pos.func = mdp.joint_pos_rel_without_wheel
                obs_group.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg("robot", joint_names=self.wheel_joint_names)
                obs_group.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            if hasattr(obs_group, "joint_vel") and obs_group.joint_vel is not None:
                obs_group.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.observations.blind_student_policy.base_lin_vel = None
        self.observations.student_policy.base_lin_vel = None

        self.observations.pretraincfg.base_lin_vel.scale = 2.0
        self.observations.pretraincfg.base_ang_vel.scale = 0.25
        self.observations.pretraincfg.joint_pos.scale = 1.0
        self.observations.pretraincfg.joint_vel.scale = 0.05
        self.observations.pretraincfg.base_lin_vel = None
        self.actions.joint_pos.scale = {".*_hipx_joint": 0.125, "^(?!.*_hipx_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        # ground terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=MOE_ROUGH_TERRAINS_CFG,
            max_init_terrain_level=1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        self.scene.terrain.terrain_generator = MOE_ROUGH_TERRAINS_CFG2
        # 拓宽 friction 范围 [0.6, 1.2] → [0.4, 1.4]: 覆盖更滑/更粘的真机平台表面 (爬高台关键依赖轮子边缘抓地)
        if(self.scene.terrain.terrain_generator == MOE_ROUGH_TERRAINS_CFG):
            self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.4, 1.4]
            self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.4, 1.4]
            self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]
        elif(self.scene.terrain.terrain_generator == MOE_ROUGH_TERRAINS_CFG2):
            self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.4, 1.4]
            self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.4, 1.4]
            self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]
        else:
            self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.4, 1.4]
            self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.4, 1.4]
            self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]
        # self.events.randomize_rigid_body_material.params["static_friction_range"] = [1.0, 1.0]
        # self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [1.0, 1.0]
        # self.events.randomize_rigid_body_material.params["restitution_range"] = [0.7, 0.7]
        
        FRONT_LIDAR_POS = (0.32028, 0.0, -0.013)
        REAR_LIDAR_POS = (-0.32028, 0.0, -0.013)

        # 真半球 pattern (匹配 mujoco sim2sim & Robosense Airy)
        # pole = sensor-local +X (boresight)，polar ∈ [0, π/2]，azimuth ∈ [-π, π]
        # 单 sensor: 16 polar × 31 azimuth = 496 ray
        # 前 + 后 共 992 ray，覆盖整球
        SCAN_PATTERN = HemisphericalLidarPatternCfg(num_polar=16, num_azimuth=31)
        SCAN_MESHES = ["/World/ground"]

        # 前向雷达：朝 +x，azimuth 中心对齐 +x 方向，无额外旋转
        fwd_sensor = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=FRONT_LIDAR_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            ray_alignment="base",
            pattern_cfg=SCAN_PATTERN,
            max_distance=2.5,
            debug_vis=False,
            reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )
        fwd_sensor.update_period = 0.1
        self.scene.forward_lidar = fwd_sensor

        # 后向雷达：朝 -x，绕 z 轴旋 180°；quat = (cos(90°), 0, 0, sin(90°)) = (0, 0, 0, 1)
        bwd_sensor = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=REAR_LIDAR_POS, rot=(0.0, 0.0, 0.0, 1.0)),
            ray_alignment="base",
            pattern_cfg=SCAN_PATTERN,
            max_distance=2.5,
            debug_vis=False,
            reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )
        bwd_sensor.update_period = 0.1
        self.scene.backward_lidar = bwd_sensor
        # Rewards
        self.rewards.is_terminated.weight = -100
        self.rewards.lin_vel_z_l2.weight = -0.03
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_roll_l2.weight = -10.0
        self.rewards.base_height_l2.weight = -0.3   # 0 → -0.3 (姿态回弹: flat 上 cost≈0, rough 上轻度激活)
        self.rewards.base_height_l2.params["target_height"] = 0.5
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -4e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -1e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.hipx_joint_pos_penalty.weight = -0.5
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.hipy_joint_pos_penalty.weight = -0.25
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -0.1
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = 0.0  # sym_loss 已替代
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        self.rewards.action_mirror.weight = 0.0
        self.rewards.action_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.undesired_contacts.weight = -0.3
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hipx", ".*_hipy"]
        self.rewards.undesired_contacts_knee.weight = -0.1
        self.rewards.undesired_contacts_knee.params["sensor_cfg"].body_names = [".*_knee"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 4.0 # 1.8 -> 3.0 -> 4.0 (对齐 PLATFORM)
        self.rewards.track_ang_vel_z_exp.weight = 3.0 # 1.2 -> 2.0 -> 3.0 (对齐 PLATFORM)
        self.rewards.track_lin_vel_xy_pre_exp.weight = 0
        self.rewards.track_ang_vel_z_pre_exp.weight = 0

        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_long.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.3
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -0.2   # 0 → -0.2 (站姿回弹, 防止 knee 在 flat 上残余弯曲)
        self.rewards.feet_height_body.params["target_height"] = -0.48  # was -0.4 (默认站姿实测 -0.44, 取 -0.48 略偏站直)
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        self.rewards.upward.weight = 0.05   # 0.0 → 0.05 (轻度保持 z 朝上, 防止 pitch/roll 漂移)

        if self.__class__.__name__ == "DeeproboticsM20MoETeacherEnvCfg":
            self.disable_zero_weight_rewards()
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        # self.terminations.illegal_contact = None
        self.terminations.bad_orientation_2 = None

        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.1, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.5, 1.0)

        # 改回不带 0.5 阈值过滤的 UniformVelocityCommandCfg, 否则 (1e-4, 0.5)
        # 区间的命令会被强制重采到 [0.5, 1.5], 让课程在 X 上前 ~70% 训练空跑.
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=0.85,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(0.0, 0.0),
            ),
        )
        # ------------------------------Commands------------------------------
        # 课程指令采样策略
        # self.commands.base_velocity = TerrainAwareVelocityCommandCfg(
        #     asset_name="robot",
        #     resampling_time_range=(8.0, 12.0),
        #     rel_standing_envs=0.05,
        #     rel_heading_envs=1.0,
        #     heading_command=True,
        #     heading_control_stiffness=0.5,
        #     debug_vis=False,
            
        #     ranges=TerrainAwareVelocityCommandCfg.Ranges(
        #         lin_vel_x=(-2.0, 2.0),
        #         lin_vel_y=(-1.5, 1.5),
        #         ang_vel_z=(-1.5, 1.5),
        #         heading=(-math.pi, math.pi)
        #     ),
            
        #     terrain_level_threshold=10,
        #     easy_ranges=TerrainAwareVelocityCommandCfg.Ranges(
        #         # easy_ranges 其实成了备用字段，实际简单地形用的是受课程控制的 ranges
        #         lin_vel_x=(-2.0, 2.0),
        #         lin_vel_y=(-1.5, 1.5),
        #         ang_vel_z=(-1.5, 1.5),
        #         heading=(-math.pi, math.pi)
        #     ),
        #     hard_ranges=TerrainAwareVelocityCommandCfg.Ranges(
        #         lin_vel_x=(-1.5, 1.5),   # 困难地形上限锁定 1.5
        #         lin_vel_y=(0.0, 0.0),    # 困难地形不侧移
        #         ang_vel_z=(-0.5, 0.5),   # 困难地形角速度收窄到 0.5 (对齐你之前源码里硬编码的想法)
        #         heading=(-math.pi, math.pi)
        #     )
        # )
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp_curriculum
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_exp_curriculum
        # self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        # self.curriculum.command_levels.params["range_multiplier"] = (1.0, 1.0)
        # override rewards
        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # self.scene.terrain2 = None
        # # no terrain curriculum
        # self.curriculum.terrain_levels = None
        # self.rewards.lin_vel_z_l2.func = mdp.lin_vel_z_l2
        # self.rewards.feet_air_time.func = mdp.feet_air_time_including_ang_z
        # self.rewards.base_height_l2.func = mdp.base_height_l2

# ==============================================================================
# 派生验证环境配置 (用于消融实验)
# ==============================================================================

@configclass
class DeeproboticsM20MoETeacherEnvCfg_EleOnly(DeeproboticsM20MoETeacherEnvCfg):
    """仅使用 Elevation Map 的环境"""
    def __post_init__(self):
        super().__post_init__()
        # 禁用半球 LidarPattern（保留 elevation map）
        self.observations.noisy_elevation.forward_scan = None
        self.observations.noisy_elevation.backward_scan = None
        if self.__class__.__name__ == "DeeproboticsM20MoETeacherEnvCfg_EleOnly":
            self.disable_zero_weight_rewards()

@configclass
class DeeproboticsM20MoETeacherEnvCfg_ScanOnly(DeeproboticsM20MoETeacherEnvCfg):
    """仅使用 MultiLayer Scan 的环境"""
    def __post_init__(self):
        super().__post_init__()
        if self.__class__.__name__ == "DeeproboticsM20MoETeacherEnvCfg_ScanOnly":
            self.disable_zero_weight_rewards()