# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import math
import torch # Added torch for depth calculation
import torch.nn.functional as F
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.sensors import RayCasterCfg, patterns, RayCasterCameraCfg
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
from rl_training.sensors import (
    HemisphericalArcPatternCfg,
    MultiPitchArcPatternCfg,
    SinglePitchArcPatternCfg,
)

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
    # 用 _ray_starts_w (真实射线起点世界坐标, 含 OffsetCfg 偏移), 而不是 data.pos_w (= base_link).
    # data.pos_w 报的是父 prim, 用它会让深度系统性大 ~0.32m, 与真机 lidar_to_scan.cpp 之间产生 sim2real 偏差。
    rel_vec = sensor.data.ray_hits_w - sensor._ray_starts_w
    depths = torch.norm(rel_vec, dim=-1)
    return process_lidar_data(depths, is_student=False)

def lidar_depth_scan_student(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    # 用 _ray_starts_w (真实射线起点世界坐标, 含 OffsetCfg 偏移), 而不是 data.pos_w (= base_link).
    # data.pos_w 报的是父 prim, 用它会让深度系统性大 ~0.32m, 与真机 lidar_to_scan.cpp 之间产生 sim2real 偏差。
    rel_vec = sensor.data.ray_hits_w - sensor._ray_starts_w
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
    """处理多层雷达扫描，输出归一化的距离数组 (严格区分安全区与盲区)"""
    sensor = env.scene.sensors[sensor_cfg.name]

    # 用 _ray_starts_w (真实射线起点世界坐标, 含 OffsetCfg 偏移), 而不是 data.pos_w (= base_link).
    # data.pos_w 报的是父 prim, 用它会让深度系统性大 ~0.32m, 与真机 lidar_to_scan.cpp 之间产生 sim2real 偏差。
    rel_vec = sensor.data.ray_hits_w - sensor._ray_starts_w
    depths = torch.norm(rel_vec, dim=-1)
    
    # 将 NaN 和 inf 视为安全距离 (= sensor max_distance 2.5m, 匹配真机 no-hit 行为)
    depths = torch.nan_to_num(depths, posinf=2.5, neginf=2.5, nan=2.5)

    # 归一化：[0, 2.5] 映射到 [0.0, 1.0] (近场聚焦, 跟 main 一致)
    normalized_depths = torch.clip(depths / 2.5, 0.0, 1.0)

    # 盲区覆写：物理距离 < 0.3m 填充为最大量程归一化值 (匹配真机 no-hit = 2.5m)
    normalized_depths = torch.where(
        depths < 0.3,
        torch.full_like(normalized_depths, 1.0),
        normalized_depths
    )
    return normalized_depths

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
        weight=-0.03,
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
    # 仅惩罚 base roll (projected gravity y 分量), 不惩罚斜坡上的 pitch.
    # PLATFORM / SCAN per-rank reward 用到 (weight 0 默认会被 disable_zero_weight_rewards 移除).
    base_roll_l2 = RewTerm(
        func=mdp.base_roll_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
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
            noise=Unoise(n_min=-0.2, n_max=0.2),
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
        # 高程图已彻底移出！交给独立的 noisy_elevation 处理
        height_scan = None 
        base_lin_vel = None
        def __post_init__(self):
            # Teacher Actor 使用完全无噪声的本体感觉，以追求性能上限
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoisyElevationCfg(ObsGroup):
        """专门用于训练 AE 以及提供给所有策略 (Teacher/Student) 使用的带噪声高程组"""
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        # --- 2 sensor 多 pitch 弧 (跟 6+6 SinglePitchArc 几何 byte-equal) ---
        # 每 sensor 一次性输出 6 pitch × 21 azim = 126 ray flat (pitch-major).
        # 后向 sensor 通过 sensor offset rot=180°Z 反向, ScanAE reshape (B, 12, 21) 对齐.
        forward_scan = ObsTerm(
            func=multi_layer_scan,
            params={"sensor_cfg": SceneEntityCfg("forward_lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        backward_scan = ObsTerm(
            func=multi_layer_scan,
            params={"sensor_cfg": SceneEntityCfg("backward_lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class BlindStudentPolicyCfg(ObsGroup):
        """纯粹的 Student 本体感知组 (无真实线速度, 无高程图)"""
        base_lin_vel = None
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
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
        height_scan = None

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
        # 同样去除高程图，Critic 的环境感知将通过 `noisy_elevation` 提供
        height_scan = None
        
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
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EstimatorCfg(ObsGroup):
        history_length = 15  
        flatten_history_dim = True 
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
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
            noise=Unoise(n_min=-0.2, n_max=0.2),
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
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

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
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = 0.1
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
        self.observations.pretraincfg.height_scan = None
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
                "yaw": (-3.14, 3.14),
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
        # 单一 terrain importer (跟 main 一致, 取消之前的 terrain + terrain2 双层结构)
        # 默认 generator = MOE_ROUGH_TERRAINS_CFG (混合训练地形)
        # 双层结构里的 CFG2 是边界死亡区 (全部 3m 深 pit) — 单层结构下不需要
        # per-rank terrain dispatch 在 train_moe.py 里按 rank 覆盖 self.scene.terrain.terrain_generator
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=MOE_ROUGH_TERRAINS_CFG,
            max_init_terrain_level=0,  # 默认从 level 0 起步; 训练脚本可通过 --max_init_terrain_level 覆盖
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
        # 拓宽 friction 范围 [0.4, 1.4] (sim2real, 58aad13 port)
        self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.4, 1.4]
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.4, 1.4]
        self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]
        # self.events.randomize_rigid_body_material.params["static_friction_range"] = [1.0, 1.0]
        # self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [1.0, 1.0]
        # self.events.randomize_rigid_body_material.params["restitution_range"] = [0.7, 0.7]
        
        # =====================================================================
        # 2-sensor 多 pitch 水平扇面 LIDAR (跟 baseline 6+6 SinglePitchArc 几何 byte-equal)
        # 每 sensor 内打包 6 个 pitch × 21 azim = 126 ray, 双 sensor 共 252 ray.
        # 跟独立 12-sensor 版本字节级等价 (验证已通过), 但少 10 个 sensor 对象 → 节省
        # IsaacLab sensor 管理开销, raycast kernel launch 数从 12 减到 2.
        # 几何约定: elevation/azimuth (从水平面起算), 跟"同一俯仰角" semantics 一致.
        # 后向 sensor 整体绕 Z 转 180° → boresight = -X.
        # =====================================================================
        FRONT_LIDAR_POS = (0.32028, 0.0, -0.013)
        REAR_LIDAR_POS = (-0.32028, 0.0, -0.013)
        SCAN_MESHES    = ["/World/ground"]

        # 12 个 pitch (从水平面起算 deg, 负=向下), 加密版 (baseline 是 6 个)
        # linspace(-25, 25, 12) → 步长 ~4.5°, 覆盖 baseline 同范围但密度翻倍
        SCAN_PITCHES = [
            -25.000, -20.455, -15.909, -11.364, -6.818, -2.273,
              2.273,   6.818,  11.364,  15.909,  20.455,  25.000,
        ]
        # azimuth ±45° (90° 前向 FOV, 兼顾侧向感知与近场分辨率)
        # 1m 距离侧向覆盖 ±1m, 1.5m 覆盖 ±1.5m. 步长 4.5° 对 1m+ 距离的 ≥4.5cm 障碍仍 OK.
        # 比 baseline 等 Y 线的有效 ~±25° 范围略宽, 给转向时的侧向障碍感知留余量.
        SCAN_AZIM_RANGE = (-45.0, 45.0)
        SCAN_NUM_AZIM = 21

        SCAN_PATTERN = MultiPitchArcPatternCfg(
            pitch_angles_deg=SCAN_PITCHES,
            num_azimuth=SCAN_NUM_AZIM,
            azimuth_range_deg=SCAN_AZIM_RANGE,
        )

        # 前向 sensor: 无旋转, boresight=+X. 6 pitch × 21 azim = 126 ray (pitch-major flat).
        self.scene.forward_lidar = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=FRONT_LIDAR_POS),
            ray_alignment="base",
            pattern_cfg=SCAN_PATTERN,
            max_distance=2.5,
            debug_vis=False,
            reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )
        self.scene.forward_lidar.update_period = 0.1

        # 后向 sensor: 绕 Z 转 180° → boresight = -X. 同 pattern, 物理位置在车后.
        self.scene.backward_lidar = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(
                pos=REAR_LIDAR_POS,
                rot=(0.0, 0.0, 0.0, 1.0),   # quaternion (w,x,y,z) = 绕 Z 转 180°
            ),
            ray_alignment="base",
            pattern_cfg=SCAN_PATTERN,
            max_distance=2.5,
            debug_vis=False,
            reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )
        self.scene.backward_lidar.update_period = 0.1
        # Rewards
        self.rewards.is_terminated.weight = 0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = -0.5
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
        self.rewards.hipx_joint_pos_penalty.weight = -0.6
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.hipy_joint_pos_penalty.weight = -0.3
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -0.3
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        self.rewards.action_mirror.weight = -0.0
        self.rewards.action_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.undesired_contacts.weight = -0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # track_lin_vel: 精准 (pre_exp) 权重并入粗略, 简化 reward landscape.
        # y 关闭走的是命令侧: lin_vel_y 全局 = (0, 0), 自动让 track_lin_vel_xy_exp
        # 退化成"x 跟踪 + 横向漂移惩罚", 不需要改 reward func。
        self.rewards.track_lin_vel_xy_exp.weight = 3.0   # was 2.5 (合并了 pre_exp 的 0.5)
        self.rewards.track_lin_vel_xy_pre_exp.weight = 0.0  # was 0.5, 已合并入 xy_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.5 # 1.2
        self.rewards.track_ang_vel_z_pre_exp.weight = 1.5

        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_long.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
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
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        self.rewards.upward.weight = 0.08

        if self.__class__.__name__ == "DeeproboticsM20MoETeacherEnvCfg":
            self.disable_zero_weight_rewards()
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        # self.terminations.illegal_contact = None
        self.terminations.bad_orientation_2 = None

        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.6, 1.0) 

        # env 默认 lin_vel_x 收紧到 ±1.0 (was ±1.5), 对齐 main, 减少硬地形上"高速失稳".
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # y 命令暂时全局关掉 (lateral 训练没起效, 简化目标先做好 x). 命令为 0 时
        # track_lin_vel_xy_exp 自动惩罚横向漂移 (跟踪误差 = |actual_vy|).
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        
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
        # 禁用所有多层雷达扫描
        self.observations.noisy_elevation.forward_scan_l0 = None
        self.observations.noisy_elevation.forward_scan_l1 = None
        self.observations.noisy_elevation.forward_scan_l2 = None
        self.observations.noisy_elevation.forward_scan_l3 = None
        self.observations.noisy_elevation.forward_scan_l4 = None
        self.observations.noisy_elevation.forward_scan_l5 = None
        self.observations.noisy_elevation.backward_scan_l0 = None
        self.observations.noisy_elevation.backward_scan_l1 = None
        self.observations.noisy_elevation.backward_scan_l2 = None
        self.observations.noisy_elevation.backward_scan_l3 = None
        self.observations.noisy_elevation.backward_scan_l4 = None
        self.observations.noisy_elevation.backward_scan_l5 = None
        if self.__class__.__name__ == "DeeproboticsM20MoETeacherEnvCfg_EleOnly":
            self.disable_zero_weight_rewards()

@configclass
class DeeproboticsM20MoETeacherEnvCfg_ScanOnly(DeeproboticsM20MoETeacherEnvCfg):
    """仅使用 MultiLayer Scan 的环境"""
    def __post_init__(self):
        super().__post_init__()
        # 禁用高程图
        self.observations.noisy_elevation.height_scan = None
        if self.__class__.__name__ == "DeeproboticsM20MoETeacherEnvCfg_ScanOnly":
            self.disable_zero_weight_rewards()