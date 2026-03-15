# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import math
import torch # Added torch for depth calculation
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
        # 使用 -1.0 作为独特的盲区标识符 (Out-of-Band Flag)
        normalized_depths = torch.where(
            depths < 0.2, 
            torch.full_like(normalized_depths, -1.0), 
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
        weight=-0.0,
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
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-100.0, 100.0),
            scale=0.25, 
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.0, n_max=0.0),
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
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.0, n_max=0.0),
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

        def __post_init__(self):
            # Teacher Actor 使用完全无噪声的本体感觉，以追求性能上限
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class NoisyElevationCfg(ObsGroup):
        """专门用于训练 AE 以及提供给所有策略 (Teacher/Student) 使用的带噪声高程组"""
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1), # 加入噪声模拟真实传感器
            clip=(-1.0, 1.0),
            scale=1.0,
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
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
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
        # ground terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/obstacles",
            terrain_type="generator",
            terrain_generator=MOE_ROUGH_TERRAINS_CFG2,
            max_init_terrain_level=5,
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
        self.scene.terrain2 = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=MOE_ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
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
        self.scene.terrain2.terrain_generator = MOE_ROUGH_TERRAINS_CFG
        if(self.scene.terrain2.terrain_generator == MOE_ROUGH_TERRAINS_CFG):
            self.scene.terrain2.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.2)
            self.scene.terrain2.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.16)
            self.scene.terrain2.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
            self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.35, 1.5]
            self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.35, 1.5]
            self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]
        elif(self.scene.terrain2.terrain_generator == ROUGH_TERRAINS_CFG):
            self.scene.terrain2.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.2)
            self.scene.terrain2.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.16)
            self.scene.terrain2.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
            self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.35, 1.5]
            self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.35, 1.5]
            self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]
        else:
            self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.35, 1.5]
            self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.35, 1.5]
            self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]
        # ==============================================================
        # 多层 2D 扫描：分成 3 个独立的 1D 传感器，彻底避免网格重叠
        # ==============================================================
        # 基础四元数 w=0.707, y=-0.707 (绕 Y 轴负向转 90 度，使射线对准 +X 正前方)
        FORWARD_ROT = (0.7071068, 0.0, -0.7071068, 0.0) 
        # 视野配置：Z方向 0m(1层)，Y方向 3m (水平铺开 61 条射线)
        SCAN_PATTERN = patterns.GridPatternCfg(resolution=0.05, size=[0.0, 3.0])
        # 扫描目标：地面 + 障碍物 (开启网格实时追踪)
        SCAN_MESHES = [
            "/World/ground", "/World/obstacles",
        ]

        # 第 1 层：底盘下方 (处理矮小障碍/门槛)
        # M20 base_link 初始高度约 0.52m。偏移 -0.2m -> 绝对高度约 0.32m
        self.scene.forward_scanner_layer0 = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.3, 0.0, -0.2), rot=FORWARD_ROT),
            ray_alignment="base", 
            pattern_cfg=SCAN_PATTERN, max_distance=3.0, debug_vis=False, reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )

        # 第 2 层：底盘正前方 (躯干高度)
        # 偏移 0.0m -> 绝对高度约 0.52m
        self.scene.forward_scanner_layer1 = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 0.0), rot=FORWARD_ROT),
            ray_alignment="base", 
            pattern_cfg=SCAN_PATTERN, max_distance=3.0, debug_vis=False, reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )

        # 第 3 层：底盘上方 (探测悬挂物/圆环顶部)
        # 偏移 +0.2m -> 绝对高度约 0.72m
        self.scene.forward_scanner_layer2 = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 0.2), rot=FORWARD_ROT),
            ray_alignment="base", 
            pattern_cfg=SCAN_PATTERN, max_distance=3.0, debug_vis=False, reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )
        self.scene.forward_scanner_layer0.update_period = 0.1  # 每0.1秒更新一次 (10Hz)
        self.scene.forward_scanner_layer1.update_period = 0.1  # 每0.1秒更新一次 (10Hz)
        self.scene.forward_scanner_layer2.update_period = 0.1  # 每0.1秒更新一次 (10Hz)
        # ==============================================================
        # 向后看的多层 2D 扫描 (后雷达)
        # ==============================================================
        # 基础四元数 w=0.707, y=0.707 (绕 Y 轴正向转 90 度，使射线对准 -X 正后方)
        BACKWARD_ROT = (0.7071068, 0.0, 0.7071068, 0.0) 

        # 第 1 层：后方底盘下方
        self.scene.backward_scanner_layer0 = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(-0.3, 0.0, -0.2), rot=BACKWARD_ROT),
            ray_alignment="base", 
            pattern_cfg=SCAN_PATTERN, max_distance=3.0, debug_vis=False, reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )

        # 第 2 层：后方底盘正中
        self.scene.backward_scanner_layer1 = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(-0.3, 0.0, 0.0), rot=BACKWARD_ROT),
            ray_alignment="base", 
            pattern_cfg=SCAN_PATTERN, max_distance=3.0, debug_vis=False, reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )

        # 第 3 层：后方底盘上方
        self.scene.backward_scanner_layer2 = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(-0.3, 0.0, 0.2), rot=BACKWARD_ROT),
            ray_alignment="base", 
            pattern_cfg=SCAN_PATTERN, max_distance=3.0, debug_vis=False, reference_meshes=True,
            mesh_prim_paths=SCAN_MESHES,
        )

        # 设置更新频率为 10Hz
        self.scene.backward_scanner_layer0.update_period = 0.1
        self.scene.backward_scanner_layer1.update_period = 0.1  
        self.scene.backward_scanner_layer2.update_period = 0.1

        # Rewards
        self.rewards.is_terminated.weight = 0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.40
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
        self.rewards.joint_acc_l2.weight = -2e-7
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
        self.rewards.hipx_joint_pos_penalty.weight = -0.4
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.hipy_joint_pos_penalty.weight = -0.1
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -0.1
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.03
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

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 2.5 # 1.8
        self.rewards.track_ang_vel_z_exp.weight = 1.5 # 1.2
        self.rewards.track_lin_vel_xy_pre_exp.weight = 0.5
        self.rewards.track_ang_vel_z_pre_exp.weight = 1.5

        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
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

        self.terminations.illegal_contact = None
        self.terminations.bad_orientation_2 = None

        self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.curriculum.command_levels.params["range_multiplier"] = (1.0, 1.0)