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
from isaaclab.sensors import RayCasterCfg, patterns
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
##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_CFG  # isort: skip
from rl_training.terrains.config.rough import *

# ==============================================================================
# Helper Functions
# ==============================================================================

def lidar_depth_scan(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculates the Euclidean distance (depth) from the sensor to the ray hit points.
    Returns a tensor of shape (num_envs, num_rays).
    """
    # Access the sensor object
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # sensor.data.pos_w is the sensor position in world frame: [num_envs, 3]
    # sensor.data.ray_hits_w is the hit position in world frame: [num_envs, num_rays, 3]
    
    # Compute the vector from sensor origin to hit point
    rel_vec = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    
    # Compute Euclidean distance (L2 norm) along the last dimension (x,y,z)
    depths = torch.norm(rel_vec, dim=-1)
    
    return depths

def euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """
    Converts Euler angles (in radians) to a quaternion (w, x, y, z).
    Rotation order: X -> Y -> Z (intrinsic).
    """
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

# ==============================================================================
# New: Custom Scene Configuration with Dual LiDARS
# ==============================================================================
@configclass
class DeeproboticsM20SceneCfg(MySceneCfg):
    """Scene configuration with M20 Robot and Dual LiDARS."""
    
    # 1. Front Lidar (Data from config.yaml: x=0.32028, y=0, z=-0.013, pitch=-1.57079, yaw=-3.14159)
    # frequency: 10Hz -> update_period = 0.1s
    lidar_front = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.32028, 0.0, -0.013),
            # Convert Euler (Roll=0, Pitch=-1.57079, Yaw=-3.14159) to Quaternion
            rot=euler_xyz_to_quat(0.0, -1.57079, -3.14159)
        ),
        update_period=0.1, # 10Hz
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=4, # CHANGED: 32 -> 4 (Sparse sampling for MLP input)
            vertical_fov_range=(-15.0, 15.0), 
            horizontal_fov_range=(0.0, 360.0), 
            horizontal_res=10.0, # CHANGED: 2.0 -> 10.0 (36 points horizontal * 4 channels = 144 points)
            # Total Size: 144 floats
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # 2. Rear Lidar (Data from config.yaml: x=-0.32028, y=0, z=-0.013, pitch=-1.57079, yaw=0)
    lidar_rear = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(-0.32028, 0.0, -0.013),
            # Convert Euler (Roll=0, Pitch=-1.57079, Yaw=0) to Quaternion
            rot=euler_xyz_to_quat(0.0, -1.57079, 0.0)
        ),
        update_period=0.1, # 10Hz
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=4, # CHANGED: 32 -> 4
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=10.0, # CHANGED: 2.0 -> 10.0
            # Total Size: 144 floats
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


# ==============================================================================
# Custom Observation Config
# ==============================================================================
@configclass
class DeeproboticsM20ObservationsCfg:
    """Observation specifications for the M20 environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Teacher Policy: Has Privileged Height Scan + Proprioception + Both Lidars."""
        
        # --- Proprioception ---
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
        
        # --- Privileged Information (Global Height Map) ---
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # --- New Lidar Sensors (Front & Rear) ---
        lidar_front_scan = ObsTerm(
            func=lidar_depth_scan, # CHANGED: Use depth/range instead of height
            params={"sensor_cfg": SceneEntityCfg("lidar_front")},
            noise=Unoise(n_min=-0.05, n_max=0.05), # Depth noise
            clip=(0.0, 60.0), # Depth is positive. Max range is 60m per config.yaml
            scale=1.0,
        )
        lidar_rear_scan = ObsTerm(
            func=lidar_depth_scan, # CHANGED: Use depth/range instead of height
            params={"sensor_cfg": SceneEntityCfg("lidar_rear")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 60.0),
            scale=1.0,
        )

    @configclass
    class BlindStudentPolicyCfg(ObsGroup):
        """Student Policy: Only Lidar (Front/Rear) + Proprioception. NO Global Height Map."""
        
        # Removed: base_lin_vel (Privileged)
        base_lin_vel = None
        
        # --- Proprioception ---
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
        
        # --- Removed Global Height Scan ---
        height_scan = None
        
        # --- Added Lidar Sensors (Only inputs for terrain perception) ---
        lidar_front_scan = ObsTerm(
            func=lidar_depth_scan, # CHANGED: Use depth/range
            params={"sensor_cfg": SceneEntityCfg("lidar_front")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 60.0),
            scale=1.0,
        )
        lidar_rear_scan = ObsTerm(
            func=lidar_depth_scan, # CHANGED: Use depth/range
            params={"sensor_cfg": SceneEntityCfg("lidar_rear")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 60.0),
            scale=1.0,
        )

    @configclass
    class StudentPolicyCfg(BlindStudentPolicyCfg):
        """Student Policy alias."""
        pass

    @configclass
    class CriticCfg(PolicyCfg):
        """Critic gets everything (Teacher inputs) but without noise."""
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
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            scale=1.0,
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
            scale=1.0,
        )
        
        # Critic also sees clean Lidar data (Depth)
        lidar_front_scan = ObsTerm(
            func=lidar_depth_scan, # CHANGED
            params={"sensor_cfg": SceneEntityCfg("lidar_front")},
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(0.0, 60.0),
            scale=1.0,
        )
        lidar_rear_scan = ObsTerm(
            func=lidar_depth_scan, # CHANGED
            params={"sensor_cfg": SceneEntityCfg("lidar_rear")},
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(0.0, 60.0),
            scale=1.0,
        )

    @configclass
    class EstimatorCfg(ObsGroup):
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

    policy: PolicyCfg = PolicyCfg()
    blind_student_policy: BlindStudentPolicyCfg = BlindStudentPolicyCfg() # Student now uses Lidar
    student_policy: StudentPolicyCfg = StudentPolicyCfg() 
    critic: CriticCfg = CriticCfg()
    estimator: EstimatorCfg = EstimatorCfg()


@configclass
class DeeproboticsM20MoETeacherEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: DeeproboticsM20ActionsCfg = DeeproboticsM20ActionsCfg()
    rewards: DeeproboticsM20RewardsCfg = DeeproboticsM20RewardsCfg()
    # 使用自定义的观测配置
    observations: DeeproboticsM20ObservationsCfg = DeeproboticsM20ObservationsCfg()
    # Use custom scene with 2 Lidars
    scene: DeeproboticsM20SceneCfg = DeeproboticsM20SceneCfg(num_envs=2048, env_spacing=2.5) # Reduced Envs to save memory

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

        # ------------------------------Sence------------------------------
        self.scene.robot = DEEPROBOTICS_M20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        
        # Ensure new lidars are attached to base
        if self.scene.lidar_front is not None:
             self.scene.lidar_front.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        if self.scene.lidar_rear is not None:
             self.scene.lidar_rear.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        # Apply special logic to Policy (Teacher)
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        
        # Apply same logic to Student Policy (Crucial for correct joint mapping)
        self.observations.blind_student_policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.blind_student_policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        
        # Apply to StudentPolicy as well
        self.observations.student_policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.student_policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )

        # Critic logic
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        
        # Scales
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        
        # Scales for Blind Student
        self.observations.blind_student_policy.base_ang_vel.scale = 0.25
        self.observations.blind_student_policy.joint_pos.scale = 1.0
        self.observations.blind_student_policy.joint_vel.scale = 0.05
        
        # Set Joint Names pattern
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        
        self.observations.blind_student_policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.blind_student_policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        
        self.observations.student_policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.student_policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names


        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hipx_joint": 0.25, "^(?!.*_hipx_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
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
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=MOE_ROUGH_TERRAINS_CFG,
            max_init_terrain_level=0,
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
        self.scene.terrain.terrain_generator = MOE_ROUGH_TERRAINS_CFG
        if(self.scene.terrain.terrain_generator == MOE_ROUGH_TERRAINS_CFG):
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.2)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.10)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
            self.scene.terrain.terrain_generator.sub_terrains["rail"].rail_height_range = (0.05, 0.5)
            self.scene.terrain.terrain_generator.sub_terrains["rail"].rail_thickness_range = (0.05, 0.1)
            self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.35, 1.5]
            self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.35, 1.5]
            self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]
            pass
        elif(self.scene.terrain.terrain_generator == MOE_ROUGH_TEST_TERRAINS_CFG):
            pass

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = -0.1
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.0e-5
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
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.hipx_joint_pos_penalty.weight = -1.0
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.hipy_joint_pos_penalty.weight = -0.75
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -0.25
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        self.rewards.action_mirror.weight = -0.05
        self.rewards.action_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.5 # 1.8
        self.rewards.track_ang_vel_z_exp.weight = 1.5 # 1.2
        self.rewards.track_lin_vel_xy_pre_exp.weight = 2.0
        self.rewards.track_ang_vel_z_pre_exp.weight = 3.0
        # Others
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
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
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        self.rewards.upward.weight = 0.08

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DeeproboticsM20MoETeacherEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation_2 = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)