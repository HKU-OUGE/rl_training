from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg, DeeproboticsM20RewardsCfg, DeeproboticsM20MoETeacherEnvCfg_ScanOnly
from rl_training.terrains.config.rough import SCAN_TEACHER_TERRAINS_CFG, SCAN_TEACHER_TERRAINS_CFG2
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
##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_CFG  # isort: skip
from rl_training.terrains.config.rough import *
from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import TerrainAwareVelocityCommandCfg
from isaaclab.sensors import ContactSensorCfg
@configclass
class ScanRewardsCfg(DeeproboticsM20RewardsCfg):
    test = None


@configclass
class DeeproboticsM20TeacherScanEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    """[Teacher 3] 空间扫描专家环境配置"""
    
    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------
        # 1. 双地形叠加配置
        # ---------------------------------------------------------
        # 主地形 (Pit, Rings 等)
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=SCAN_TEACHER_TERRAINS_CFG,
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

        FRONT_LIDAR_POS = (0.32028, 0.0, -0.013)
        REAR_LIDAR_POS = (-0.32028, 0.0, -0.013)


        # L5 (last entry) 改为 50° 当贴脚悬崖探测器：sensor z≈0.6m → ground @ 0.5m 水平 / 0.78m 量程
        down_angles_deg = [-25.0, -15.0, -5.0, 5.0, 15.0, 50.0]

        SCAN_PATTERN = patterns.GridPatternCfg(resolution=0.05, size=[0.0, 1.0])
        SCAN_MESHES = ["/World/ground"]  # 扫描地形1和地形2，捕捉跨栏和坑洞信息

        for i, angle_deg in enumerate(down_angles_deg):

            # 前向雷达
            fwd_pitch_deg = -(90.0 - angle_deg)
            fwd_half_rad = math.radians(fwd_pitch_deg) / 2.0
            fwd_rot = (math.cos(fwd_half_rad), 0.0, math.sin(fwd_half_rad), 0.0)

            fwd_sensor = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base_link",
                offset=RayCasterCfg.OffsetCfg(pos=FRONT_LIDAR_POS, rot=fwd_rot),
                ray_alignment="base",
                pattern_cfg=SCAN_PATTERN,
                max_distance=2.5,
                debug_vis=False,
                mesh_prim_paths=SCAN_MESHES,
            )
            fwd_sensor.update_period = 0.1
            setattr(self.scene, f"forward_scanner_layer{i}", fwd_sensor)

            # 后向雷达
            bwd_pitch_deg = (90.0 - angle_deg)
            bwd_half_rad = math.radians(bwd_pitch_deg) / 2.0
            bwd_rot = (math.cos(bwd_half_rad), 0.0, math.sin(bwd_half_rad), 0.0)

            bwd_sensor = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base_link",
                offset=RayCasterCfg.OffsetCfg(pos=REAR_LIDAR_POS, rot=bwd_rot),
                ray_alignment="base",
                pattern_cfg=SCAN_PATTERN,
                max_distance=2.5,
                debug_vis=False,
                mesh_prim_paths=SCAN_MESHES,
            )
            bwd_sensor.update_period = 0.1
            setattr(self.scene, f"backward_scanner_layer{i}", bwd_sensor)
        # ---------------------------------------------------------
        # 2. [新增] 惩罚地形专属传感器
        # ---------------------------------------------------------
        self.scene.obstacle_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",             # <--- 改为大写 R
            filter_prim_paths_expr=["/World/ground/.*"],  
            update_period=0.02,
            debug_vis=True,
            force_threshold=2.0,  # 只要接触力大于 1N 就触发惩罚
        )

        self.commands.base_velocity = mdp.UniformThresholdVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.05,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        )

        # ---------------------------------------------------------
        # 3. 强制 2.5D 运动模式 (基于 Heading 的闭环纠偏)
        # ---------------------------------------------------------
        if self.commands.base_velocity is not None:
            # 1. 开启 100% 的 Heading 控制
            self.commands.base_velocity.rel_heading_envs = 1.0 
            self.commands.base_velocity.heading_command = True
            
            # P控制器的刚度，0.5是一个很好的默认值。
            # 如果发现机器人纠偏太慢，可以稍微调大到 1.0
            self.commands.base_velocity.heading_control_stiffness = 0.5 

            # 2. 锁定目标参数
            # 目标航向角永远是 0 (面朝正前方跑道)
            self.commands.base_velocity.ranges.heading = (0.0, 0.0) 
            # 依然不允许侧向平移
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            # 只有这里有范围，Heading控制器算出的纠偏角速度才能下发给机器人
            self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

            # 正常的前向速度训练范围
            self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)


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
                "z": (-0.0, 0.2),
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

        # Rewards
        self.rewards.is_terminated.weight = -100
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_roll_l2.weight = -10.0
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
        self.rewards.knee_joint_pos_penalty.weight = -0.1
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.025
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        self.rewards.joint_mirror_lr.weight = -0.015
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

        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_pre_exp.weight = 0
        self.rewards.track_ang_vel_z_pre_exp.weight = 0

        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["threshold"] = 0.25
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
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        self.rewards.upward.weight = 0.08
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp_curriculum
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_exp_curriculum
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        # self.terminations.illegal_contact = None
        self.terminations.bad_orientation_2 = None

        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (1.0, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (1.0, 1.0)

        # ------------------------------Commands------------------------------

        if hasattr(self, "disable_zero_weight_rewards"):
            self.disable_zero_weight_rewards()