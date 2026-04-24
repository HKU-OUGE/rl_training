from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg, DeeproboticsM20RewardsCfg
from rl_training.terrains.config.rough import PLATFORM_TEACHER_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import math
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class PlatformRewardsCfg(DeeproboticsM20RewardsCfg):
    """T6 高台攀爬专家的奖励函数"""

    # 放宽 Z 轴速度惩罚：攀爬高台必然产生较大的垂直速度
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.03)
    # 放宽 Roll/Pitch 惩罚：上下高台时允许身体倾斜
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)

    # 鼓励长腾空时间：攀爬高台需要大幅抬腿
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_curriculum,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "threshold": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel"),
        },
    )

    # 允许小腿 (knee) 和大腿 (hipy) 支撑台阶边缘进行攀爬
    # 仅惩罚 base_link (底盘撞台=摔倒) 和 hipx (内侧关节卡住=机构损坏)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_hipx", ".*_hipy"]),
            "threshold": 1.0,
        }
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class DeeproboticsM20TeacherPlatformEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    """[Teacher 6] 高台攀爬专家环境配置

    运动模态：大落差攀爬和降落
    地形：Pit (上高台) + Box (下高台)
    """

    def __post_init__(self):
        super().__post_init__()

        # sub_terrain one-hot 的 num_types 和列顺序只在 MOE_TEACHER_TERRAINS_CFG 下有语义,
        # 此子 teacher 的 terrain 不同, 关掉以免喂给 Critic 错位的 one-hot.
        self.observations.critic.sub_terrain_id = None

        # 1. 地形
        self.scene.terrain.terrain_generator = PLATFORM_TEACHER_TERRAINS_CFG

        # 2. 速度指令 (2.5D 闭环纠偏)
        if self.commands.base_velocity is not None:
            self.commands.base_velocity.rel_heading_envs = 1.0
            self.commands.base_velocity.heading_command = True
            self.commands.base_velocity.heading_control_stiffness = 0.5
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)
            self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

        # 3. 奖励
        self.rewards = PlatformRewardsCfg()
        self.rewards.is_terminated.weight = 0
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_roll_l2.weight = -10.0
        self.rewards.base_height_l2.weight = 0
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
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_pre_exp.weight = 0
        self.rewards.track_ang_vel_z_pre_exp.weight = 0

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
        self.rewards.joint_mirror_lr.weight = -0.015

        # 4. 终止条件
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.bad_orientation_2 = None

        # 5. 课程学习
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (1.0, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (1.0, 1.0)

        # 6. 随机化 (yaw 固定为 0)
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
        self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.6, 1.2]
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.6, 1.2]
        self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]

        if hasattr(self, "disable_zero_weight_rewards"):
            self.disable_zero_weight_rewards()

        self.episode_length_s = 20.0
