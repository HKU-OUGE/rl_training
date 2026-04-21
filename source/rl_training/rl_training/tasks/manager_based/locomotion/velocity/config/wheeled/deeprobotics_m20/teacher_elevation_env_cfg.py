from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg_EleOnly, DeeproboticsM20RewardsCfg
from rl_training.terrains.config.rough import ELEVATION_TEACHER_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import math
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class ElevationRewardsCfg(DeeproboticsM20RewardsCfg):
    """精准高程老师的专属奖励函数"""
    
    # =========================================================
    # 1. 释放平稳性限制 (专为上下楼梯设计)
    # =========================================================
    # 放宽对 Z 轴速度的惩罚，因为上下楼梯必然会产生较大的 Z 轴速度波动
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05) 
    # 放宽对 Pitch 角的惩罚，允许在斜坡和楼梯上略微仰头或低头
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01) 
    
    # =========================================================
    # 2. 强化跨越行为 (高抬腿与防磕碰)
    # =========================================================
    # 鼓励长距离抬脚：跨越缝隙和上台阶需要比平地跑拥有更长的腾空时间
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_curriculum,
        weight=1.5, # 相比普通粗糙地形提高权重
        params={
            "command_name": "base_velocity",
            "threshold": 0.3, # 腾空时间超过 0.3s 开始给大额奖励
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel"),
        },
    )
    
    # 严惩除了轮子以外的身体碰撞：逼迫机器人学会把腿抬高跨过台阶，而不是拖着底盘和小腿刮过去
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0, # 极重惩罚
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_wheel).*"), 
            "threshold": 1.0,
        }
    )
    
    # 基础生存惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class DeeproboticsM20TeacherElevationEnvCfg(DeeproboticsM20MoETeacherEnvCfg_EleOnly):
    """[Teacher 2] 精准高程专家环境配置"""
    
    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------
        # 1. 挂载专属地形 (楼梯、缝隙、平地等)
        # ---------------------------------------------------------
        self.scene.terrain.terrain_generator = ELEVATION_TEACHER_TERRAINS_CFG

        # ---------------------------------------------------------
        # 2. 速度指令调整 (2.5D 闭环纠偏, 对齐 scan 环境)
        # ---------------------------------------------------------
        if self.commands.base_velocity is not None:
            self.commands.base_velocity.rel_heading_envs = 1.0
            self.commands.base_velocity.heading_command = True
            self.commands.base_velocity.heading_control_stiffness = 0.5
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)
            self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

        # ---------------------------------------------------------
        # 3. 替换奖励并清理
        # ---------------------------------------------------------
        self.rewards = ElevationRewardsCfg()
        self.rewards.is_terminated.weight = 0
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_roll_l2.weight = -10.0
        self.rewards.base_height_l2.weight = -0.0
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

        self.rewards.track_lin_vel_xy_exp.weight = 3.0 # 1.8
        self.rewards.track_ang_vel_z_exp.weight = 2.0 # 1.2
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

        # ---------------------------------------------------------
        # 4. 终止条件对齐
        # ---------------------------------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.bad_orientation_2 = None

        # ---------------------------------------------------------
        # 5. 课程学习：取消命令课程
        # ---------------------------------------------------------
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (1.0, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (1.0, 1.0)

        # ---------------------------------------------------------
        # 6. 随机化对齐
        # ---------------------------------------------------------
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
            
        # 给足通过复杂楼梯和缝隙的单局时间
        self.episode_length_s = 20.0