# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg, DeeproboticsM20RewardsCfg
from rl_training.terrains.config.rough import PLACEMENT_TEACHER_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class PlacementRewardsCfg(DeeproboticsM20RewardsCfg):
    """T4 精准落足专家的奖励函数"""
    
    # 放宽平稳性限制
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.03)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    
    # 强化精准踏足
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_curriculum,
        weight=0.8,
        params={
            "command_name": "base_velocity",
            "threshold": 0.2,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel"),
        },
    )
    
    # 严惩不良接触
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_wheel).*"),
            "threshold": 1.0,
        }
    )
    
@configclass
class DeeproboticsM20TeacherPlacementEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    """[Teacher 4] 精准落足专家环境配置 (T4)
    
    运动模态：精准踩踏、避开孔洞
    地形：带孔网格（多种网格宽度）
    难度：中等
    """
    
    def __post_init__(self):
        super().__post_init__()

        # 1. 地形设置
        self.scene.terrain.terrain_generator = PLACEMENT_TEACHER_TERRAINS_CFG

        # 2. 速度指令调整
        if self.commands.base_velocity is not None:
            self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 3. 奖励函数
        self.rewards = PlacementRewardsCfg()
        self.rewards.is_terminated.weight = -100  # 摔/撞惩罚 (统一用 is_terminated, 不再用 termination_penalty)
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.action_rate_l2.weight = -0.01

        # 4. 观测：启用高程估计器
        # policy 和 critic 使用本体感觉 + 高程特征
        # 高程传感器已在 noisy_elevation 中提供

        # 5. 终止条件
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.bad_orientation_2 = None

        # 6. 课程学习：不调整命令难度
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (1.0, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (1.0, 1.0)

        # 7. 随机化
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.0, 0.1),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
            },
        }

        if hasattr(self, "disable_zero_weight_rewards"):
            self.disable_zero_weight_rewards()

        self.episode_length_s = 18.0
