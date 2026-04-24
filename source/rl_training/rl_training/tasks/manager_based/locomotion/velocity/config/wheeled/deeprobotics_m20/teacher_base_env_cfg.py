# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg, DeeproboticsM20RewardsCfg
from rl_training.terrains.config.rough import BASE_TEACHER_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class BaseRewardsCfg(DeeproboticsM20RewardsCfg):
    """T1 盲视基础专家的奖励函数"""
    
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.02)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class DeeproboticsM20TeacherBaseEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    """[Teacher 1] 盲视基础专家环境配置 (T1)
    
    运动模态：本体感觉适应，小动作
    地形：平地、粗糙面、缓坡
    难度：最低
    """
    
    def __post_init__(self):
        super().__post_init__()

        # sub_terrain one-hot 的 num_types 和列顺序只在 MOE_TEACHER_TERRAINS_CFG 下有语义,
        # 此子 teacher 的 terrain 不同, 关掉以免喂给 Critic 错位的 one-hot.
        self.observations.critic.sub_terrain_id = None

        # 1. 地形设置
        self.scene.terrain.terrain_generator = BASE_TEACHER_TERRAINS_CFG

        # 2. 速度指令调整
        if self.commands.base_velocity is not None:
            self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 3. 奖励函数
        self.rewards = BaseRewardsCfg()
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.action_rate_l2.weight = -0.01
        
        # 4. 关闭高级感知（盲视）
        self.observations.policy.noisy_elevation = None
        self.observations.policy.scan = None

        # 5. 终止条件
        self.terminations.bad_orientation_2 = None

        # 6. 课程学习
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (1.0, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (1.0, 1.0)

        # 7. 随机化
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.3, 0.3),
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

        self.episode_length_s = 15.0
