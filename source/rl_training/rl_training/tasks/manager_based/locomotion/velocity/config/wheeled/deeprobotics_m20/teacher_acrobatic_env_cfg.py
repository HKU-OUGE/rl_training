import math
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
# 【修正导入】使用 IsaacLab 原生的均匀速度指令
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from rl_training.terrains.config.rough import (
    ACROBATIC_TEACHER_TERRAINS_CFG,
    BASE_TEACHER_TERRAINS_CFG,
    ELEVATION_TEACHER_TERRAINS_CFG,
    SCAN_TEACHER_TERRAINS_CFG,
    PLACEMENT_TEACHER_TERRAINS_CFG,
)


@configclass
class AcrobaticRewardsCfg(RewardsCfg):
    """专门为后空翻/直立设计的奖励函数"""
    
    # =========================================================
    # 1. 彻底禁用移动专家的常规奖励
    # =========================================================
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    feet_air_time = None
    undesired_contacts = None
    flat_orientation_l2 = None
    feet_gait = None
    
    # =========================================================
    # 2. [核心修复] 彻底禁用所有“平稳性惩罚”
    # (后空翻需要极大的加速度和角速度，绝不能惩罚这些项！)
    # =========================================================
    body_lin_acc_l2 = None     # 修复报错：禁用躯干线性加速度惩罚
    body_ang_acc_l2 = None     # 禁用躯干角加速度惩罚
    lin_vel_z_l2 = None        # 禁用 Z 轴速度惩罚 (起跳需要极大的 Z 轴速度)
    ang_vel_xy_l2 = None       # 禁用 XY 角速度惩罚 (后空翻需要极大的 Pitch 角速度)
    dof_acc_l2 = None          # 禁用关节加速度惩罚 (允许爆发发力)
    wheel_vel_penalty = None   # 修复报错：允许轮子作为动量飞轮高速旋转
    joint_torques_l2 = None    # 建议也把力矩惩罚关掉，允许爆发最大扭矩起跳
    dof_vel_l2 = None          # 如果有的话，禁用普通关节速度惩罚
    dof_pos_limits = None      # 禁用关节限位惩罚（空翻可能需要达到极限位）
# [NEW] 添加禁用接触力惩罚
    contact_forces = None      # 修复报错：禁用脚部等传感器接触力惩罚
    feet_stumble = None        # 如果有的话，顺便禁用绊脚惩罚
    # =========================================================
    # 3. 杂技动作阶段分发器
    # =========================================================
    acrobatic_expert_reward = RewTerm(
        func=mdp.acrobatic_router_reward,
        weight=1.0,
        params={
            "command_name": "acrobatic_cmd",
            "skill_weights": {
                "backflip": 2.0,
                "sideflip": 2.0,
                "sideroll": 1.5,
                "handstand": 2.0
            }
        }
    )
    
    # =========================================================
    # 4. 基础生存与平滑度
    # =========================================================
    # 动作平滑度惩罚 (避免高频抽搐，但不限制低频爆发)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # 基础生存惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)


from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp  # Ensure this is imported

@configclass
class DeeproboticsM20TeacherAcrobaticEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    """[Teacher 5] 杂技专家训练环境 (盲视, 平地, 特殊指令)"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 1. 强制使用全平地地形，关闭地形课程
        self.scene.terrain.terrain_generator = ACROBATIC_TEACHER_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = False
        
        # 2. 彻底禁用视觉观测 (盲视训练防干扰)
        self.observations.policy.noisy_elevation = None
        self.observations.policy.scan = None
        
        # 3. 替换指令生成器
        self.commands.base_velocity = None 
        self.commands.acrobatic_cmd = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0), # 固定 4 秒做一次动作
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 4.999), 
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            )
        )

    # ---------------------------------------------------------
        # [NEW] 4. 修复 Observation 依赖 (全局彻底替换)
        # ---------------------------------------------------------
        # 遍历所有的观测组（如 policy, critic, estimator 等）
        for group_name in dir(self.observations):
            if group_name.startswith("__"): continue
            group = getattr(self.observations, group_name)
            
            # 如果该组内部存在 velocity_commands，则将其禁用并替换
            if hasattr(group, "velocity_commands") and getattr(group, "velocity_commands") is not None:
                # 移除旧的 base_velocity 观测
                group.velocity_commands = None
                # 添加新的 acrobatic_cmd 观测
                group.acrobatic_commands = ObsTerm(
                    func=mdp.generated_commands, params={"command_name": "acrobatic_cmd"}
                )
        # ---------------------------------------------------------
        
        # 5. 替换奖励函数
        self.rewards = AcrobaticRewardsCfg()
        if hasattr(self, "disable_zero_weight_rewards"):
            self.disable_zero_weight_rewards()
        # 6. 放宽 Termination 条件
        self.terminations.illegal_contact = None 
        self.terminations.bad_orientation_2 = None 
        
        # 7. 提高控制频率
        self.decimation = 2 
        self.episode_length_s = 4.0

# ---------------------------------------------------------
        # [NEW] 8. 彻底禁用 Curriculum (课程管理器) 依赖
        # (杂技专家在固定平地训练，不需要根据移动速度进行地形和指令升级)
        # ---------------------------------------------------------
        for attr in dir(self.curriculum):
            if not attr.startswith("__"):
                setattr(self.curriculum, attr, None)