# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """独立控制 XY 线速度的课程学习 (鲁棒版)"""
    term_cfg = env.command_manager.get_term("base_velocity").cfg
    
    # 1. 鲁棒的初始化 (不再依赖 common_step_counter == 0)
    if not hasattr(env, "_cmd_curr_initialized_lin"):
        env._cmd_curr_initialized_lin = True
        env._original_vel_x = torch.tensor(term_cfg.ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(term_cfg.ranges.lin_vel_y, device=env.device)
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]
        
        # 应用初始难度
        term_cfg.ranges.lin_vel_x = (env._original_vel_x * range_multiplier[0]).tolist()
        term_cfg.ranges.lin_vel_y = (env._original_vel_y * range_multiplier[0]).tolist()
        
        env._last_lin_update_step = 0

    # 2. 基于时间差的更新机制 (绝对不会被跳过)
    if env.common_step_counter - env._last_lin_update_step >= env.max_episode_length:
        env._last_lin_update_step = env.common_step_counter
        
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_weight = env.reward_manager.get_term_cfg(reward_term_name).weight
        
        # 计算当前重置环境的平均表现
        mean_reward_per_step = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s
        
        # 如果跟踪奖励超过满分的 75% (稍微放宽一点，让课程推进更顺滑)
        if mean_reward_per_step > 0.75 * reward_weight:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            
            new_vel_x = torch.tensor(term_cfg.ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(term_cfg.ranges.lin_vel_y, device=env.device) + delta_command
            
            # 限制在最终范围内
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])
            
            # 写入 Config
            term_cfg.ranges.lin_vel_x = new_vel_x.tolist()
            term_cfg.ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(term_cfg.ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """独立控制 Z 轴角速度的课程学习 (鲁棒版)"""
    term_cfg = env.command_manager.get_term("base_velocity").cfg
    
    if not hasattr(env, "_cmd_curr_initialized_ang"):
        env._cmd_curr_initialized_ang = True
        env._original_ang_vel_z = torch.tensor(term_cfg.ranges.ang_vel_z, device=env.device)
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]
        
        term_cfg.ranges.ang_vel_z = (env._original_ang_vel_z * range_multiplier[0]).tolist()
        env._last_ang_update_step = 0

    if env.common_step_counter - env._last_ang_update_step >= env.max_episode_length:
        env._last_ang_update_step = env.common_step_counter
        
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_weight = env.reward_manager.get_term_cfg(reward_term_name).weight
        
        mean_reward_per_step = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s
        
        if mean_reward_per_step > 0.75 * reward_weight:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            
            new_ang_vel_z = torch.tensor(term_cfg.ranges.ang_vel_z, device=env.device) + delta_command
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])
            
            term_cfg.ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(term_cfg.ranges.ang_vel_z[1], device=env.device)