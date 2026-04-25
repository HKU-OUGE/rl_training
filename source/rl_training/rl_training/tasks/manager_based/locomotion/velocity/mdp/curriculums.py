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


def _meaningful_env_ids(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    axis: str,
    cur_max: float,
    filter_ratio: float,
    filter_floor: float,
) -> torch.Tensor | None:
    """根据当前命令幅度过滤 env_ids, 排除 trivial 命令的环境.

    背景:
      pure_turn 桶 (vx=vy=0) 在 lin track reward 上 trivially 满分;
      standing envs 在两个 reward 上都 trivially 满分.
      这些 envs 进入 trigger 平均时, 把 reward baseline 拉高 ~0.15-0.20,
      导致 curriculum 容易"假阳性"升级. 过滤掉它们后 curriculum 只看
      "真实在跟踪有意义命令" 的 envs.

    Returns:
        过滤后的 env_ids tensor (subset). 如果有意义样本 < 4 (太少不可靠), 返回 None.
    """
    cmd = env.command_manager.get_command("base_velocity")
    env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if axis == "lin":
        cmd_norm = torch.norm(cmd[env_ids_t, :2], dim=1)
    else:  # "ang"
        cmd_norm = cmd[env_ids_t, 2].abs()
    threshold = max(filter_ratio * cur_max, filter_floor)
    meaningful = cmd_norm > threshold
    n_meaningful = int(meaningful.sum().item())
    if n_meaningful < 4:
        return None
    return env_ids_t[meaningful]


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    cmd_filter_ratio: float = 0.3,
    cmd_filter_floor: float = 0.1,
) -> torch.Tensor:
    """独立控制 XY 线速度的课程学习 (鲁棒版 + cmd-magnitude 过滤).

    每隔 max_episode_length 步评估一次:
      - 只统计**当前命令 |cmd_xy| > threshold** 的 envs (排除 pure_turn 桶和 standing
        envs, 这些 envs 的 lin tracking 是 trivial 满分, 会污染 trigger 信号)
      - threshold = max(cmd_filter_ratio * lin_max_cur, cmd_filter_floor)
      - 如果有意义样本 < 4, 跳过这次评估 (信号太弱)
      - 在剩余 envs 上算平均 track reward; > 0.75 × weight 才推进
    """
    term_cfg = env.command_manager.get_term("base_velocity").cfg

    if not hasattr(env, "_cmd_curr_initialized_lin"):
        env._cmd_curr_initialized_lin = True
        env._original_vel_x = torch.tensor(term_cfg.ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(term_cfg.ranges.lin_vel_y, device=env.device)
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]
        term_cfg.ranges.lin_vel_x = (env._original_vel_x * range_multiplier[0]).tolist()
        term_cfg.ranges.lin_vel_y = (env._original_vel_y * range_multiplier[0]).tolist()
        env._last_lin_update_step = 0

    if env.common_step_counter - env._last_lin_update_step >= env.max_episode_length:
        env._last_lin_update_step = env.common_step_counter

        # NEW: 用当前 lin_max 算 threshold, 过滤掉 trivial 命令的 envs
        lin_max_cur = max(abs(term_cfg.ranges.lin_vel_x[0]), abs(term_cfg.ranges.lin_vel_x[1]))
        filtered = _meaningful_env_ids(env, env_ids, "lin", lin_max_cur, cmd_filter_ratio, cmd_filter_floor)
        if filtered is None:
            return torch.tensor(term_cfg.ranges.lin_vel_x[1], device=env.device)

        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_weight = env.reward_manager.get_term_cfg(reward_term_name).weight
        mean_reward_per_step = torch.mean(episode_sums[filtered]) / env.max_episode_length_s

        if mean_reward_per_step > 0.75 * reward_weight:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            new_vel_x = torch.tensor(term_cfg.ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(term_cfg.ranges.lin_vel_y, device=env.device) + delta_command
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])
            term_cfg.ranges.lin_vel_x = new_vel_x.tolist()
            term_cfg.ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(term_cfg.ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    cmd_filter_ratio: float = 0.3,
    cmd_filter_floor: float = 0.1,
) -> torch.Tensor:
    """独立控制 Z 轴角速度的课程学习 (鲁棒版 + cmd-magnitude 过滤).

    与 lin 版本对称: 只统计 |cmd_z| > threshold 的 envs (排除 standing envs).
    pure_turn 桶在这里恰好被保留 (它们 |cmd_z|>=0.5 满足 threshold).
    """
    term_cfg = env.command_manager.get_term("base_velocity").cfg

    if not hasattr(env, "_cmd_curr_initialized_ang"):
        env._cmd_curr_initialized_ang = True
        env._original_ang_vel_z = torch.tensor(term_cfg.ranges.ang_vel_z, device=env.device)
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]
        term_cfg.ranges.ang_vel_z = (env._original_ang_vel_z * range_multiplier[0]).tolist()
        env._last_ang_update_step = 0

    if env.common_step_counter - env._last_ang_update_step >= env.max_episode_length:
        env._last_ang_update_step = env.common_step_counter

        ang_max_cur = max(abs(term_cfg.ranges.ang_vel_z[0]), abs(term_cfg.ranges.ang_vel_z[1]))
        filtered = _meaningful_env_ids(env, env_ids, "ang", ang_max_cur, cmd_filter_ratio, cmd_filter_floor)
        if filtered is None:
            return torch.tensor(term_cfg.ranges.ang_vel_z[1], device=env.device)

        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_weight = env.reward_manager.get_term_cfg(reward_term_name).weight
        mean_reward_per_step = torch.mean(episode_sums[filtered]) / env.max_episode_length_s

        if mean_reward_per_step > 0.75 * reward_weight:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            new_ang_vel_z = torch.tensor(term_cfg.ranges.ang_vel_z, device=env.device) + delta_command
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])
            term_cfg.ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(term_cfg.ranges.ang_vel_z[1], device=env.device)