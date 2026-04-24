# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

# rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py

def terrain_level_normalized(env: ManagerBasedRLEnv) -> torch.Tensor:
    """提取归一化的地形等级作为特权信息 (-1.0 到 1.0)"""
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        levels = env.scene.terrain.terrain_levels.float().unsqueeze(1)
        normalized_levels = levels / 30.0
        return normalized_levels
    else:
        return torch.zeros((env.num_envs, 1), device=env.device)


def sub_terrain_one_hot(env: ManagerBasedRLEnv, num_types: int) -> torch.Tensor:
    """子地形类型的 one-hot 编码 (Critic 特权信息).

    依赖 TerrainImporter 的 ``terrain_types`` 张量 (每个 env 的列号).
    要求: ``num_cols`` 等于 sub_terrains 的类型数, 使得 列号 <-> 类型 一一对应.
    非 generator 地形 (flat plane) 下返回全零向量.
    """
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_types"):
        terrain_types = env.scene.terrain.terrain_types.long().clamp(min=0, max=num_types - 1)
        return torch.nn.functional.one_hot(terrain_types, num_classes=num_types).float()
    return torch.zeros((env.num_envs, num_types), device=env.device)