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
    """提取归一化的地形等级作为特权信息 (0.0 到 1.0).

    动态读 terrain.cfg.num_rows, 适配不同 task 的 num_rows 设定 (e.g. GAP=20, Platform/Scan=30).
    若读不到 num_rows (旧版本或 None), 回退到 30.0.
    """
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        levels = env.scene.terrain.terrain_levels.float().unsqueeze(1)
        num_rows = getattr(getattr(env.scene.terrain, "cfg", None), "num_rows", None) or 30
        max_level = max(int(num_rows) - 1, 1)
        return levels / max_level
    else:
        return torch.zeros((env.num_envs, 1), device=env.device)