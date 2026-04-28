# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
import math
import isaaclab.utils.math as math_utils

class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator with explicit standing / pure-turn / moving categories.

    Categorization (per resample):
      - **Standing** (rel_standing_envs of all envs): cmd = (0, 0, 0). Set by parent.
      - **Pure-turn** (rel_pure_turn_envs of all envs, parallel to standing):
        cmd = (0, 0, ±[0.5, 1.0]) — in-place rotation training samples.
      - **Moving** (the rest): parent's uniform sample, with a weak filter that
        re-assigns wishy-washy small commands (3D norm < min_cmd_norm) to
        |lin_x|∈[0.5, 1.5] keeping ang_z.

    Naturally-sampled "lin small + ang large" commands automatically pass through
    the moving filter (since their 3D norm is large), giving extra free pure-turn
    samples on top of the explicit allocation.
    """

    cfg: "UniformThresholdVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        min_cmd_norm = getattr(self.cfg, "min_cmd_norm", 0.5)

        # Pure-turn 比例：默认与 rel_standing_envs 一致
        rate_pt = getattr(self.cfg, "rel_pure_turn_envs", None)
        if rate_pt is None:
            rate_pt = getattr(self.cfg, "rel_standing_envs", 0.0)

        # standing envs (parent 已设为 0)，从命令是否为 0 反查
        cur_norm_after_parent = torch.norm(self.vel_command_b[env_ids_tensor], dim=1)
        standing_mask = cur_norm_after_parent < 1e-4

        # 在非 standing envs 中显式抽出 rate_pt 比例做纯转向
        rand = torch.rand(len(env_ids_tensor), device=self.device)
        pure_turn_mask = (~standing_mask) & (rand < rate_pt)
        if pure_turn_mask.any():
            pt_ids = env_ids_tensor[pure_turn_mask]
            n = len(pt_ids)
            ang_dirs = torch.randint(0, 2, (n,), device=self.device) * 2 - 1
            ang_mags = math_utils.sample_uniform(0.5, 1.0, (n,), device=self.device)
            self.vel_command_b[pt_ids, 0] = 0.0
            self.vel_command_b[pt_ids, 1] = 0.0
            self.vel_command_b[pt_ids, 2] = ang_dirs * ang_mags

        # 余下"moving" envs：原 weak filter——total_norm 太小则重采样为直行
        moving_mask = (~standing_mask) & (~pure_turn_mask)
        if moving_mask.any():
            mv_ids = env_ids_tensor[moving_mask]
            mv_norm = torch.norm(self.vel_command_b[mv_ids], dim=1)
            invalid = (mv_norm < min_cmd_norm) & (mv_norm > 1e-4)
            if invalid.any():
                inv_ids = mv_ids[invalid]
                n = len(inv_ids)
                lin_dirs = torch.randint(0, 2, (n,), device=self.device) * 2 - 1
                lin_mags = math_utils.sample_uniform(0.5, 1.5, (n,), device=self.device)
                self.vel_command_b[inv_ids, 0] = lin_dirs * lin_mags
                self.vel_command_b[inv_ids, 1] = 0.0
                # ang_z 保留


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

    min_cmd_norm: float = 0.5
    """Minimum 3D command norm sqrt(lin_x^2 + lin_y^2 + ang_z^2). Below this the
    moving (non-standing, non-pure-turn) command is re-sampled to have |lin_x|>=0.5."""

    rel_pure_turn_envs: float | None = None
    """Fraction of envs assigned pure-turn commands (lin=0, |ang_z|∈[0.5,1.0]) per
    resample. If ``None``, defaults to ``rel_standing_envs`` so pure-turn rate
    matches standing rate exactly. Set to 0.0 to disable explicit pure-turn
    allocation (natural samples can still produce pure-turn-like cmds)."""


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """

class TerrainAwareVelocityCommand(UniformThresholdVelocityCommand):
    """感知地形难度的速度指令生成器：根据地形等级分配不同的速度指令范围"""

    cfg: "TerrainAwareVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        # 1. 执行基类采样
        super()._resample_command(env_ids)

        # 2. 获取地形等级
        if hasattr(self._env.scene, "terrain") and hasattr(self._env.scene.terrain, "terrain_levels"):
            levels = self._env.scene.terrain.terrain_levels[env_ids]
        else:
            levels = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        hard_mask = levels >= self.cfg.terrain_level_threshold
        hard_ids = env_ids_tensor[hard_mask]

        # 3. 针对困难地形进行覆盖采样
        if len(hard_ids) > 0:
            self.vel_command_b[hard_ids, 0] = math_utils.sample_uniform(
                self.cfg.hard_ranges.lin_vel_x[0], self.cfg.hard_ranges.lin_vel_x[1], (len(hard_ids),), device=self.device
            )
            self.vel_command_b[hard_ids, 1] = math_utils.sample_uniform(
                self.cfg.hard_ranges.lin_vel_y[0], self.cfg.hard_ranges.lin_vel_y[1], (len(hard_ids),), device=self.device
            )
            self.vel_command_b[hard_ids, 2] = math_utils.sample_uniform(
                self.cfg.hard_ranges.ang_vel_z[0], self.cfg.hard_ranges.ang_vel_z[1], (len(hard_ids),), device=self.device
            )

        # 4. 标准化处理：standing / pure-turn / moving 三类显式分配
        #    （和 UniformThresholdVelocityCommand 同逻辑）
        min_cmd_norm = getattr(self.cfg, "min_cmd_norm", 0.5)
        rate_pt = getattr(self.cfg, "rel_pure_turn_envs", None)
        if rate_pt is None:
            rate_pt = getattr(self.cfg, "rel_standing_envs", 0.0)

        cur_norm_after_parent = torch.norm(self.vel_command_b[env_ids_tensor], dim=1)
        standing_mask = cur_norm_after_parent < 1e-4

        rand = torch.rand(len(env_ids_tensor), device=self.device)
        pure_turn_mask = (~standing_mask) & (rand < rate_pt)
        if pure_turn_mask.any():
            pt_ids = env_ids_tensor[pure_turn_mask]
            n = len(pt_ids)
            ang_dirs = torch.randint(0, 2, (n,), device=self.device) * 2 - 1
            ang_mags = math_utils.sample_uniform(0.5, 1.0, (n,), device=self.device)
            self.vel_command_b[pt_ids, 0] = 0.0
            self.vel_command_b[pt_ids, 1] = 0.0
            self.vel_command_b[pt_ids, 2] = ang_dirs * ang_mags

        moving_mask = (~standing_mask) & (~pure_turn_mask)
        if moving_mask.any():
            mv_ids = env_ids_tensor[moving_mask]
            mv_norm = torch.norm(self.vel_command_b[mv_ids], dim=1)
            invalid = (mv_norm < min_cmd_norm) & (mv_norm > 1e-4)
            if invalid.any():
                inv_ids = mv_ids[invalid]
                n = len(inv_ids)
                lin_dirs = torch.randint(0, 2, (n,), device=self.device) * 2 - 1
                lin_mags = math_utils.sample_uniform(0.5, 1.5, (n,), device=self.device)
                self.vel_command_b[inv_ids, 0] = lin_dirs * lin_mags
                self.vel_command_b[inv_ids, 1] = 0.0


@configclass
class TerrainAwareVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """感知地形难度的配置类"""
    class_type: type = TerrainAwareVelocityCommand
    
    terrain_level_threshold: int = 10
    
    # 简单地形的指令范围 (< threshold)
    easy_ranges: UniformThresholdVelocityCommandCfg.Ranges = UniformThresholdVelocityCommandCfg.Ranges(
        lin_vel_x=(-2.0, 2.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-1.5, 1.5), heading=(-math.pi, math.pi)
    )
    # 困难地形的指令范围 (>= threshold)
    hard_ranges: UniformThresholdVelocityCommandCfg.Ranges = UniformThresholdVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
    )
