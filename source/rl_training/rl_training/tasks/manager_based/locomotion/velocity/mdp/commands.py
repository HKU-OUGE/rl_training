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
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


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
        # 1. 执行基类的基础准备工作 (这会基于 cfg.ranges 采样，受课程学习控制)
        super()._resample_command(env_ids)

        # 2. 获取当前环境的地形等级
        if hasattr(self._env.scene, "terrain") and hasattr(self._env.scene.terrain, "terrain_levels"):
            levels = self._env.scene.terrain.terrain_levels[env_ids]
        else:
            levels = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)

        # 3. 准备索引掩码
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        hard_mask = levels >= self.cfg.terrain_level_threshold
        hard_ids = env_ids_tensor[hard_mask]

        # 4. 针对困难地形 (Hard) 进行特殊采样覆盖 (读取 Config 里的 hard_ranges)
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

        # 5. 重新应用死区逻辑 (保持一致性)
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
        # if 0 in env_ids_tensor:
        #     idx = (env_ids_tensor == 0).nonzero(as_tuple=True)[0][0]
        #     lvl = levels[idx].item()
        #     print(f"👉 [Command Debug] Env 0 | 地形等级: {lvl} | 指令 -> X: {self.vel_command_b[0, 0].item():.2f}, Y: {self.vel_command_b[0, 1].item():.2f}, Yaw: {self.vel_command_b[0, 2].item():.2f}")


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
