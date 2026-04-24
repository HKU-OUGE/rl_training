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
    """Command generator with threshold and explicit pure-turn bucket.

    在均匀采样的基础上:
      1. 按 `pure_turn_probability` 划出一个 "纯转向" 桶，这些环境 vx=vy=0、|wz|>=0.5,
         并且显式关闭 heading_command / standing，使 wz 能走完整个 `_update_command` 流程.
      2. 对剩余环境沿用原 <0.5 截断，但只在 lin_vel 和 ang_vel_z 同时较小时才视为 "无效",
         否则保留 (小 lin, 大 ang) / (大 lin, 小 ang) 的正常组合.
    """

    cfg: "UniformThresholdVelocityCommandCfg"
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):

        super()._resample_command(env_ids)

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = env_ids_tensor.numel()
        if n == 0:
            return

        ang_low, ang_high = self.cfg.ranges.ang_vel_z
        ang_abs_max = max(abs(ang_low), abs(ang_high))
        pure_turn_min = min(self.cfg.pure_turn_min_abs, ang_abs_max) if ang_abs_max > 1e-4 else 0.0

        # -------- 1. 纯转向桶 --------
        pure_turn_prob = float(getattr(self.cfg, "pure_turn_probability", 0.0))
        if pure_turn_prob > 0.0 and pure_turn_min > 1e-4:
            turn_mask = torch.rand(n, device=self.device) < pure_turn_prob
        else:
            turn_mask = torch.zeros(n, dtype=torch.bool, device=self.device)

        turn_ids = env_ids_tensor[turn_mask]
        if turn_ids.numel() > 0:
            magnitudes = math_utils.sample_uniform(
                pure_turn_min, ang_abs_max, (turn_ids.numel(),), device=self.device
            )
            signs = torch.randint(0, 2, (turn_ids.numel(),), device=self.device) * 2 - 1
            self.vel_command_b[turn_ids, 0] = 0.0
            self.vel_command_b[turn_ids, 1] = 0.0
            self.vel_command_b[turn_ids, 2] = signs * magnitudes
            # 关闭 heading 控制，防止 _update_command 用航向误差覆盖 wz
            if hasattr(self, "is_heading_env"):
                self.is_heading_env[turn_ids] = False
            # 关闭 standing 覆盖，保证 wz 能存活
            if hasattr(self, "is_standing_env"):
                self.is_standing_env[turn_ids] = False

        # -------- 2. 其余环境：沿用原阈值，但 lin/ang 都很小才截断 --------
        other_mask = ~turn_mask
        other_ids = env_ids_tensor[other_mask]
        if other_ids.numel() > 0:
            vel_norm = torch.norm(self.vel_command_b[other_ids, :2], dim=1)
            ang_abs = self.vel_command_b[other_ids, 2].abs()
            invalid_mask = (vel_norm < 0.5) & (ang_abs < 0.5) & (vel_norm > 1e-4)

            if invalid_mask.any():
                invalid_ids = other_ids[invalid_mask]
                num_invalid = invalid_ids.numel()

                directions = torch.randint(0, 2, (num_invalid,), device=self.device) * 2 - 1
                magnitudes = math_utils.sample_uniform(0.5, 1.5, (num_invalid,), device=self.device)

                self.vel_command_b[invalid_ids, 0] = directions * magnitudes
                self.vel_command_b[invalid_ids, 1] = 0.0


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

    pure_turn_probability: float = 0.15
    """Fraction of environments that receive a pure-turn command (vx=vy=0, |wz|>=pure_turn_min_abs)."""

    pure_turn_min_abs: float = 0.5
    """Minimum absolute wz value for pure-turn samples; clipped to the configured ang_vel_z range."""


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
        # 1. 执行基类采样 (已经包含纯转向桶 + lin/ang 双小截断)
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

        # 4. 对被 hard 覆盖过的环境再跑一次 "lin/ang 双小" 截断，保持规则一致
        if len(hard_ids) > 0:
            vel_norm = torch.norm(self.vel_command_b[hard_ids, :2], dim=1)
            ang_abs = self.vel_command_b[hard_ids, 2].abs()
            invalid_mask = (vel_norm < 0.5) & (ang_abs < 0.5) & (vel_norm > 1e-4)

            if invalid_mask.any():
                invalid_ids = hard_ids[invalid_mask]
                directions = torch.randint(0, 2, (len(invalid_ids),), device=self.device) * 2 - 1
                magnitudes = math_utils.sample_uniform(0.5, 1.5, (len(invalid_ids),), device=self.device)

                self.vel_command_b[invalid_ids, 0] = directions * magnitudes
                self.vel_command_b[invalid_ids, 1] = 0.0
        
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
