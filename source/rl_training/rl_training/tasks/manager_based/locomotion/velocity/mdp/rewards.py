# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     # print(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1), "command norm")
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # return reward

# def feet_air_time(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     mode_time: float,
#     velocity_threshold: float,
# ) -> torch.Tensor:
#     """Reward longer feet air and contact time."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: Articulation = env.scene[asset_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     t_max = torch.max(current_air_time, current_contact_time)
#     t_min = torch.clip(t_max, max=mode_time)
#     stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
#     cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
#     reward = torch.where(
#         torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
#         torch.where(t_max < mode_time, t_min, 0),
#         stance_cmd_reward,
#     )
#     return torch.sum(reward, dim=1)


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)
    # print(last_air_time, "last air time")
    # print(last_contact_time, "last contact time")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward




def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.5
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # print(contact, "contact")
    reward = torch.sum(contact, dim=-1).float()
    # print(reward, "reward after sum")
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.5
    # print(env.command_manager.get_command(command_name), "env.command_manager.get_command(command_name)")
    # print(reward, "reward after multiply")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # foot_velocity_tanh = torch.tanh(
    #     tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    # )
    # reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward = torch.sum(foot_z_target_error, dim=1)
    # print(foot_z_target_error, "foot_z_target_error")
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.2
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# def stand_still_joint_deviation_l1(
#     env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize offsets from the default joint positions when the command is very small."""
    # command = env.command_manager.get_command(command_name)
#     # Penalize motion when command is nearly zero.
#     return joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :], dim=1) < command_threshold)

# def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize joint positions that deviate from the default one."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # compute out of limits constraints
#     angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     return torch.sum(torch.abs(angle), dim=1)


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def joint_acc_l2_new(env: ManagerBasedRLEnv) -> torch.Tensor:

# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     # print(torch.sum(diff, dim=1), "smoothness l2")
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward





def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_including_ang_z(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward

def lin_vel_xy_l2_with_ang_z_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Penalize xy-axis base linear velocity using L2 squared kernel if command is ang_vel_z."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    command = env.command_manager.get_command(command_name)
    reward *= (torch.sum(torch.square(command[:, 2:]), dim=1) > command_threshold) & \
            (torch.sum(torch.square(command[:, :2]), dim=1) < command_threshold)
    # reward *= torch.sum(torch.square(env.command_manager.get_command(command_name)[:, 2:]), dim=1) > command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def lin_vel_z_l2_curriculum(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalize z-axis velocity, but relax the penalty on harder terrains.
    Assuming max terrain level is roughly 20-30.
    """
    # 1. 计算原始惩罚
    asset: RigidObject = env.scene[asset_cfg.name]
    raw_penalty = torch.square(asset.data.root_lin_vel_b[:, 2])
    
    # 2. 获取当前地形等级 (0 ~ max_level)
    # 注意：如果未启用 terrain curriculum，这将全是 0
    curr_levels = env.scene.terrain.terrain_levels.float()
    
    # 3. 计算缩放因子 (Curriculum Factor)
    # 在 Level 0，因子为 1.0 (全额惩罚)
    # 在 Level 15+，因子逐渐降低到 0.1 或更低
    # 这里的 20.0 是一个软上限，你可以根据 num_rows 调整
    scale = torch.clamp(1.0 - (curr_levels / 20.0), min=0.5, max=1.0)
    
    return raw_penalty * scale

def base_height_l2_curriculum(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """
    Penalize height deviation, but relax strictly on harder terrains to allow crouching/jumping.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 标准的高度计算逻辑
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height

    # 计算偏差
    error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    
    # --- 课程逻辑 ---
    # 高难度地形下（如钻圈、高台阶），允许更大的高度误差
    curr_levels = env.scene.terrain.terrain_levels.float()
    scale = torch.clamp(1.0 - (curr_levels / 15.0), min=0.2, max=1.0)
    
    return error * scale

def feet_air_time_curriculum(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float
) -> torch.Tensor:
    """
    Reward long steps taken by the feet, scaled by terrain difficulty.
    - Flat terrain (Level 0): Reward is 0. Encourages efficient wheel rolling.
    - Rough terrain (Level N): Reward scales up to 1.0. Encourages stepping/jumping over obstacles.
    """
    # 提取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 计算基础的腾空时间奖励 (只在脚掌落地的瞬间结算)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    raw_reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    
    # 过滤微小指令：如果没有移动指令，不给予腾空奖励
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    raw_reward *= (command_norm > 0.1)
    
    # --- 核心：地形课程 (Curriculum) 缩放 ---
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        curr_levels = env.scene.terrain.terrain_levels.float()
        
        # 设定一个过渡区间，例如从 Level 0 到 Level 5
        # Level 0 (平地): scale = 0.0 -> 纯轮式
        # Level 5+ (复杂地形): scale = 1.0 -> 强制足式抬腿
        # 你可以根据你地形生成的难度，调整这里的 5.0
        scale = torch.clamp(curr_levels / 5.0, min=0.0, max=1.0)
    else:
        # 如果环境没有启用 terrain curriculum，则默认给全额奖励
        scale = 1.0
        
    return raw_reward * scale

def track_ang_vel_z_exp_tool(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float
) -> torch.Tensor:
    """奖励机器人精确跟踪 Z 轴角速度指令。
    
    使用指数内核 (Exponential Kernel)，误差越小奖励越高，且对大误差有很好的惩罚过渡。
    """
    # 1. 获取当前机身的角速度 (Base angular velocity in body frame)
    ang_vel_z = env.scene["robot"].data.root_com_ang_vel_b[:, 2]
    
    # 2. 获取目标指令角速度
    target_ang_vel_z = env.command_manager.get_command(command_name)[:, 2]
    
    # 3. 计算误差并映射到 [0, 1]
    error = torch.square(target_ang_vel_z - ang_vel_z)
    reward = torch.exp(-error / std)
    
    return reward

def yaw_foot_placement_rotation(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """转向时的‘足端落点’启发奖励。
    
    逻辑：当机器人向左转时，右侧的脚应该向前迈，左侧的脚应该向后迈（相对于机身）。
    这能有效防止机器人原地‘搓地’，强迫它走出漂亮的圆弧步。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取足端相对于机身中心(CoM)的水平位置 (x, y)
    foot_pos_b = contact_sensor.data.test_points_w[:, :, :2] - env.scene["robot"].data.root_com_pos_w[:, None, :2]
    
    # 获取转向指令
    yaw_command = env.command_manager.get_command(command_name)[:, 2] # rad/s
    
    # 计算每个足端在转向时理想的切向位移趋势
    # 简化逻辑：向左转(yaw > 0)时，位于机身右侧的脚(y < 0)应该有正向的x速度/位置
    # reward = yaw_rate * (foot_pos_x * (-foot_pos_y) + foot_pos_y * (foot_pos_x))
    # 这里我们只奖励在摆动相(Swing)结束准备落地时，脚是否站到了能产生转矩的位置
    
    # 简易版：奖励转向指令与足端相对位置的乘积一致性
    reward = yaw_command[:, None] * foot_pos_b[:, :, 0] * (-foot_pos_b[:, :, 1])
    
    return torch.sum(reward, dim=1)

    import isaaclab.utils.math as math_utils
from isaaclab.sensors import ContactSensor

def wheel_lateral_slip_penalty(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """轮足终极防侧滑：只惩罚接地的轮子在基座 Y 轴（侧向）的滑动"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 1. 获取轮子在世界坐标系下的线速度 (B, 4, 3)
    wheel_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    
    # 2. 获取基座的姿态并扩展，用于坐标系转换
    base_quat = asset.data.root_quat_w
    num_wheels = wheel_vel_w.shape[1]
    base_quat_expanded = base_quat.unsqueeze(1).repeat(1, num_wheels, 1).view(-1, 4)
    wheel_vel_w_flat = wheel_vel_w.view(-1, 3)
    
    # 3. 将轮子的速度转换到基座坐标系 (Base Frame) 下
    wheel_vel_b = math_utils.quat_apply_inverse(base_quat_expanded, wheel_vel_w_flat).view(wheel_vel_w.shape)
    
    # 4. 提取侧向速度 (假设基座系下 X是前进，Y是侧向)
    lateral_vel_sq = torch.square(wheel_vel_b[:, :, 1])
    
    # 5. 检测轮子是否触地
    # 注意：这里的 contact_sensor 应该指向 contact_forces 传感器
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    
    # 6. 只有触地时的侧滑才会导致惩罚
    return torch.sum(lateral_vel_sq * contacts, dim=1)

def track_lin_vel_xy_exp_curriculum(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """带地形课程的线速度跟踪奖励：地形越难，将标准差(std)放宽，扩大误差容忍区间"""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        curr_levels = env.scene.terrain.terrain_levels.float()
        # 核心：计算 std 乘数。
        # 平地 (Level 0) -> std_mult = 1.0 (保持原本的高精度要求)
        # 高台 (比如 Level 30) -> std_mult = 3.0 (容忍度扩大3倍，误差大也不会直接 0 分)
        std_mult = torch.clamp(1.0 + (curr_levels / 15.0), min=1.0, max=3.0)
    else:
        std_mult = 1.0
        
    dynamic_std = std * std_mult
    return torch.exp(-lin_vel_error / (dynamic_std**2))


def track_ang_vel_z_exp_curriculum(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """带地形课程的角速度跟踪奖励：动态放宽 std"""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        curr_levels = env.scene.terrain.terrain_levels.float()
        std_mult = torch.clamp(1.0 + (curr_levels / 15.0), min=1.0, max=3.0)
    else:
        std_mult = 1.0
        
    dynamic_std = std * std_mult
    return torch.exp(-ang_vel_error / (dynamic_std**2))

def base_roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize ONLY the base roll angle using L2 squared kernel.
    
    Computed by penalizing the y-component of the projected gravity vector.
    This prevents lateral tilting without penalizing pitching on slopes.
    """
    # extract the used quantities
    asset: RigidObject = env.scene[asset_cfg.name]
    
    reward = torch.square(asset.data.projected_gravity_b[:, 1])
    
    return reward

# ==============================================================================
# Acrobatic Stage-Wise Rewards (后空翻、侧翻、侧滚、双手行走)
# ==============================================================================

def acrobatic_router_reward(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    skill_weights: dict[str, float], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    路由函数：根据当前的杂技指令 (0:站立, 1:后空翻, 2:侧翻, 3:侧滚, 4:双轮直立) 分发奖励。
    """
    cmd = env.command_manager.get_command(command_name)[:, 0].long()
    rewards = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
    # 获取有效索引
    idx_backflip = (cmd == 1).nonzero(as_tuple=False).flatten()
    idx_sideflip = (cmd == 2).nonzero(as_tuple=False).flatten()
    idx_sideroll = (cmd == 3).nonzero(as_tuple=False).flatten()
    idx_handstand = (cmd == 4).nonzero(as_tuple=False).flatten()
    
    if len(idx_backflip) > 0:
        rewards[idx_backflip] += backflip_stage_reward(env, idx_backflip, asset_cfg) * skill_weights["backflip"]
        
    if len(idx_sideflip) > 0:
        rewards[idx_sideflip] += sideflip_stage_reward(env, idx_sideflip, asset_cfg) * skill_weights["sideflip"]
        
    if len(idx_sideroll) > 0:
        rewards[idx_sideroll] += sideroll_stage_reward(env, idx_sideroll, asset_cfg) * skill_weights["sideroll"]
        
    if len(idx_handstand) > 0:
        rewards[idx_handstand] += handstand_reward(env, idx_handstand, asset_cfg) * skill_weights["handstand"]
        
    return rewards

def backflip_stage_reward(env: ManagerBasedRLEnv, ids: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """后空翻分阶段奖励"""
    asset = env.scene[asset_cfg.name]
    time = env.episode_length_buf[ids] * env.step_dt
    reward = torch.zeros_like(ids, dtype=torch.float32)
    
    pitch_vel = asset.data.root_ang_vel_b[ids, 1]
    root_z_vel = asset.data.root_lin_vel_w[ids, 2]
    root_z = asset.data.root_pos_w[ids, 2]
    proj_g = asset.data.projected_gravity_b[ids]
    
    # 阶段 1: 深蹲蓄力 (0.0s - 0.4s)
    s1 = (time <= 0.4)
    if s1.any():
        reward[s1] += torch.clamp(0.25 - root_z[s1], min=0.0) * 10.0 # 鼓励降低质心
        
    # 阶段 2: 爆发起跳 (0.4s - 0.8s)
    s2 = (time > 0.4) & (time <= 0.8)
    if s2.any():
        reward[s2] += torch.clamp(root_z_vel[s2], min=0.0, max=3.0) * 2.0 # 鼓励向上速度
        reward[s2] += torch.clamp(-pitch_vel[s2], min=0.0, max=10.0) * 1.5 # 鼓励向后仰的角速度
        # 利用轮子动量：奖励起跳时轮子高速旋转
        wheel_vel = asset.data.joint_vel[ids[s2], -4:] 
        reward[s2] += torch.mean(torch.abs(wheel_vel), dim=-1) * 0.05
        
    # 阶段 3: 空中翻转 (0.8s - 1.2s)
    s3 = (time > 0.8) & (time <= 1.2)
    if s3.any():
        reward[s3] += torch.clamp(-pitch_vel[s3], min=0.0, max=15.0) * 2.0 # 保持空中向后旋转
        
    # 阶段 4: 落地缓冲 (1.2s+)
    s4 = (time > 1.2)
    if s4.any():
        # 鼓励姿态恢复水平 (重力投影回归 [0, 0, -1])
        target_g = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(s4.sum().item(), 1)
        reward[s4] += torch.exp(-torch.norm(proj_g[s4] - target_g, dim=-1)) * 5.0
        
    return reward

def sideflip_stage_reward(env: ManagerBasedRLEnv, ids: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """侧翻分阶段奖励"""
    asset = env.scene[asset_cfg.name]
    time = env.episode_length_buf[ids] * env.step_dt
    reward = torch.zeros_like(ids, dtype=torch.float32)
    
    roll_vel = asset.data.root_ang_vel_b[ids, 0] # 侧翻依赖 Roll
    root_z_vel = asset.data.root_lin_vel_w[ids, 2]
    proj_g = asset.data.projected_gravity_b[ids]
    
    # 阶段 1: 深蹲蓄力 (0.0s - 0.4s)
    s1 = (time <= 0.4)
    if s1.any():
        reward[s1] += torch.clamp(0.25 - asset.data.root_pos_w[ids[s1], 2], min=0.0) * 10.0
        
    # 阶段 2: 侧向爆发起跳 (0.4s - 0.8s)
    s2 = (time > 0.4) & (time <= 0.8)
    if s2.any():
        reward[s2] += torch.clamp(root_z_vel[s2], min=0.0, max=3.0) * 2.0 
        reward[s2] += torch.clamp(torch.abs(roll_vel[s2]), min=0.0, max=10.0) * 1.5 
        
    # 阶段 3: 空中翻转 (0.8s - 1.2s)
    s3 = (time > 0.8) & (time <= 1.2)
    if s3.any():
        reward[s3] += torch.clamp(torch.abs(roll_vel[s3]), min=0.0, max=15.0) * 2.0 
        
    # 阶段 4: 落地缓冲 (1.2s+)
    s4 = (time > 1.2)
    if s4.any():
        target_g = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(s4.sum().item(), 1)
        reward[s4] += torch.exp(-torch.norm(proj_g[s4] - target_g, dim=-1)) * 5.0
        
    return reward

def sideroll_stage_reward(env: ManagerBasedRLEnv, ids: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """地面侧滚分阶段奖励"""
    asset = env.scene[asset_cfg.name]
    time = env.episode_length_buf[ids] * env.step_dt
    reward = torch.zeros_like(ids, dtype=torch.float32)
    
    roll_vel = asset.data.root_ang_vel_b[ids, 0]
    proj_g = asset.data.projected_gravity_b[ids]
    
    # 阶段 1: 侧向倾倒 (0.0s - 0.5s)
    s1 = (time <= 0.5)
    if s1.any():
        reward[s1] += torch.clamp(torch.abs(roll_vel[s1]), min=0.0, max=5.0) * 2.0
        
    # 阶段 2: 地面连续翻滚 (0.5s - 1.5s)
    s2 = (time > 0.5) & (time <= 1.5)
    if s2.any():
        reward[s2] += torch.clamp(torch.abs(roll_vel[s2]), min=2.0, max=8.0) * 3.0
        
    # 阶段 3: 重新站立 (1.5s+)
    s3 = (time > 1.5)
    if s3.any():
        target_g = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(s3.sum().item(), 1)
        reward[s3] += torch.exp(-torch.norm(proj_g[s3] - target_g, dim=-1)) * 5.0
        
    return reward

def handstand_reward(env: ManagerBasedRLEnv, ids: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """双后轮直立行走奖励 (连续保持)"""
    asset = env.scene[asset_cfg.name]
    reward = torch.zeros_like(ids, dtype=torch.float32)
    
    # 直立时，机器人的机头(X轴)应该指向正上方。
    # 假设默认站立时重力投影是 [0, 0, -1]，机头朝上时重力投影在局部系下应为 [-1, 0, 0]
    target_g = torch.tensor([-1.0, 0.0, 0.0], device=env.device).repeat(len(ids), 1)
    proj_g = asset.data.projected_gravity_b[ids]
    
    # 姿态保持奖励
    posture_error = torch.norm(proj_g - target_g, dim=-1)
    reward += torch.exp(-posture_error / 0.5) * 5.0
    
    # 高度奖励 (直立后质心会变高)
    root_z = asset.data.root_pos_w[ids, 2]
    reward += torch.clamp(root_z - 0.4, min=0.0) * 2.0 
    
    return reward