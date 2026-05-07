# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Custom termination terms for velocity locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_filtered_contact(
    env: "ManagerBasedRLEnv",
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when contact force with FILTERED prims exceeds threshold.

    Unlike :func:`isaaclab.envs.mdp.terminations.illegal_contact` which reads
    ``net_forces_w`` (total contact force incl. ground), this reads
    ``force_matrix_w`` which is the contact force *per-filter-prim* and respects
    the ContactSensor's ``filter_prim_paths_expr``.

    Use when you want to terminate only on contact with a specific obstacle prim
    (e.g. rail) without terminating on regular ground contact.

    Requires :attr:`ContactSensorCfg.filter_prim_paths_expr` to be non-empty
    (otherwise ``force_matrix_w`` is ``None`` and termination always returns False).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.force_matrix_w  # (n_env, n_body, n_filter, 3) or None
    if forces is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    body_ids = sensor_cfg.body_ids
    if body_ids is not None and not isinstance(body_ids, slice):
        forces = forces[:, body_ids]

    # norm over xyz, max over (body, filter), compare to threshold
    return torch.amax(torch.norm(forces, dim=-1), dim=(1, 2)) > threshold
