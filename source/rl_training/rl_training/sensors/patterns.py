"""Forward-facing hemispherical LiDAR pattern.

Matches the geometry used in sdk_deploy mujoco sim2sim, which approximates a
real Robosense Airy: rays uniform in (polar, azimuth) where polar is the
angle from the boresight axis (+X local) and azimuth sweeps a full circle
around it. Coverage is the full forward hemisphere (180° solid angle).

Differences from isaaclab's built-in LidarPatternCfg:
  - Pole at sensor-local +X (boresight) rather than +Z.
  - Circular FOV (hemispherical) rather than rectangular (elevation × azimuth).
  - Density biased toward boresight (azimuth points collapse near the pole),
    matching forward-facing Airy-like sensors that need fine-grained info
    near boresight.

Flat ray layout is row-major (polar, azimuth) — polar is the outer dim, so
``data.reshape(num_polar, num_azimuth)`` recovers the per-channel layout
expected by MultiLayerScanAE.

LR-symmetry note: with ``azimuth = linspace(-pi, pi, num_azimuth)``, the
linspace is symmetric about 0, so ``flip(azimuth_dim)`` corresponds to
``azimuth -> -azimuth``. Combined with ``y = sin(polar) * sin(azimuth)``,
this is equivalent to ``y -> -y`` — the standard left/right mirror used by
the symmetry loss.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass


def hemispherical_lidar_pattern(
    cfg: "HemisphericalLidarPatternCfg", device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    polar = torch.linspace(0.0, torch.pi / 2, cfg.num_polar, device=device)
    azimuth = torch.linspace(-torch.pi, torch.pi, cfg.num_azimuth, device=device)

    pol_g, az_g = torch.meshgrid(polar, azimuth, indexing="ij")

    # Pole at +X (boresight). Azimuth measured from +Z (top), increasing toward +Y (left)
    # so that y = sin(polar) * sin(azimuth) and azimuth->-azimuth flips left/right.
    x = torch.cos(pol_g)
    y = torch.sin(pol_g) * torch.sin(az_g)
    z = torch.sin(pol_g) * torch.cos(az_g)

    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions


@configclass
class HemisphericalLidarPatternCfg(PatternBaseCfg):
    """Hemispherical LiDAR pattern (forward-facing, pole at sensor-local +X).

    Use ``RayCasterCfg.OffsetCfg.rot`` on the parent sensor to point the
    boresight in the desired robot-frame direction (e.g. identity for a
    front-facing sensor, 180° around Z for a rear-facing one).
    """

    func: Callable = hemispherical_lidar_pattern

    num_polar: int = MISSING
    """Polar samples in [0, π/2]. 0 = boresight (pole), π/2 = rim of hemisphere."""

    num_azimuth: int = MISSING
    """Azimuth samples in [-π, π] (symmetric, includes one duplicate at ±π so flip ↔ negation)."""
