from __future__ import annotations

from dataclasses import MISSING

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import custom_terrains


@configclass
class MeshSquareHurdleTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a square hurdle frame enclosing the center platform."""

    function = custom_terrains.square_hurdle_terrain

    mode: str = "hurdle"
    """Terrain mode: ``"hurdle"`` (step over, height increases with difficulty) or
    ``"crawl"`` (duck under, height decreases with difficulty)."""

    hurdle_height_range: tuple[float, float] = MISSING
    """Range of clearance heights under the bar (m). Order does not matter."""

    platform_width: float = 1.0
    """Width of the inner open space (m)."""

    post_thickness: float = 0.08
    """Cross-section side length of the four vertical posts (m)."""

    bar_width: float = 0.05
    """Depth of the top bar along the robot's travel direction (m).

    Used only if ``bar_width_range`` is None; otherwise the generator samples
    uniformly from the range per tile.
    """

    bar_width_range: tuple[float, float] | None = None
    """Optional per-tile uniform sample range for ``bar_width`` (m).

    When set, each generated tile draws a fresh ``bar_width`` from this range,
    giving within-column randomness without needing extra sub-terrain columns.
    """

    bar_thickness: float = 0.03
    """Thickness of the top bar along the z-axis (m).

    Used only if ``bar_thickness_range`` is None.
    """

    bar_thickness_range: tuple[float, float] | None = None
    """Optional per-tile uniform sample range for ``bar_thickness`` (m)."""


@configclass
class MeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the center platform.

    Unlike IsaacLab's built-in ``MeshGapTerrainCfg``, this version adds a
    floor at a configurable depth so that the gap is not infinitely deep.
    Set ``gap_depth`` to ``None`` to recover the original infinite-depth
    behaviour.
    """

    function = custom_terrains.gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    """Min and max width of the gap (m)."""

    platform_width: float = 1.0
    """Width of the square platform at the centre of the terrain (m)."""

    gap_depth: float | None = 0.5
    """Depth of the gap floor below the walking surface (m).
    ``None`` gives an infinite-depth gap (no floor mesh)."""
