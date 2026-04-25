from __future__ import annotations

from dataclasses import MISSING

from isaaclab.terrains.height_field.hf_terrains_cfg import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
)
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshBoxTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPitTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRailsTerrainCfg,
    MeshRandomGridTerrainCfg,
)
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
    """Optional per-tile uniform sample range for ``bar_width`` (m)."""

    bar_thickness: float = 0.03
    """Thickness of the top bar along the z-axis (m)."""

    bar_thickness_range: tuple[float, float] | None = None
    """Optional per-tile uniform sample range for ``bar_thickness`` (m)."""


@configclass
class MeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the center platform."""

    function = custom_terrains.gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    """Min and max width of the gap (m)."""

    platform_width: float = 1.0
    """Width of the square platform at the centre of the terrain (m)."""

    gap_depth: float | None = 0.5
    """Depth of the gap floor below the walking surface (m)."""


# ============================================================================
# Noisy variants — each subclasses the matching base config and adds three
# platform-noise fields. The actual noise overlay is built by the matching
# noisy_* generator function (see custom_terrains.py).
# ============================================================================


@configclass
class _PlatformNoiseFields:
    """Shared platform-noise parameters injected into every Noisy* config.

    The Noisy* configs subclass this alongside their base config so the
    ``noisy_*`` generator can read the same three fields uniformly.
    """

    # Defaults intentionally mirror ``HfRandomUniformTerrainCfg`` (the
    # ``random_rough`` look) so each Noisy* variant's platform area matches
    # the textbook ``random_rough`` surface out of the box.
    platform_noise_range: tuple[float, float] = (0.02, 0.16)
    """(h_min, h_max) per-vertex random height (m). Need ``h_min < h_max`` for
    any visible noise — degenerate ranges produce a flat plateau."""

    platform_noise_step: float = 0.02
    """Discrete snap step for the random heights (m). 0 disables snapping."""

    platform_noise_horizontal_step: float = 0.2
    """Cell size of the noise grid (m). Smaller = denser noise but more
    triangles. Total terrain tris scale with (1/step)^2 across all noisy
    tiles, so dropping to 0.2 keeps the full 33-column build under
    PhysX's practical cooking budget; bump back to 0.1 only if you reduce
    the variant count."""

    platform_noise_size_ratio: float = 1.0
    """Side of the noise patch as a fraction of the cfg's ``platform_width``."""


@configclass
class NoisyMeshPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_pyramid_stairs_terrain


@configclass
class NoisyMeshInvertedPyramidStairsTerrainCfg(MeshInvertedPyramidStairsTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_inverted_pyramid_stairs_terrain


@configclass
class NoisyHfPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_pyramid_sloped_terrain


@configclass
class NoisyHfInvertedPyramidSlopedTerrainCfg(HfInvertedPyramidSlopedTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_inverted_pyramid_sloped_terrain


@configclass
class NoisyMeshSquareHurdleTerrainCfg(MeshSquareHurdleTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_square_hurdle_terrain


@configclass
class NoisyMeshRailsTerrainCfg(MeshRailsTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_rails_terrain


@configclass
class NoisyMeshGapTerrainCfg(MeshGapTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_gap_terrain


@configclass
class NoisyMeshPitTerrainCfg(MeshPitTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_pit_terrain


@configclass
class NoisyMeshBoxTerrainCfg(MeshBoxTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_box_terrain


@configclass
class NoisyMeshRandomGridTerrainCfg(MeshRandomGridTerrainCfg, _PlatformNoiseFields):
    function = custom_terrains.noisy_random_grid_terrain
