from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

import isaaclab.terrains.height_field.hf_terrains as hf_terrains
import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains

if TYPE_CHECKING:
    from .custom_terrains_cfg import (
        MeshGapTerrainCfg,
        MeshSquareHurdleTerrainCfg,
        NoisyHfPyramidSlopedTerrainCfg,
        NoisyHfInvertedPyramidSlopedTerrainCfg,
        NoisyMeshBoxTerrainCfg,
        NoisyMeshGapTerrainCfg,
        NoisyMeshInvertedPyramidStairsTerrainCfg,
        NoisyMeshPitTerrainCfg,
        NoisyMeshPyramidStairsTerrainCfg,
        NoisyMeshRailsTerrainCfg,
        NoisyMeshRandomGridTerrainCfg,
        NoisyMeshSquareHurdleTerrainCfg,
    )


# ============================================================================
# Internal helper: tile of small random-height boxes around a center point.
#
# Used by all Noisy* generators to overlay platform-area noise on top of an
# already-built base terrain. The overlay is centered at ``origin`` (the
# spawn point of the sub-terrain) and only covers a square of side
# ``platform_size`` around it, so the global terrain layout is preserved.
# ============================================================================


def _add_platform_noise(
    meshes: list[trimesh.Trimesh],
    origin: np.ndarray,
    platform_size: float,
    noise_height_range: tuple[float, float],
    noise_step: float,
    horizontal_step: float,
    difficulty: float = 1.0,  # noqa: ARG001 — kept for API compatibility
    z_clearance: float = 0.0,
) -> list[trimesh.Trimesh]:
    """Append a single heightfield-style mesh patch over the platform.

    Semantics match :class:`isaaclab.terrains.HfRandomUniformTerrainCfg`:
    each grid vertex draws a height uniformly from ``noise_height_range``
    (snapped to ``noise_step``), so to actually see noise you need
    ``min < max``. Setting both ends to the same value yields a flat
    plateau, not bumps.

    Earlier versions appended one trimesh box per noise cell; for a 40x40
    grid that produced ~19k triangles per tile and exceeded the PhysX
    cooking limit once concatenated. We instead build one triangle mesh of
    shape (n+1) x (n+1) shared vertices — same visual effect as
    ``random_rough``, ~6x fewer triangles, no self-intersections with the
    underlying ground mesh.

    Note: ``difficulty`` is intentionally unused. ``random_rough`` itself
    ignores the difficulty parameter, so we keep noise amplitude constant
    across the curriculum and let the underlying structured terrain
    (stairs/hurdle/etc.) handle progressive difficulty.

    Args:
        meshes: existing terrain meshes (mutated in-place by appending).
        origin: (x, y, z) center of the platform (sub-terrain spawn point).
        platform_size: side length of the square noise patch (m).
        noise_height_range: (h_min, h_max) per-vertex height range (m).
        noise_step: discrete step (m). 0 disables snapping.
        horizontal_step: cell size of the noise grid (m).
        z_clearance: extra z-offset added on top of ``origin[2]``.
    """
    if platform_size <= 0.0 or horizontal_step <= 0.0:
        return meshes

    h_min, h_max = noise_height_range
    if h_max <= h_min:
        # Degenerate range: would yield a flat plateau, not noise.
        return meshes

    n = int(round(platform_size / horizontal_step))
    if n <= 0:
        return meshes
    cell = platform_size / n
    cx, cy, cz = float(origin[0]), float(origin[1]), float(origin[2])
    base_z = cz + z_clearance
    half = platform_size / 2.0

    # Vertex grid: (n+1) x (n+1). Each vertex carries a random z-offset.
    nv = n + 1
    xs = cx - half + np.arange(nv) * cell  # (nv,)
    ys = cy - half + np.arange(nv) * cell  # (nv,)
    xv, yv = np.meshgrid(xs, ys, indexing="ij")  # (nv, nv)
    zv = np.random.uniform(h_min, h_max, size=(nv, nv))
    if noise_step > 0.0:
        zv = np.round(zv / noise_step) * noise_step

    # Clamp the outermost ring of vertices to 0 so the patch's perimeter
    # sits flush with the platform top — same idea as ``random_rough``'s
    # ``border_width`` zero-band. Without this, edge vertices float at random
    # heights and the patch looks like a hovering rug instead of a chunk of
    # noisy ground replacing the platform surface.
    zv[0, :] = 0.0
    zv[-1, :] = 0.0
    zv[:, 0] = 0.0
    zv[:, -1] = 0.0
    # Lift the whole patch by 1mm so the clamped perimeter is just above
    # the platform top (avoids coincident triangles between the two meshes,
    # which can confuse PhysX cooking and contact resolution).
    zv = base_z + zv + 1e-3

    vertices = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1).astype(np.float32)

    # Build triangles (two per cell, n*n cells -> 2*n*n triangles).
    # Winding is CCW seen from +z so normals point upward — same as
    # ``isaaclab.terrains.height_field.utils.convert_height_field_to_mesh``.
    # An earlier version had the opposite winding, which made PhysX treat the
    # underside as the front face and pushed wheels through the patch.
    i = np.arange(n)
    j = np.arange(n)
    ii, jj = np.meshgrid(i, j, indexing="ij")
    v00 = (ii * nv + jj).ravel()
    v10 = ((ii + 1) * nv + jj).ravel()
    v01 = (ii * nv + (jj + 1)).ravel()
    v11 = ((ii + 1) * nv + (jj + 1)).ravel()
    tri_a = np.stack([v00, v10, v11], axis=1)
    tri_b = np.stack([v00, v11, v01], axis=1)
    triangles = np.concatenate([tri_a, tri_b], axis=0).astype(np.int64)

    patch = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    meshes.append(patch)
    return meshes


def _platform_size_from_cfg(cfg) -> float:
    """Pick a sensible platform side length for noise placement.

    Falls back to 80% of the smaller terrain dimension when the cfg has no
    platform_width (e.g. the random-grid terrain).
    """
    pw = getattr(cfg, "platform_width", None)
    if pw is not None and pw > 0.0:
        return float(pw)
    return 0.8 * min(cfg.size[0], cfg.size[1])


# ============================================================================
# Custom base terrains (kept from previous version)
# ============================================================================


def square_hurdle_terrain(
    difficulty: float, cfg: "MeshSquareHurdleTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a square hurdle frame enclosing the center platform."""
    min_h = min(cfg.hurdle_height_range)
    max_h = max(cfg.hurdle_height_range)

    if cfg.mode == "crawl":
        start_h, end_h = max_h, min_h
    else:
        start_h, end_h = min_h, max_h

    hurdle_clearance = start_h + difficulty * (end_h - start_h)

    meshes_list: list[trimesh.Trimesh] = []

    t_post = cfg.post_thickness

    bw_range = getattr(cfg, "bar_width_range", None)
    bt_range = getattr(cfg, "bar_thickness_range", None)
    b_w = float(np.random.uniform(*bw_range)) if bw_range is not None else cfg.bar_width
    b_t = float(np.random.uniform(*bt_range)) if bt_range is not None else cfg.bar_thickness

    W = cfg.platform_width
    H = hurdle_clearance

    terrain_height = 1.0
    center_x = 0.5 * cfg.size[0]
    center_y = 0.5 * cfg.size[1]

    ground_dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_pos = (center_x, center_y, -terrain_height / 2.0)
    meshes_list.append(
        trimesh.creation.box(ground_dim, trimesh.transformations.translation_matrix(ground_pos))
    )

    def _box(size: tuple, center: tuple):
        meshes_list.append(
            trimesh.creation.box(size, trimesh.transformations.translation_matrix(center))
        )

    bar_z = H + b_t / 2.0
    _box((b_w, W + 2 * t_post, b_t), (center_x + W / 2.0 + t_post / 2.0, center_y, bar_z))
    _box((b_w, W + 2 * t_post, b_t), (center_x - W / 2.0 - t_post / 2.0, center_y, bar_z))
    _box((W, b_w, b_t), (center_x, center_y + W / 2.0 + t_post / 2.0, bar_z))
    _box((W, b_w, b_t), (center_x, center_y - W / 2.0 - t_post / 2.0, bar_z))

    post_z = H / 2.0
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        _box(
            (t_post, t_post, H),
            (
                center_x + sx * (W / 2.0 + t_post / 2.0),
                center_y + sy * (W / 2.0 + t_post / 2.0),
                post_z,
            ),
        )

    origin = np.array([center_x, center_y, 0.0])
    return meshes_list, origin


def gap_terrain(
    difficulty: float, cfg: "MeshGapTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a gap of finite depth around the center platform."""
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])

    meshes_list: list[trimesh.Trimesh] = []

    terrain_height = 1.0
    center_x = 0.5 * cfg.size[0]
    center_y = 0.5 * cfg.size[1]
    terrain_center = (center_x, center_y, -terrain_height / 2.0)

    inner_size = (cfg.platform_width + 2 * gap_width, cfg.platform_width + 2 * gap_width)
    meshes_list += _make_border(cfg.size, inner_size, terrain_height, terrain_center)

    box_dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    meshes_list.append(
        trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    )

    if cfg.gap_depth is not None:
        floor_z = -cfg.gap_depth
        floor_dim = (inner_size[0], inner_size[1], terrain_height)
        floor_pos = (center_x, center_y, floor_z - terrain_height / 2.0)
        meshes_list.append(
            trimesh.creation.box(floor_dim, trimesh.transformations.translation_matrix(floor_pos))
        )

    origin = np.array([center_x, center_y, 0.0])
    return meshes_list, origin


def _make_border(
    size: tuple[float, float],
    inner_size: tuple[float, float],
    height: float,
    position: tuple[float, float, float],
) -> list[trimesh.Trimesh]:
    """Rectangular border with a hollow interior (standalone reimplementation)."""
    thickness_x = (size[0] - inner_size[0]) / 2.0
    thickness_y = (size[1] - inner_size[1]) / 2.0

    def _box(dims, pos):
        return trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))

    top = _box(
        (size[0], thickness_y, height),
        (position[0], position[1] + inner_size[1] / 2.0 + thickness_y / 2.0, position[2]),
    )
    bottom = _box(
        (size[0], thickness_y, height),
        (position[0], position[1] - inner_size[1] / 2.0 - thickness_y / 2.0, position[2]),
    )
    left = _box(
        (thickness_x, inner_size[1], height),
        (position[0] - inner_size[0] / 2.0 - thickness_x / 2.0, position[1], position[2]),
    )
    right = _box(
        (thickness_x, inner_size[1], height),
        (position[0] + inner_size[0] / 2.0 + thickness_x / 2.0, position[1], position[2]),
    )
    return [left, right, top, bottom]


# ============================================================================
# Noisy* terrain generators
# Each generator delegates to the matching base terrain function and then
# adds platform-area random-height bumps via :func:`_add_platform_noise`.
#
# Original noise-based terrains (random_rough) are NOT wrapped here, since
# they already inject height variation across the whole tile.
# ============================================================================


def _build_noisy(base_func, difficulty, cfg, base_is_height_field: bool = False):
    meshes, origin = base_func(difficulty, cfg)
    platform_size = _platform_size_from_cfg(cfg)
    noise_size = platform_size * cfg.platform_noise_size_ratio
    _add_platform_noise(
        meshes,
        origin=origin,
        platform_size=noise_size,
        noise_height_range=cfg.platform_noise_range,
        noise_step=cfg.platform_noise_step,
        horizontal_step=cfg.platform_noise_horizontal_step,
        difficulty=difficulty,
    )
    return meshes, origin


def noisy_pyramid_stairs_terrain(
    difficulty: float, cfg: "NoisyMeshPyramidStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _build_noisy(mesh_terrains.pyramid_stairs_terrain, difficulty, cfg)


def noisy_inverted_pyramid_stairs_terrain(
    difficulty: float, cfg: "NoisyMeshInvertedPyramidStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _build_noisy(mesh_terrains.inverted_pyramid_stairs_terrain, difficulty, cfg)


def noisy_pyramid_sloped_terrain(
    difficulty: float, cfg: "NoisyHfPyramidSlopedTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _build_noisy(hf_terrains.pyramid_sloped_terrain, difficulty, cfg, base_is_height_field=True)


def noisy_inverted_pyramid_sloped_terrain(
    difficulty: float, cfg: "NoisyHfInvertedPyramidSlopedTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # Inverted slope reuses ``pyramid_sloped_terrain`` with ``inverted=True``;
    # there's no dedicated ``inverted_pyramid_sloped_terrain`` function.
    return _build_noisy(hf_terrains.pyramid_sloped_terrain, difficulty, cfg, base_is_height_field=True)


def noisy_square_hurdle_terrain(
    difficulty: float, cfg: "NoisyMeshSquareHurdleTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _build_noisy(square_hurdle_terrain, difficulty, cfg)


def noisy_rails_terrain(
    difficulty: float, cfg: "NoisyMeshRailsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _build_noisy(mesh_terrains.rails_terrain, difficulty, cfg)


def noisy_gap_terrain(
    difficulty: float, cfg: "NoisyMeshGapTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _build_noisy(gap_terrain, difficulty, cfg)


def noisy_pit_terrain(
    difficulty: float, cfg: "NoisyMeshPitTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _build_noisy(mesh_terrains.pit_terrain, difficulty, cfg)


def noisy_box_terrain(
    difficulty: float, cfg: "NoisyMeshBoxTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _build_noisy(mesh_terrains.box_terrain, difficulty, cfg)


def noisy_random_grid_terrain(
    difficulty: float, cfg: "NoisyMeshRandomGridTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _build_noisy(mesh_terrains.random_grid_terrain, difficulty, cfg)
