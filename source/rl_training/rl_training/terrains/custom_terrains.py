from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from .custom_terrains_cfg import MeshGapTerrainCfg, MeshSquareHurdleTerrainCfg


def square_hurdle_terrain(
    difficulty: float, cfg: MeshSquareHurdleTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a square hurdle frame enclosing the center platform.

    In ``"hurdle"`` mode the clearance increases with difficulty (step over).
    In ``"crawl"`` mode the clearance decreases with difficulty (duck under).
    """
    min_h = min(cfg.hurdle_height_range)
    max_h = max(cfg.hurdle_height_range)

    if cfg.mode == "crawl":
        start_h, end_h = max_h, min_h
    else:
        start_h, end_h = min_h, max_h

    hurdle_clearance = start_h + difficulty * (end_h - start_h)

    meshes_list: list[trimesh.Trimesh] = []

    t_post = cfg.post_thickness

    # Per-tile random sampling of bar dimensions for in-column variety.
    # Falls back to the scalar config field when the range is not provided.
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
    difficulty: float, cfg: MeshGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a gap of finite depth around the center platform.

    Unlike IsaacLab's built-in ``MeshGapTerrainCfg`` where the gap is infinitely
    deep (no floor geometry), this version places a floor at ``gap_depth`` below
    the walking surface so that the gap has a realistic, configurable depth.
    """
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
