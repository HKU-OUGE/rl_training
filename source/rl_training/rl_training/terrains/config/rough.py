# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by: Tianyang TANG


"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg
MOE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=30,
    num_cols=18,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=3.0/18,  # 2/18 的概率生成此地形
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=3.0/18,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=3.0/18,
            stone_height_max=0.01,         
            stone_width_range=(1.5, 1.5), 
            stone_distance_range=(0.1, 0.5), 
            holes_depth=-0.5,
            platform_width=2.0,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=2.0/18, rail_thickness_range=(0.05, 0.1), rail_height_range=(0.05, 0.5),platform_width=2.0
        ),
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=3.0/18,                           # 完全生成此地形
            ring_width_range=(0.1, 0.5),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.4, 0.65),             # 环的离地高度范围
            ring_thickness=0.35,                       # 环厚度（z 方向）
            platform_width=2.0,                       # 地形中心的方形平台大小
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0/18, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0/18, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
    },
)

RING_TEST_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=1.0,                           # 完全生成此地形
            ring_width_range=(0.1, 0.5),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.4, 0.8),             # 环的离地高度范围
            ring_thickness=0.2,                       # 环厚度（z 方向）
            platform_width=2.0,                       # 地形中心的方形平台大小
        ),
    },
)

STEPPING_STONE_TEST_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=0.01,         
            stone_width_range=(1.5, 1.5), 
            stone_distance_range=(0.1, 0.3), 
            holes_depth=-0.5,
            platform_width=2.0,
        ),
    },
)

HIGH_BOX_TEST_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stepping_stones": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0, 
            pit_depth_range=(0.05, 1.0), 
            double_pit=False,
            platform_width=4.0,
        ),
    },
)


MOE_ROUGH_TEST_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=7,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=0.01,         
            stone_width_range=(1.5, 1.5), 
            stone_distance_range=(0.1, 0.3), 
            holes_depth=-0.5,
            platform_width=2.0,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=1.0, rail_thickness_range=(0.05, 0.1), rail_height_range=(0.05, 0.23),platform_width=2.0
        ),

        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.15), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=1.0,                           # 完全生成此地形
            ring_width_range=(0.1, 0.5),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.65, 0.65),             # 环的离地高度范围
            ring_thickness=0.2,                       # 环厚度（z 方向）
            platform_width=2.0,                       # 地形中心的方形平台大小
        ),
    },
)
"""Rough terrains configuration."""
MOE_STUDENT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=30,
    num_cols=14,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=4.0/18,  # 2/18 的概率生成此地形
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=4.0/18,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=2.0/18,
            stone_height_max=0.01,         
            stone_width_range=(1.5, 1.5), 
            stone_distance_range=(0.1, 0.2), 
            holes_depth=-0.5,
            platform_width=2.0,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=3.0/18, rail_thickness_range=(0.05, 0.1), rail_height_range=(0.05, 0.3),platform_width=2.0
        ),
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=3.0/18,                           # 完全生成此地形
            ring_width_range=(0.1, 0.5),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.4, 0.65),             # 环的离地高度范围
            ring_thickness=0.35,                       # 环厚度（z 方向）
            platform_width=2.0,                       # 地形中心的方形平台大小
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
