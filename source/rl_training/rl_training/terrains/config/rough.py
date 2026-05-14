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
from rl_training.terrains import MeshSquareHurdleTerrainCfg
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
            proportion=1.0/18,  # 2/18 的概率生成此地形
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=2.0,  # was 3.0 — 与其他 sub-terrain 对齐 (都是 2.0)
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=3.0/18,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=2.0,  # was 3.0 — 与其他 sub-terrain 对齐 (都是 2.0)
            border_width=1.0,
            holes=False,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=3.0/18,
            stone_height_max=0.01,         
            stone_width_range=(1.5, 1.5), 
            stone_distance_range=(0.1, 0.8), 
            holes_depth=-0.5,
            platform_width=2.0,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=2.0/18, rail_thickness_range=(0.05, 0.1), rail_height_range=(0.05, 0.4),platform_width=2.0
        ),
        # hurdle 拆成 3 种墙厚 (各 1/18, 合计仍 3/18), clearance 保持
        # MOE_ROUGH 较宽松的 (0.4, 0.65) — 它是 generalist anchor, 比 SCAN specialist 容易.
        "hurdle_pole": MeshSquareHurdleTerrainCfg(   # 细杆 5×5cm
            proportion=1.0/18,
            hurdle_height_range=(0.4, 0.65),
            bar_thickness=0.05,
            bar_width=0.05,
            platform_width=2.0,
            mode="crawl",
        ),
        "hurdle_board": MeshSquareHurdleTerrainCfg(  # 高薄板 20×5cm
            proportion=1.0/18,
            hurdle_height_range=(0.4, 0.65),
            bar_thickness=0.2,
            bar_width=0.05,
            platform_width=2.0,
            mode="crawl",
        ),
        "hurdle_wall": MeshSquareHurdleTerrainCfg(   # 厚墙 20×30cm
            proportion=1.0/18,
            hurdle_height_range=(0.4, 0.65),
            bar_thickness=0.2,
            bar_width=0.3,
            platform_width=2.0,
            mode="crawl",
        ),
        "pit": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=2.0/18,      
            pit_depth_range=(0.05, 0.8), 
            double_pit=True,
            platform_width=2.0,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0/18, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0/18, noise_range=(0.02, 0.16), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
    },
)
MOE_ROUGH_TERRAINS_CFG2 = TerrainGeneratorCfg(
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
        "pyramid_stairs": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "pyramid_stairs_inv": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(3.0, 3.0),  
            double_pit=False,
            platform_width=8.0,
        ),
        "stepping_stones": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=2.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "hurdle": MeshSquareHurdleTerrainCfg(
            proportion=3.0/18,
            hurdle_height_range=(0.4, 0.75),
            bar_thickness=0.25,
            bar_width=0.05,
            platform_width=2.0,
            mode="crawl",
        ),
        "pit": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=2.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "boxes": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "random_rough": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "hf_pyramid_slope": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "hf_pyramid_slope_inv": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
    },
)

# ==============================================================================
# Per-rank single-modality TEACHER terrain configs (used by PER_RANK_TERRAIN=1)
# Rank-modality mapping (from cfb9e3b convention):
#   rank 0: FLAT          — plane only
#   rank 1: STAIR_SLOPE   — pyramid_stairs + pyramid_stairs_inv + hf_pyramid_slope (+inv)
#   rank 2: PLATFORM      — pit + boxes (climb up/down)
#   rank 3: SCAN          — hurdle (crawl mode)
#   rank 4: STEPPING_STONES — baseline 的 stepping_stones (替代之前的 GAP)
#   rank 5: RAIL          — rail bars
#   rank 6: NOISE         — random_rough
#   rank 7: GRID          — boxes (discrete grid)
# 所有 cfg 复用 baseline MOE_ROUGH_TERRAINS_CFG 里相同模态的参数, 只调 proportion / num_cols
# ==============================================================================

FLAT_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=10, num_cols=10,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=False,
    sub_terrains={
        "plane": terrain_gen.trimesh.mesh_terrains_cfg.MeshPlaneTerrainCfg(proportion=1.0),
    },
)

STAIR_SLOPE_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=30, num_cols=10,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=(0.05, 0.25),
            step_width=0.3, platform_width=3.0, border_width=1.0, holes=False),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=(0.05, 0.25),
            step_width=0.3, platform_width=3.0, border_width=1.0, holes=False),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.55),
            platform_width=2.0, border_width=0.25),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.55),
            platform_width=2.0, border_width=0.25),
    },
)

PLATFORM_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=30, num_cols=10,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=True,
    sub_terrains={
        # 双层 pit (前后/两侧坑): 0.2 — 训练在窄平台上避免侧滑跌落
        "pit_double": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.2, pit_depth_range=(0.05, 1.0),
            double_pit=True, platform_width=2.0),
        # 单层 pit (周围 1m 深坑): 0.6 — 主体训练场景, 学习"跳坑/避坑"
        "pit_single": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.6, pit_depth_range=(0.05, 1.0),
            double_pit=False, platform_width=2.0),
        # 中央高 box: 0.2 — 训练下高台 (机器人在 box 上, 周围低)
        "box": terrain_gen.trimesh.mesh_terrains_cfg.MeshBoxTerrainCfg(
            proportion=0.2, box_height_range=(0.05, 1.0),
            double_box=False, platform_width=2.0),
    },
)

SCAN_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=30, num_cols=10,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=True,
    sub_terrains={
        # 共同特征: crawl 模式, 最难 clearance=0.15m (curriculum 从 0.5 → 0.15)
        # 4 种形状提供多样的视觉/几何挑战:
        # 1) 杆 (5×5cm): 细横杆, 视觉上小, 钻过时机短
        "hurdle_pole": MeshSquareHurdleTerrainCfg(
            proportion=0.25, hurdle_height_range=(0.15, 0.5),
            bar_thickness=0.05, bar_width=0.05,
            platform_width=2.0, mode="crawl"),
        # 2) 板子 (20cm 高 × 5cm 厚, 沿前进方向): 视觉上较大, 钻过快
        "hurdle_board": MeshSquareHurdleTerrainCfg(
            proportion=0.25, hurdle_height_range=(0.15, 0.5),
            bar_thickness=0.2, bar_width=0.05,
            platform_width=2.0, mode="crawl"),
        # 3) 薄墙 (20cm 高 × 15cm 厚): 钻行需保持低姿一段距离
        "hurdle_thin_wall": MeshSquareHurdleTerrainCfg(
            proportion=0.25, hurdle_height_range=(0.15, 0.5),
            bar_thickness=0.2, bar_width=0.15,
            platform_width=2.0, mode="crawl"),
        # 4) 厚墙 (20cm 高 × 50cm 厚, 最厚): 长时间低姿穿过隧道型障碍
        "hurdle_thick_wall": MeshSquareHurdleTerrainCfg(
            proportion=0.25, hurdle_height_range=(0.15, 0.5),
            bar_thickness=0.2, bar_width=0.5,
            platform_width=2.0, mode="crawl"),
    },
)

STEPPING_STONES_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=30, num_cols=10,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=True,
    sub_terrains={
        # 完全复用 baseline MOE_ROUGH_TERRAINS_CFG 的 stepping_stones 参数 (用户要求)
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0, stone_height_max=0.01,
            stone_width_range=(1.5, 1.5), stone_distance_range=(0.1, 0.8),
            holes_depth=-0.5, platform_width=2.0),
    },
)

RAIL_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=30, num_cols=10,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=True,
    sub_terrains={
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=1.0, rail_thickness_range=(0.05, 0.1),
            rail_height_range=(0.05, 0.4), platform_width=2.0),
    },
)

NOISE_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=30, num_cols=10,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.16),
            noise_step=0.02, border_width=0.25),
    },
)

GRID_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=30, num_cols=10,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=True,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45,
            grid_height_range=(0.05, 0.2), platform_width=2.0),
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
        "hurdle": MeshSquareHurdleTerrainCfg(
            proportion=3.0/4.0,
            hurdle_height_range=(0.4, 0.75),
            bar_thickness=0.25,
            bar_width=0.05,
            platform_width=2.0,
            mode="crawl",
        ),
        "plane": terrain_gen.trimesh.mesh_terrains_cfg.MeshPlaneTerrainCfg(
            proportion=1.0/4.0,
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
        "hurdle": MeshSquareHurdleTerrainCfg(
            proportion=1.0,
            hurdle_height_range=(0.65, 0.65),
            bar_thickness=0.2,
            bar_width=0.05,
            platform_width=2.0,
            mode="crawl",
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
        "hurdle": MeshSquareHurdleTerrainCfg(
            proportion=3.0/18,
            hurdle_height_range=(0.4, 0.65),
            bar_thickness=0.35,
            bar_width=0.05,
            platform_width=2.0,
            mode="crawl",
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

STAIR_TEST_TERRAINS_CFG = TerrainGeneratorCfg(
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
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,  # 2/18 的概率生成此地形
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    }
)

PRE_TRAIN_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
    }
)




ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


ELEMOE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
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
            proportion=2.0/18,  # 2/18 的概率生成此地形
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=3.0/18,
            step_height_range=(0.05, 0.23),
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
            proportion=3.0/18, rail_thickness_range=(0.05, 0.1), rail_height_range=(0.05, 0.35),platform_width=2.0
        ),
        "pit": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(0.05, 0.6), 
            double_pit=True,
            platform_width=2.0,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0/18, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0/18, noise_range=(0.02, 0.16), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
    },
)
ELEMOE_ROUGH_TERRAINS_CFG2 = TerrainGeneratorCfg(
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
        "pyramid_stairs": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=2.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "pyramid_stairs_inv": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(3.0, 3.0),  
            double_pit=False,
            platform_width=8.0,
        ),
        "stepping_stones": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "pit": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "boxes": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "random_rough": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "hf_pyramid_slope": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "hf_pyramid_slope_inv": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
    },
)


SCAN_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
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
            proportion=2.0/18,  # 2/18 的概率生成此地形
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=3.0/18,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=3.0/18, rail_thickness_range=(0.05, 0.1), rail_height_range=(0.05, 0.35),platform_width=2.0
        ),
        "hurdle": MeshSquareHurdleTerrainCfg(
            proportion=3.0/18,
            hurdle_height_range=(0.4, 0.65),
            bar_thickness=0.2,
            bar_width=0.05,
            platform_width=2.0,
            mode="crawl",
        ),
        "pit": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(0.05, 0.8), 
            double_pit=True,
            platform_width=2.0,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0/18, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0/18, noise_range=(0.02, 0.16), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=2.0, border_width=0.25
        ),
    },
)
SCAN_ROUGH_TERRAINS_CFG2 = TerrainGeneratorCfg(
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
        "pyramid_stairs": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=2.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "pyramid_stairs_inv": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(3.0, 3.0),  
            double_pit=False,
            platform_width=8.0,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "hurdle": MeshSquareHurdleTerrainCfg(
            proportion=3.0/18,
            hurdle_height_range=(0.4, 0.75),
            bar_thickness=0.25,
            bar_width=0.05,
            platform_width=2.0,
            mode="crawl",
        ),
        "pit": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=3.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "boxes": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "random_rough": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "hf_pyramid_slope": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "hf_pyramid_slope_inv": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
    },
)