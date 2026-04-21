# rough.py (优化版：多老师蒸馏架构)

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg

# ==============================================================================
# 1. 基础配置参数
# ==============================================================================
TERRAIN_SIZE = (12.0, 12.0)
NUM_ROWS = 30
NUM_COLS = 18

# ==============================================================================
# 2. 子地形定义库 (Sub-Terrain Definitions)
# ==============================================================================

# -- [基础类]
flat_cfg = terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)

random_rough_cfg = terrain_gen.HfRandomUniformTerrainCfg(
    proportion=1.0, noise_range=(0.02, 0.16), noise_step=0.02, border_width=0.25
)

pyramid_slope_cfg = terrain_gen.HfPyramidSlopedTerrainCfg(
    proportion=1.0, slope_range=(0.0, 0.4), platform_width=4.0, border_width=0.25
)

pyramid_stairs_cfg = terrain_gen.MeshPyramidStairsTerrainCfg(
    proportion=1.0, 
    step_height_range=(0.05, 0.25), 
    step_width=0.3, 
    platform_width=5.0,
    border_width=1.0
)

gap_cfg = terrain_gen.MeshGapTerrainCfg(
    proportion=1.0, gap_width_range=(0.3, 0.8), platform_width=4.0
)

boxes_cfg = terrain_gen.MeshRandomGridTerrainCfg(
    proportion=1.0, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=4.0
)

# -- [极端挑战类]
# 悬空圆环系列 (用于训练雷达扫描感知的跳跃/钻孔)
floating_ring_cfg = terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
    proportion=1.0, ring_width_range=(0.1, 0.5), ring_height_range=(0.4, 0.75), ring_thickness=0.2, platform_width=4.0
)

pit_cfg = terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
    proportion=1.0, pit_depth_range=(0.05, 0.8), double_pit=True, platform_width=4.0
)


inverted_stairs_cfg = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
    proportion=1.0, 
    step_height_range=(0.05, 0.25), 
    step_width=0.3, 
    platform_width=5.0,
    border_width=1.0
)

rails_cfg = terrain_gen.MeshRailsTerrainCfg(
    proportion=1.0, 
    rail_thickness_range=(0.05, 0.1),  # 栏杆的厚度（较薄，逼真模拟跨栏）
    rail_height_range=(0.2, 0.45),      # 栏杆的高度（根据机器人的极限跳跃能力调整）
    platform_width=4.0
)

# 假设你在地形配置文件中定义所有的子地形配置
square_hurdle_cfg = terrain_gen.MeshSquareHurdleTerrainCfg(
    proportion=1.0,                 # 在整个地形网格中，这种跨栏地形占据的比例 (例如 20%)
    hurdle_height_range=(0.25, 0.6), # 跨栏悬空高度随难度从 0.1m (10cm) 递增到 0.5m (50cm)
    bar_thickness=0.2,             # 横杆和立柱粗细设为 8cm
    platform_width=4.0,            # 内部活动空间边长设为 2m，给机器人足够的加速/助跑空间
    bar_width=0.05,
    mode="crawl",
)

# 新增：带孔网格 (对落足点精度要求极高)
grid_with_holes_cfg = terrain_gen.MeshRandomGridTerrainCfg(
    proportion=1.0, grid_width=0.6, grid_height_range=(0.0, 0.1), holes=True
)

# ==============================================================================
# 3. 老师训练专用地形配置 (Teacher-Specific Terrain Configs)
# ==============================================================================

# [Teacher 1] 盲视基础专家 (Blind Base)
# 适用地形：平地、粗糙面、缓坡
BASE_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=NUM_COLS, curriculum=True,
    sub_terrains={
        "flat": flat_cfg.replace(proportion=0.2),
        "random_rough": random_rough_cfg.replace(proportion=0.4),
        "slopes": pyramid_slope_cfg.replace(proportion=0.4),
    }
)

# [Teacher 2] 精准高程专家 (Precision Elevation)
# 适用地形：楼梯、大缝隙、方块
ELEVATION_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=10, curriculum=True,
    sub_terrains={
        "stairs": pyramid_stairs_cfg.replace(proportion=0.2),
        "inv_stairs": inverted_stairs_cfg.replace(proportion=0.2),
        "gaps": gap_cfg.replace(proportion=0.4),
        "flat": flat_cfg.replace(proportion=0.1),
        "random_rough": random_rough_cfg.replace(proportion=0.1),
    }
)

# [Teacher 3] 空间扫描老师 (Extreme Scan)
SCAN_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=10, curriculum=True,
    sub_terrains={
        "rails": rails_cfg.replace(proportion=0.3),   
        "rings": square_hurdle_cfg.replace(proportion=0.4),
        "pit": pit_cfg.replace(proportion=0.3),                  # 深坑
    }
)

SCAN_TEACHER_TERRAINS_CFG2 = TerrainGeneratorCfg(
    size=TERRAIN_SIZE,
    border_width=0.0,
    num_rows=NUM_ROWS,
    num_cols=10,
    curriculum=True,
    sub_terrains={
        "rails": rails_cfg.replace(proportion=0.3),
        "rings": pit_cfg.replace(
            proportion=0.1,      
            pit_depth_range=(3.0, 3.0),  
            double_pit=False,
            platform_width=8.0,
        ),
        "rings2": pit_cfg.replace(
            proportion=0.1,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "rings3": pit_cfg.replace(
            proportion=0.1,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "pit": pit_cfg.replace(
            proportion=0.3,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
        "flat": pit_cfg.replace(
            proportion=0.1,      
            pit_depth_range=(3.0, 3.0), 
            double_pit=False,
            platform_width=8.0,
        ),
    }
)


# [Teacher 4] 极限落足专家 (Precision Placement - 代替原来的 Wave)
# 适用地形：带孔网格、随机障碍物
PLACEMENT_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=NUM_COLS, curriculum=True,
    sub_terrains={
        "holes": grid_with_holes_cfg.replace(proportion=0.25, grid_width=0.35),
        # 0.5 会被 8.0 完美整除导致边界为0，改为 0.55
        "holes2": grid_with_holes_cfg.replace(proportion=0.25, grid_width=0.3),
        # 0.4 会被 8.0 完美整除导致边界为0，改为 0.45
        "holes3": grid_with_holes_cfg.replace(proportion=0.25, grid_width=0.26),
        # 0.3 不能被 8.0 整除 (8.0/0.3=26.6)，所以没问题，也可以改成 0.35
        "holes4": grid_with_holes_cfg.replace(proportion=0.25, grid_width=0.21),
    }
)

# [Teacher 5] 杂技专家 (Acrobatic - Always Flat)
ACROBATIC_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=NUM_COLS, curriculum=False,
    sub_terrains={"flat": flat_cfg}
)

# ==============================================================================
# 4. 最终蒸馏使用的混合地形 (MOE Mixture)
# ==============================================================================
MOE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=NUM_COLS, curriculum=True,
    sub_terrains={
        "flat": flat_cfg.replace(proportion=0.1),
        "random_rough": random_rough_cfg.replace(proportion=0.1),
        "slopes": pyramid_slope_cfg.replace(proportion=0.1),
        "stairs": pyramid_stairs_cfg.replace(proportion=0.1),
        "gaps": gap_cfg.replace(proportion=0.1),
        "boxes": boxes_cfg.replace(proportion=0.1),
        "pit": pit_cfg.replace(proportion=0.1),
        "rings": floating_ring_cfg.replace(proportion=0.1),
        "inverted": inverted_stairs_cfg.replace(proportion=0.1),
        "rails": rails_cfg.replace(proportion=0.1),  # <--- 新增：跨栏出现在混合环境中
    }
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
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0/18,  # 2/18 的概率生成此地形
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0/18,  # 2/18 的概率生成此地形
            step_height_range=(0.05, 0.25),
            step_width=0.2,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0/18,
            step_height_range=(0.05, 0.25),
            step_width=0.4,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv2": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0/18,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv3": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0/18,
            step_height_range=(0.05, 0.25),
            step_width=0.2,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0/18,
            stone_height_max=0.01,         
            stone_width_range=(1.5, 1.5), 
            stone_distance_range=(0.1, 0.8), 
            holes_depth=-0.65,
            platform_width=4.0,
        ),
        "stepping_stones2": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0/18,
            stone_height_max=0.01,         
            stone_width_range=(1.5, 1.5), 
            stone_distance_range=(0.1, 0.8), 
            holes_depth=-0.5,
            platform_width=4.0,
        ),
        "stepping_stones3": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0/18,
            stone_height_max=0.01,         
            stone_width_range=(1.5, 1.5), 
            stone_distance_range=(0.1, 0.8), 
            holes_depth=-0.35,
            platform_width=4.0,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=2.0/18, rail_thickness_range=(0.05, 0.1), rail_height_range=(0.05, 0.4),platform_width=4.0
        ),
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=1.0/18,                         
            ring_width_range=(0.1, 0.5),             
            ring_height_range=(0.4, 0.75),           
            ring_thickness=0.4,                 
            platform_width=4.0, 
        ),
        "floating_ring2": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=1.0/18,                         
            ring_width_range=(0.1, 0.5),             
            ring_height_range=(0.4, 0.75),           
            ring_thickness=0.25,                 
            platform_width=4.0, 
        ),
        "floating_ring3": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=1.0/18,                         
            ring_width_range=(0.1, 0.5),             
            ring_height_range=(0.4, 0.75),           
            ring_thickness=0.1,                 
            platform_width=4.0, 
        ),
        "pit": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,      
            pit_depth_range=(0.05, 0.8), 
            double_pit=True,
            platform_width=4.0,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0/18, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=4.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0/18, noise_range=(0.02, 0.16), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=4.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=4.0, border_width=0.25
        ),
    },
)