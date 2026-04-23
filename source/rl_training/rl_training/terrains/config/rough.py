# rough.py (优化版：多老师蒸馏架构)

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg

from rl_training.terrains import MeshGapTerrainCfg, MeshSquareHurdleTerrainCfg

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

gap_cfg = MeshGapTerrainCfg(
    proportion=1.0, gap_width_range=(0.3, 0.8), platform_width=4.0, gap_depth=0.5
)

boxes_cfg = terrain_gen.MeshRandomGridTerrainCfg(
    proportion=1.0, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=4.0
)

# -- [极端挑战类]
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

box_cfg = terrain_gen.trimesh.mesh_terrains_cfg.MeshBoxTerrainCfg(
    proportion=1.0, box_height_range=(0.1, 0.4), platform_width=4.0, double_box=True
)

square_hurdle_cfg = MeshSquareHurdleTerrainCfg(
    proportion=1.0,
    hurdle_height_range=(0.25, 0.6),
    bar_thickness=0.2,
    platform_width=4.0,
    bar_width=0.05,
    mode="crawl",
)

# 新增：带孔网格 (对落足点精度要求极高)
grid_with_holes_cfg = terrain_gen.MeshRandomGridTerrainCfg(
    proportion=1.0, grid_width=0.6, grid_height_range=(0.0, 0.1), holes=True
)

# ==============================================================================
# 3. 老师训练专用地形配置 (Teacher-Specific Terrain Configs)
#    分组依据：运动模态 (Motor Modality)
# ==============================================================================

# [Teacher 1] 盲视基础专家 (Blind Base)
# 运动模态：本体感觉适应，小动作
# 适用地形：平地、粗糙面、缓坡
BASE_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=NUM_COLS, curriculum=True,
    sub_terrains={
        "flat": flat_cfg.replace(proportion=0.2),
        "random_rough": random_rough_cfg.replace(proportion=0.4),
        "slopes": pyramid_slope_cfg.replace(proportion=0.4),
    }
)

# [Teacher 2] 大动作跨越专家 (Stride & Leap)
# 运动模态：大幅抬腿、跳跃、精准踩踏 (需要高程+扫描感知)
# 适用地形：楼梯、反向楼梯、缝隙、跨栏(跳过)、高台(pit)
ELEVATION_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=10, curriculum=True,
    sub_terrains={
        "stairs": pyramid_stairs_cfg.replace(proportion=0.2),
        "inv_stairs": inverted_stairs_cfg.replace(proportion=0.15),
        "gaps": gap_cfg.replace(proportion=0.2),
        "rails": rails_cfg.replace(proportion=0.2),
        "pit": pit_cfg.replace(proportion=0.15),
        "random_rough": random_rough_cfg.replace(proportion=0.1),
    }
)

# [Teacher 3] 低姿钻越专家 (Crawl Under)
# 运动模态：压低重心、蹲伏通过 (需要前方扫描感知障碍物高度)
# 适用地形：hurdle (crawl mode) 多种参数变体
SCAN_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=10, curriculum=True,
    sub_terrains={
        "hurdle": square_hurdle_cfg.replace(
            proportion=0.25,
            hurdle_height_range=(0.3, 0.6),
            bar_thickness=0.3,
        ),
        "hurdle2": square_hurdle_cfg.replace(
            proportion=0.25,
            hurdle_height_range=(0.3, 0.6),
            bar_thickness=0.2,
        ),
        "hurdle3": square_hurdle_cfg.replace(
            proportion=0.25,
            hurdle_height_range=(0.3, 0.6),
            bar_thickness=0.1,
        ),
        "hurdle4": square_hurdle_cfg.replace(
            proportion=0.25,
            hurdle_height_range=(0.3, 0.6),
            bar_thickness=0.15,
        ),
    }
)

SCAN_TEACHER_TERRAINS_CFG2 = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=10, curriculum=True,
    sub_terrains={
        "hurdle": square_hurdle_cfg.replace(
            proportion=0.2,
            hurdle_height_range=(0.3, 0.65),
            bar_thickness=0.25,
        ),
        "hurdle2": square_hurdle_cfg.replace(
            proportion=0.2,
            hurdle_height_range=(0.25, 0.55),
            bar_thickness=0.2,
        ),
        "hurdle3": square_hurdle_cfg.replace(
            proportion=0.2,
            hurdle_height_range=(0.2, 0.5),
            bar_thickness=0.15,
        ),
        "hurdle4": square_hurdle_cfg.replace(
            proportion=0.2,
            hurdle_height_range=(0.15, 0.4),
            bar_thickness=0.1,
        ),
        "hurdle5": square_hurdle_cfg.replace(
            proportion=0.2,
            hurdle_height_range=(0.1, 0.3),
            bar_thickness=0.08,
        ),
    }
)

# [Teacher 4] 精准落足专家 (Precision Placement)
# 运动模态：每步踩准、避开孔洞
# 适用地形：带孔网格 (多种网格宽度)
PLACEMENT_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=NUM_COLS, curriculum=True,
    sub_terrains={
        "holes": grid_with_holes_cfg.replace(proportion=0.25, grid_width=0.35),
        "holes2": grid_with_holes_cfg.replace(proportion=0.25, grid_width=0.3),
        "holes3": grid_with_holes_cfg.replace(proportion=0.25, grid_width=0.26),
        "holes4": grid_with_holes_cfg.replace(proportion=0.25, grid_width=0.21),
    }
)

# [Teacher 5] 杂技恢复专家 (Acrobatic - Always Flat)
# 运动模态：推力恢复、翻倒自救
ACROBATIC_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=NUM_COLS, curriculum=False,
    sub_terrains={"flat": flat_cfg}
)

# ==============================================================================
# 4. 学生蒸馏使用的混合地形 (Student Distillation Mixture)
#    覆盖所有 5 个老师的地形，让学生在混合环境中学习所有运动模态
# ==============================================================================

# 简洁版：每种地形类型各 1 个代表，均匀分布
STUDENT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=NUM_COLS, curriculum=True,
    sub_terrains={
        # --- T1 盲视基础 (15%) ---
        "flat": flat_cfg.replace(proportion=0.05),
        "random_rough": random_rough_cfg.replace(proportion=0.05),
        "slopes": pyramid_slope_cfg.replace(proportion=0.05),
        # --- T2 大动作跨越 (50%) ---
        "stairs": pyramid_stairs_cfg.replace(proportion=0.10),
        "inv_stairs": inverted_stairs_cfg.replace(proportion=0.10),
        "gaps": gap_cfg.replace(proportion=0.10),
        "rails": rails_cfg.replace(proportion=0.10),
        "pit": pit_cfg.replace(proportion=0.10),
        # --- T3 低姿钻越 (10%) ---
        "hurdle": square_hurdle_cfg.replace(proportion=0.10),
        # --- T4 精准落足 (10%) ---
        "holes": grid_with_holes_cfg.replace(proportion=0.05, grid_width=0.3),
        "holes2": grid_with_holes_cfg.replace(proportion=0.05, grid_width=0.21),
        # --- 辅助鲁棒性 (5%) ---
        "boxes": boxes_cfg.replace(proportion=0.05),
    }
)

# 详细版：每种地形类型有多个参数变体，更丰富的训练多样性
STUDENT_TERRAINS_CFG2 = TerrainGeneratorCfg(
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
        # --- T1 盲视基础 (3/18) ---
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0/18, noise_range=(0.02, 0.16), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=4.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0/18, slope_range=(0.0, 0.55), platform_width=4.0, border_width=0.25
        ),
        # --- T2 大动作跨越 (8/18) ---
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0/18,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0/18,
            step_height_range=(0.05, 0.25),
            step_width=0.2,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0/18,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv2": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0/18,
            step_height_range=(0.05, 0.25),
            step_width=0.2,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        "gaps": MeshGapTerrainCfg(
            proportion=1.0/18, gap_width_range=(0.3, 0.8), platform_width=4.0, gap_depth=0.5
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=2.0/18, rail_thickness_range=(0.05, 0.1), rail_height_range=(0.05, 0.4), platform_width=4.0
        ),
        "pit": terrain_gen.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0/18,
            pit_depth_range=(0.05, 0.8),
            double_pit=True,
            platform_width=4.0,
        ),
        # --- T3 低姿钻越 (3/18) ---
        "hurdle": MeshSquareHurdleTerrainCfg(
            proportion=1.0/18,
            hurdle_height_range=(0.25, 0.6),
            bar_thickness=0.2,
            platform_width=4.0,
            bar_width=0.05,
            mode="crawl",
        ),
        "hurdle2": MeshSquareHurdleTerrainCfg(
            proportion=1.0/18,
            hurdle_height_range=(0.2, 0.5),
            bar_thickness=0.15,
            platform_width=4.0,
            bar_width=0.05,
            mode="crawl",
        ),
        "hurdle3": MeshSquareHurdleTerrainCfg(
            proportion=1.0/18,
            hurdle_height_range=(0.15, 0.4),
            bar_thickness=0.1,
            platform_width=4.0,
            bar_width=0.05,
            mode="crawl",
        ),
        # --- T4 精准落足 (3/18) ---
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
            holes_depth=-0.35,
            platform_width=4.0,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0/18, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=4.0
        ),
    },
)

# [Teacher 6] 高台攀爬专家 (Platform Climbing)
# 运动模态：大落差攀爬和降落
# Pit 地形：机器人从底部出生，学习向上攀爬 (训练上高台)
# Box 地形：机器人从顶部出生，学习向下降落 (训练下高台)
PLATFORM_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=10, curriculum=True,
    sub_terrains={
        "pit_shallow": pit_cfg.replace(
            proportion=0.2,
            pit_depth_range=(0.05, 0.8),
            double_pit=True,
        ),
        "pit_deep": pit_cfg.replace(
            proportion=0.15,
            pit_depth_range=(0.05, 0.8),
            double_pit=True,
        ),
        "pit_single": pit_cfg.replace(
            proportion=0.15,
            pit_depth_range=(0.05, 0.8),
            double_pit=False,
        ),
        "box_low": box_cfg.replace(
            proportion=0.15,
            box_height_range=(0.05, 0.8),
            double_box=False,
        ),
        "box_high": box_cfg.replace(
            proportion=0.15,
            box_height_range=(0.05, 0.8),
            double_box=True,
        ),
        "box_tall": box_cfg.replace(
            proportion=0.2,
            box_height_range=(0.05, 0.8),
            double_box=True,
        ),
    }
)

# [Teacher 7] 跨越沟壑专家 (Gap Crossing)
# 运动模态：跨越宽沟 (需要高程+扫描感知)
# 使用自定义 MeshGapTerrainCfg，gap_depth 参数确保沟底有实际地面 (非虚空)
# gap_width_range 控制沟壑宽度，难度越高沟越宽
GAP_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=10, curriculum=True,
    sub_terrains={
        "gap_shallow": MeshGapTerrainCfg(
            proportion=0.25,
            gap_width_range=(0.1, 0.8),
            platform_width=4.0,
            gap_depth=0.35,
        ),
        "gap_mid": MeshGapTerrainCfg(
            proportion=0.25,
            gap_width_range=(0.1, 0.8),
            platform_width=4.0,
            gap_depth=0.5,
        ),
        "gap_deep": MeshGapTerrainCfg(
            proportion=0.25,
            gap_width_range=(0.1, 0.8),
            platform_width=4.0,
            gap_depth=0.65,
        ),
        "gap_very_deep": MeshGapTerrainCfg(
            proportion=0.25,
            gap_width_range=(0.1, 0.8),
            platform_width=4.0,
            gap_depth=0.8,
        ),
    }
)

# [Teacher 8] 跨栏跳跃专家 (Rail Jumping)
# 运动模态：跳跃跨越栏杆 (需要高程+扫描感知前方障碍)
# 不同高度和厚度的栏杆组合，难度递增
RAIL_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TERRAIN_SIZE, border_width=20.0, num_rows=NUM_ROWS, num_cols=10, curriculum=True,
    sub_terrains={
        "rail_low": rails_cfg.replace(
            proportion=0.25,
            rail_thickness_range=(0.01, 0.06),
            rail_height_range=(0.05, 0.4),
            platform_width=4.0,
        ),
        "rail_mid": rails_cfg.replace(
            proportion=0.25,
            rail_thickness_range=(0.01, 0.06),
            rail_height_range=(0.05, 0.4),
            platform_width=4.0,
        ),
        "rail_high": rails_cfg.replace(
            proportion=0.25,
            rail_thickness_range=(0.01, 0.06),
            rail_height_range=(0.05, 0.4),
            platform_width=4.0,
        ),
        "rail_thick": rails_cfg.replace(
            proportion=0.25,
            rail_thickness_range=(0.01, 0.06),
            rail_height_range=(0.05, 0.4),
            platform_width=4.0,
        ),
    }
)

# 向后兼容别名
MOE_ROUGH_TERRAINS_CFG = STUDENT_TERRAINS_CFG
MOE_ROUGH_TERRAINS_CFG2 = STUDENT_TERRAINS_CFG2