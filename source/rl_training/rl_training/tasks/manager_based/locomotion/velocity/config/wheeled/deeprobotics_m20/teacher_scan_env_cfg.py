from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg
from rl_training.terrains.config.rough import SCAN_TEACHER_TERRAINS_CFG, SCAN_TEACHER_TERRAINS_CFG2
import math
import torch # Added torch for depth calculation
import torch.nn.functional as F
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.sensors import RayCasterCfg, patterns, RayCasterCameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    MySceneCfg, # Import base scene config
)
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import MultiMeshRayCasterCfg
##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_CFG  # isort: skip
from rl_training.terrains.config.rough import *
from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import TerrainAwareVelocityCommandCfg
from isaaclab.sensors import ContactSensorCfg
@configclass
class ScanRewardsCfg(RewardsCfg):
    """空间扫描老师的奖励函数"""
    
    # =========================================================
    # 1. 核心机制：跨栏专属碰撞惩罚
    # =========================================================
    # 监听 obstacle_sensor。只要机器人的任何 link (包含轮子) 
    # 碰到了 terrain2 (即跨栏)，就给予严厉惩罚！
    rails_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("obstacle_sensor", body_names=".*"), # 监听所有 link
            "threshold": 2.0 # 接触力大于 1N 触发惩罚
        }
    )
    
    # =========================================================
    # 2. 释放部分平稳性限制 (为了越野和跳跃)
    # =========================================================
    # 为了鼓励腾空跨栏，放宽 Z轴速度和 Pitch 角速度的惩罚
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05) 
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01) 
    dof_pos_limits = None # 跨栏可能需要极端的关节角度
    
    # 基础生存惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class DeeproboticsM20TeacherScanEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    """[Teacher 3] 空间扫描专家环境配置"""
    
    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------
        # 1. 双地形叠加配置
        # ---------------------------------------------------------
        # 主地形 (Pit, Rings 等)
        self.scene.terrain2 = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=SCAN_TEACHER_TERRAINS_CFG,
            max_init_terrain_level=1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/obstacles",
            terrain_type="generator",
            terrain_generator=SCAN_TEACHER_TERRAINS_CFG2,
            max_init_terrain_level=1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        FRONT_LIDAR_POS = (0.32028, 0.0, -0.013)
        REAR_LIDAR_POS = (-0.32028, 0.0, -0.013)
        

        down_angles_deg = [-25.0, -15.0, -5.0, 5.0, 15.0, 25.0]

        SCAN_PATTERN = patterns.GridPatternCfg(resolution=0.05, size=[0.0, 1.0])
        SCAN_MESHES = ["/World/ground", "/World/obstacles"]  # 扫描地形1和地形2，捕捉跨栏和坑洞信息

        for i, angle_deg in enumerate(down_angles_deg):
            
            # 前向雷达
            fwd_pitch_deg = -(90.0 - angle_deg)
            fwd_half_rad = math.radians(fwd_pitch_deg) / 2.0
            fwd_rot = (math.cos(fwd_half_rad), 0.0, math.sin(fwd_half_rad), 0.0)
            
            fwd_sensor = MultiMeshRayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base_link",
                offset=MultiMeshRayCasterCfg.OffsetCfg(pos=FRONT_LIDAR_POS, rot=fwd_rot),
                ray_alignment="base", 
                pattern_cfg=SCAN_PATTERN, 
                max_distance=5.0, # 修改为 5.0m
                debug_vis=False, 
                reference_meshes=True,
                mesh_prim_paths=SCAN_MESHES,
            )
            fwd_sensor.update_period = 0.1
            setattr(self.scene, f"forward_scanner_layer{i}", fwd_sensor)

            # 后向雷达
            bwd_pitch_deg = (90.0 - angle_deg)
            bwd_half_rad = math.radians(bwd_pitch_deg) / 2.0
            bwd_rot = (math.cos(bwd_half_rad), 0.0, math.sin(bwd_half_rad), 0.0)
            
            bwd_sensor = MultiMeshRayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base_link",
                offset=MultiMeshRayCasterCfg.OffsetCfg(pos=REAR_LIDAR_POS, rot=bwd_rot),
                ray_alignment="base", 
                pattern_cfg=SCAN_PATTERN, 
                max_distance=5.0, # 修改为 5.0m
                debug_vis=False, 
                reference_meshes=True,
                mesh_prim_paths=SCAN_MESHES,
            )
            bwd_sensor.update_period = 0.1
            setattr(self.scene, f"backward_scanner_layer{i}", bwd_sensor)
        # ---------------------------------------------------------
        # 2. [新增] 惩罚地形专属传感器
        # ---------------------------------------------------------
        self.scene.obstacle_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",             # <--- 改为大写 R
            filter_prim_paths_expr=["/World/obstacles/.*"],  
            update_period=0.02,
            debug_vis=True,
            force_threshold=2.0,  # 只要接触力大于 1N 就触发惩罚
        )

        # ---------------------------------------------------------
        # 3. 强制向前速度与参数替换
        # ---------------------------------------------------------
        # 强制向前跑，否则机器人会停在栏杆前不敢动
        if self.commands.base_velocity is not None:
            self.commands.base_velocity.ranges.lin_vel_x = (1.5, 3.0) 
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

        # 替换奖励
        self.rewards = ScanRewardsCfg()
        if hasattr(self, "disable_zero_weight_rewards"):
            self.disable_zero_weight_rewards()
            
        # 设置更长的单局时间让它跑完场地
        self.episode_length_s = 20.0