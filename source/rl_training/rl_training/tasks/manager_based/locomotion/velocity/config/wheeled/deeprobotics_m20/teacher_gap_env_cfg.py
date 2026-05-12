from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg, DeeproboticsM20RewardsCfg
from rl_training.terrains.config.rough import GAP_TEACHER_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import math
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class GapRewardsCfg(DeeproboticsM20RewardsCfg):
    """T7 跨越沟壑专家的奖励函数"""

    # 放宽 Z 轴速度惩罚：跨越沟壑时会有较大垂直运动
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    # 放宽 Roll/Pitch 惩罚
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)

    # 鼓励长腾空时间：跨沟需要大步跨越. threshold 0.3 → 0.4 鼓励 fully-committed jump
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_curriculum,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "threshold": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel"),
        },
    )

    # 严惩 base/hipx/hipy 非轮接触：避免用身体刮蹭沟壑边缘
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_hipx", ".*_hipy"]),
            "threshold": 1.0,
        }
    )

    # knee 单独轻惩 (-0.1, vs 主项 -0.3): knee 蹭石头边缘当杠杆是合理姿态; 与 PlatformRewardsCfg 保持一致
    undesired_contacts_knee = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_knee"]),
            "threshold": 1.0,
        }
    )
@configclass
class DeeproboticsM20TeacherGapEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    """[Teacher 7] 跨越沟壑专家环境配置

    运动模态：跨越宽沟
    地形：SteppingStones (不同沟壑宽度和深度组合)
    设计说明：使用 SteppingStones 替代 MeshGap，因为后者沟底直通虚空不符合现实物理。
    SteppingStones 的 holes_depth 参数确保沟底有实际地面。
    """

    def __post_init__(self):
        super().__post_init__()

        # 1. 地形
        self.scene.terrain.terrain_generator = GAP_TEACHER_TERRAINS_CFG

        # 2. 速度指令 (2.5D 闭环纠偏)
        if self.commands.base_velocity is not None:
            self.commands.base_velocity.rel_heading_envs = 0.85
            self.commands.base_velocity.heading_command = True
            self.commands.base_velocity.heading_control_stiffness = 1.0
            self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
            self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
            self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
            self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 3. 奖励
        self.rewards = GapRewardsCfg()
        # ---------------------------------------------------------------------------
        # Gap regression fix (vs 2026-04-02 video baseline f22c8fc):
        # 4-02 视频时能训出来 gap, 但当时 reward landscape 没有下面这些"反跳跃压力"
        # 项. 后续版本逐步加上让通才/其它专才更稳, 但对 gap (跨越类) 是直接反激励.
        # 在这里把它们归零, 恢复 4-02 的"敢跳"reward 形状.
        #   - is_terminated:        0 (was -100)  摔倒不毒打, 否则 policy 永远不敢跳
        #   - base_roll_l2:         0 (was -10)   跳跃落地必然 roll, -10 = 严罚跳跃
        #   - feet_height_body:     0 (was -0.1)  蓄力/腾空时 body 偏离站姿是必须的
        # track_*_pre_exp 软目标 (4-02 = 0.5 + 1.5) 暂不改, 先观察上述三项效果.
        # ---------------------------------------------------------------------------
        self.rewards.is_terminated.weight = 0      # 4-02 video baseline
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_roll_l2.weight = 0       # 4-02 video baseline (项不存在)
        self.rewards.base_height_l2.weight = -0.0
        self.rewards.base_height_l2.params["target_height"] = 0.5
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -4e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -1e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.hipx_joint_pos_penalty.weight = -0.5
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.hipy_joint_pos_penalty.weight = -0.25
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -0.1
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = 0.0  # sym_loss 已替代
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        self.rewards.action_mirror.weight = 0.0
        self.rewards.action_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        # pre_exp 软目标先保持 0 (用户决定先观察 is_terminated + base_roll +
        # feet_height_body 三项归零的效果, pre_exp 后续视情况再说)
        self.rewards.track_lin_vel_xy_pre_exp.weight = 0
        self.rewards.track_ang_vel_z_pre_exp.weight = 0

        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.3
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0     # was -0.1; 4-02 video baseline (项不存在)
        self.rewards.feet_height_body.params["target_height"] = -0.48  # 默认站姿实测 -0.44
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        self.rewards.upward.weight = 0  # 跨大 hole 时身体会 pitch, "保持竖直"是反向信号 (was 0.08)
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp_curriculum
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_exp_curriculum
        self.rewards.joint_mirror_lr.weight = 0.0

        # 4. 终止条件
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.bad_orientation_2 = None

        # 5. 课程学习
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.1, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.5, 1.0)

        # 6. 随机化 (yaw 固定为 0)
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.0, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        # friction/restitution 沿用父类 [0.4, 1.4] (sim2real 拓宽); 删除此处 [0.6, 1.2] override

        if hasattr(self, "disable_zero_weight_rewards"):
            self.disable_zero_weight_rewards()

        self.episode_length_s = 20.0
