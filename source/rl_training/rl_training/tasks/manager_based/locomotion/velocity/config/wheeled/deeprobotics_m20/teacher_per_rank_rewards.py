# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Per-rank reward overrides ported from main 的 teacher_platform_env_cfg / teacher_scan_env_cfg.

当前分支用 per-rank terrain dispatch (train_moe.py), 8 个 rank 共享一个 env_cfg.
这里提供两个 helper, 在 train_moe.py 解析完 env_cfg 之后按 terrain 身份调用:
  - apply_platform_rewards(env_cfg): rank=PLATFORM 用 main 的高台攀爬 reward
  - apply_scan_rewards(env_cfg):     rank=SCAN     用 main 的钻栏 reward

只搬 reward, 不动 terrain / command / curriculum / events (用户明确要求).

注意时序: parse_env_cfg 已经跑过 __post_init__ 并调用过 disable_zero_weight_rewards
(weight 0 的 term 已被设为 None). 所以 helper 必须先用 fresh cfg 实例完整重建
env_cfg.rewards, 再逐项 tune, 最后重新调用 disable_zero_weight_rewards.
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .moe_teacher_env_cfg import DeeproboticsM20RewardsCfg

# 对角镜像 joint 组 (main platform/scan 共用)
_MIRROR_JOINTS_DIAG = [
    ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
    ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
]


@configclass
class PlatformRewardsCfg(DeeproboticsM20RewardsCfg):
    """T6 高台攀爬专家奖励 (port main PlatformRewardsCfg)."""

    # 放宽 Z 轴速度惩罚: 攀爬高台必然产生较大垂直速度
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.03)
    # 放宽 Roll/Pitch 惩罚: 上下高台允许身体倾斜
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    # 鼓励长腾空时间: 攀爬高台需大幅抬腿
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_curriculum,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "threshold": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel"),
        },
    )
    # 惩罚 base_link / hipx / hipy 接触: 强制用 wheel 接触台面, 抑制"半身在台拖着另一半"捷径
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_hipx", ".*_hipy"]),
            "threshold": 1.0,
        },
    )
    # knee 接触惩罚单独降到 -0.1: 爬 30-60cm 高台时膝盖蹭平台边缘当杠杆是合理姿态
    undesired_contacts_knee = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_knee"]),
            "threshold": 1.0,
        },
    )


def apply_platform_rewards(env_cfg) -> None:
    """把 main 的 PLATFORM (高台攀爬) reward 应用到 env_cfg.

    端口自 main DeeproboticsM20TeacherPlatformEnvCfg.__post_init__ 的 reward 部分。
    只动 env_cfg.rewards, 不碰 terrain / command / curriculum / events。
    """
    r = PlatformRewardsCfg()
    env_cfg.rewards = r

    r.is_terminated.weight = -100  # 摔/撞惩罚

    r.flat_orientation_l2.weight = 0
    r.base_roll_l2.weight = -10.0
    r.base_height_l2.weight = 0
    r.base_height_l2.params["target_height"] = 0.5
    r.base_height_l2.params["asset_cfg"].body_names = [env_cfg.base_link_name]
    r.body_lin_acc_l2.weight = 0
    r.body_lin_acc_l2.params["asset_cfg"].body_names = [env_cfg.base_link_name]

    r.joint_torques_l2.weight = -2.5e-5
    r.joint_torques_l2.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.joint_torques_wheel_l2.weight = 0
    r.joint_torques_wheel_l2.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names
    r.joint_vel_l2.weight = 0
    r.joint_vel_l2.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.joint_vel_wheel_l2.weight = 0
    r.joint_vel_wheel_l2.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names
    r.joint_acc_l2.weight = -4e-7
    r.joint_acc_l2.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.joint_acc_wheel_l2.weight = -1e-7
    r.joint_acc_wheel_l2.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names
    r.joint_pos_limits.weight = -5.0
    r.joint_pos_limits.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.joint_vel_limits.weight = 0
    r.joint_vel_limits.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names
    r.joint_power.weight = -2e-5
    r.joint_power.params["asset_cfg"].joint_names = env_cfg.leg_joint_names

    r.stand_still.weight = -2.0
    r.stand_still.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.hipx_joint_pos_penalty.weight = -0.5
    r.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = env_cfg.hipx_joint_names
    r.hipy_joint_pos_penalty.weight = -0.25
    r.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = env_cfg.hipy_joint_names
    r.knee_joint_pos_penalty.weight = -0.1
    r.knee_joint_pos_penalty.params["asset_cfg"].joint_names = env_cfg.knee_joint_names
    r.wheel_vel_penalty.weight = 0
    r.wheel_vel_penalty.params["sensor_cfg"].body_names = env_cfg.foot_link_name
    r.wheel_vel_penalty.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names

    r.joint_mirror.weight = 0.0  # sym_loss 已替代
    r.joint_mirror.params["mirror_joints"] = _MIRROR_JOINTS_DIAG
    r.action_mirror.weight = 0.0
    r.action_mirror.params["mirror_joints"] = _MIRROR_JOINTS_DIAG
    r.joint_mirror_lr.weight = 0.0
    r.action_rate_l2.weight = -0.01
    r.contact_forces.weight = -1.5e-4
    r.contact_forces.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]

    r.track_lin_vel_xy_exp.weight = 4.0
    r.track_ang_vel_z_exp.weight = 3.0
    r.track_lin_vel_xy_pre_exp.weight = 0
    r.track_ang_vel_z_pre_exp.weight = 0
    r.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp_curriculum
    r.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_exp_curriculum

    r.feet_contact.weight = 0
    r.feet_contact.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_contact_without_cmd.weight = 0.1
    r.feet_contact_without_cmd.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_stumble.weight = 0
    r.feet_stumble.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_slide.weight = 0
    r.feet_slide.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_slide.params["asset_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_height.weight = 0
    r.feet_height.params["target_height"] = 0.3
    r.feet_height.params["asset_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_height_body.weight = -0.5  # 抑制蹲走
    r.feet_height_body.params["target_height"] = -0.48
    r.feet_height_body.params["asset_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_gait.weight = 0
    r.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
    r.upward.weight = 0  # platform: 攀爬必然 pitch, "保持竖直"是反向信号

    env_cfg.disable_zero_weight_rewards()


def apply_scan_rewards(env_cfg) -> None:
    """把 main 的 SCAN (钻栏) reward 应用到 env_cfg.

    端口自 main DeeproboticsM20TeacherScanEnvCfg.__post_init__ 的 reward 部分。
    main 的 ScanRewardsCfg 只是 DeeproboticsM20RewardsCfg 的空壳子类, 故直接用基类重建。
    """
    r = DeeproboticsM20RewardsCfg()
    env_cfg.rewards = r

    r.is_terminated.weight = -100  # 摔/撞惩罚

    r.lin_vel_z_l2.weight = -2.0
    r.ang_vel_xy_l2.weight = -0.05
    r.flat_orientation_l2.weight = 0
    r.base_roll_l2.weight = -10.0
    r.base_height_l2.weight = -0.5
    r.base_height_l2.params["target_height"] = 0.5
    r.base_height_l2.params["asset_cfg"].body_names = [env_cfg.base_link_name]
    r.body_lin_acc_l2.weight = 0
    r.body_lin_acc_l2.params["asset_cfg"].body_names = [env_cfg.base_link_name]

    r.joint_torques_l2.weight = -2.5e-5
    r.joint_torques_l2.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.joint_torques_wheel_l2.weight = 0
    r.joint_torques_wheel_l2.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names
    r.joint_vel_l2.weight = 0
    r.joint_vel_l2.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.joint_vel_wheel_l2.weight = 0
    r.joint_vel_wheel_l2.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names
    r.joint_acc_l2.weight = -4e-7
    r.joint_acc_l2.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.joint_acc_wheel_l2.weight = -1e-7
    r.joint_acc_wheel_l2.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names
    r.joint_pos_limits.weight = -5.0
    r.joint_pos_limits.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.joint_vel_limits.weight = 0
    r.joint_vel_limits.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names
    r.joint_power.weight = -2e-5
    r.joint_power.params["asset_cfg"].joint_names = env_cfg.leg_joint_names

    r.stand_still.weight = -2.0
    r.stand_still.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    r.hipx_joint_pos_penalty.weight = -0.6
    r.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = env_cfg.hipx_joint_names
    r.hipy_joint_pos_penalty.weight = -0.3
    r.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = env_cfg.hipy_joint_names
    r.knee_joint_pos_penalty.weight = -0.1
    r.knee_joint_pos_penalty.params["asset_cfg"].joint_names = env_cfg.knee_joint_names
    r.wheel_vel_penalty.weight = 0
    r.wheel_vel_penalty.params["sensor_cfg"].body_names = env_cfg.foot_link_name
    r.wheel_vel_penalty.params["asset_cfg"].joint_names = env_cfg.wheel_joint_names

    r.joint_mirror.weight = 0.0  # sym_loss 已替代
    r.joint_mirror.params["mirror_joints"] = _MIRROR_JOINTS_DIAG
    r.joint_mirror_lr.weight = 0.0
    r.action_mirror.weight = 0.0
    r.action_mirror.params["mirror_joints"] = _MIRROR_JOINTS_DIAG
    r.action_rate_l2.weight = -0.01

    r.undesired_contacts.weight = -0.1
    r.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{env_cfg.foot_link_name}).*"]
    r.contact_forces.weight = -1.5e-4
    r.contact_forces.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]

    r.track_lin_vel_xy_exp.weight = 4.0
    r.track_ang_vel_z_exp.weight = 3.0
    r.track_lin_vel_xy_pre_exp.weight = 0
    r.track_ang_vel_z_pre_exp.weight = 0
    r.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp_curriculum
    r.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_exp_curriculum

    r.feet_air_time.weight = 0.0
    r.feet_air_time.params["threshold"] = 0.25
    r.feet_air_time.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_air_time_long.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_contact.weight = 0
    r.feet_contact.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_contact_without_cmd.weight = 0.1
    r.feet_contact_without_cmd.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_stumble.weight = 0
    r.feet_stumble.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_slide.weight = 0
    r.feet_slide.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_slide.params["asset_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_height.weight = 0
    r.feet_height.params["target_height"] = 0.3
    r.feet_height.params["asset_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_height_body.weight = 0  # scan: 蹲伏过栏杆是任务本身, 不抑制
    r.feet_height_body.params["target_height"] = -0.48
    r.feet_height_body.params["asset_cfg"].body_names = [env_cfg.foot_link_name]
    r.feet_gait.weight = 0
    r.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
    r.upward.weight = 0.08

    env_cfg.disable_zero_weight_rewards()


def apply_stair_slope_rewards(env_cfg) -> None:
    """STAIR_SLOPE rank: 楼梯/斜坡专项 reward 微调 (T3 tier).

    诊断 "双脚同台阶才迈" 现象:
      joint_mirror = (q[FL]-q[HR])^2 + (q[FR]-q[HL])^2 (同相约束, 见 mdp/rewards.py:259).
      台阶上 FL/HR 必处于不同高度的台阶 → 关节角必不同 → mirror 罚很大,
      策略学到的最优解就是 "先聚脚再迈步" 把 mirror 压到 0. 几何根因.
      hipy/knee pos penalty 同时限制单腿大幅抬升 (台阶 riser ≈ 17cm 时关键).

    本 helper 调 7 个权重:
      (原 2 项)
      - lin_vel_z_l2:           -2.0  → -0.05    (放宽 40×, 允许台阶起跳的 Z 速度)
      - undesired_contacts:     -0.1  → -0.5     (强化 5×, 不许胫骨蹭台阶)
      (新增 T3)
      - joint_mirror:           -0.05 → -0.01    (放宽 5×, 允许 trot 单步上一阶)
      - hipy_joint_pos_penalty: -0.3  → -0.1     (放宽 3×, hip pitch 允许大幅前甩)
      - knee_joint_pos_penalty: -0.3  → -0.1     (放宽 3×, knee 允许大幅收缩)
      - joint_acc_l2:           -4e-7 → -1.5e-7  (放宽 2.7×, swing 加速更快)
      - action_rate_l2:         -0.01 → -0.005   (放宽 2×, lift-step-place 序列切换)

    目标 gait: 对角 trot 爬楼 (FL+HR 同时跨到 N+1, 下一步 FR+HL 同时跨到 N+2),
    每步上一阶, 而不是 "前脚先上, 后脚跟到同一阶, 再前脚上下一阶" 的 bound-shuffle.
    """
    r = env_cfg.rewards
    # 原 2 项
    r.lin_vel_z_l2.weight = -0.05
    r.undesired_contacts.weight = -0.5
    # T3 追加: 解几何根因 + 放开单腿大幅抬升 + 放开 swing 加速
    r.joint_mirror.weight = -0.01
    r.hipy_joint_pos_penalty.weight = -0.1
    r.knee_joint_pos_penalty.weight = -0.1
    r.joint_acc_l2.weight = -1.5e-7
    r.action_rate_l2.weight = -0.005
