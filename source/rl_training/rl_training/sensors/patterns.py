"""Hemispherical lidar pattern aligned with sdk_deploy/mujoco/lidar_to_scan.py.

匹配 Robosense Airy 真机 + sim2sim 端到端的几何约定:
- 所有 ray 起源于 (0,0,0) (sensor local 光学中心)
- ray 方向由 (polar, azimuth) 球面角决定, 其中:
    polar    = 与 sensor +X (boresight) 之间的夹角, ∈ [0, polar_max] (默认 0~80°)
    azimuth  = 绕 boresight 一圈的旋转角, ∈ [-π, π].
               az=0 → +Z (boresight 上方)
               az=π/2 → +Y (boresight 左方)
               az=π  → -Z (boresight 下方)
               az=-π/2 → -Y (boresight 右方)
- ray_dir = (cos(polar), sin(polar)*sin(az), sin(polar)*cos(az))

flat 顺序: polar-major, azimuth-minor (匹配 sdk_deploy 的
`flat_idx = polar_idx * NUM_AZ + azimuth_idx`).

后向 sensor 整体绕 Z 转 180° 即可复用同一 pattern (sensor +X 变 -X).

对齐参数 (sdk_deploy/mujoco/lidar_to_scan.py):
  NUM_POLAR=16, NUM_AZ=31, POLAR_MAX=80°, AZ range [-π, π].
  Per direction: 16 × 31 = 496 ray. Both directions: 992 ray.
"""

from __future__ import annotations

from typing import Callable

import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass


def hemispherical_arc_pattern(cfg: "HemisphericalArcPatternCfg", device: str):
    """生成 (num_polar * num_azimuth, 3) 的 ray_starts (全零) + ray_directions.

    flatten 顺序: polar-major, azimuth-minor (匹配 sdk_deploy 的 bin 公式).
    第 k 条 ray 对应 (polar_idx, azim_idx), k = polar_idx * num_azimuth + azim_idx.

    Returns:
        ray_starts: (N, 3) 全零 — 光学中心 in sensor frame
        ray_directions: (N, 3) 单位向量
    """
    if cfg.num_polar < 1:
        raise ValueError(f"num_polar must be >= 1, got {cfg.num_polar}")
    if cfg.num_azimuth < 1:
        raise ValueError(f"num_azimuth must be >= 1, got {cfg.num_azimuth}")
    if cfg.polar_max_deg <= 0 or cfg.polar_max_deg > 90:
        raise ValueError(f"polar_max_deg must be in (0, 90], got {cfg.polar_max_deg}")

    # polar ∈ [0, polar_max_deg], azim ∈ azimuth_range_deg
    polar_deg = torch.linspace(0.0, cfg.polar_max_deg, cfg.num_polar,
                               device=device, dtype=torch.float32)
    azim_deg = torch.linspace(cfg.azimuth_range_deg[0], cfg.azimuth_range_deg[1],
                              cfg.num_azimuth, device=device, dtype=torch.float32)
    polar = torch.deg2rad(polar_deg)
    azim = torch.deg2rad(azim_deg)

    # (P, A) 网格. indexing='ij' → polar 外, azim 内. flat 顺序 = polar_idx * num_azim + azim_idx.
    P, A = torch.meshgrid(polar, azim, indexing="ij")
    sin_p = torch.sin(P)
    cos_p = torch.cos(P)
    # mujoco 约定: x_b = boresight 沿 +X, perp = sqrt(y² + z²),
    # azim = atan2(y, z) → 当 az=0 时 y=0, z=perp_mag > 0 (上方).
    # 反向计算: y = perp * sin(az), z = perp * cos(az), perp = sin(polar), x = cos(polar)
    dx = cos_p
    dy = sin_p * torch.sin(A)
    dz = sin_p * torch.cos(A)
    ray_directions = torch.stack([dx.flatten(), dy.flatten(), dz.flatten()], dim=-1)
    n_rays = ray_directions.shape[0]
    ray_starts = torch.zeros((n_rays, 3), device=device, dtype=torch.float32)
    return ray_starts, ray_directions


@configclass
class HemisphericalArcPatternCfg(PatternBaseCfg):
    """Hemispherical lidar pattern matching Robosense Airy / sdk_deploy mujoco.

    Convention (cone-from-boresight, see sdk_deploy/mujoco/lidar_to_scan.py):
      - polar:    angle from sensor +X (boresight), 0 → boresight, polar_max → cone edge
      - azimuth:  rotation around boresight in Y-Z plane (az=atan2(y, z) convention)
      - flatten order: polar-major (polar_idx * num_azim + azim_idx)

    Defaults match sdk_deploy/mujoco (NUM_POLAR=16, NUM_AZ=31, polar_max=80°).
    """

    func: Callable = hemispherical_arc_pattern

    num_polar: int = 16
    """Number of polar bins (uniformly spaced from 0 to polar_max_deg)."""

    num_azimuth: int = 31
    """Number of azimuth bins (uniformly spaced over azimuth_range_deg)."""

    polar_max_deg: float = 80.0
    """Polar angle upper bound (deg). 真 Airy 在 polar>84° 密度急剧下降, 80° 是 sim2real 实测对齐值."""

    azimuth_range_deg: tuple[float, float] = (-180.0, 180.0)
    """Azimuth range (deg). 默认 ±180° = 全圆绕 boresight, 跟 mujoco lidar_to_scan 对齐."""


# ================================================================================
# Single-pitch horizontal arc pattern
# ================================================================================
# 用法: 每个 sensor 一个固定 pitch, 21 条 ray 从光学中心辐射, varying azimuth.
# 跟 baseline 12-sensor 架构兼容: 6 个 sensor per direction, 每个不同 pitch.
# 不同 sensor 之间 boresight 都横向前向, 由 pattern 自己 bake 进 pitch.
#
# 几何 (sensor frame, +X 前 +Y 左 +Z 上):
#   ray_dir = (cos(pitch)*cos(azim), cos(pitch)*sin(azim), sin(pitch))
# 这是 elevation/azimuth (球面经纬度) 约定, NOT mujoco 的 polar/azim-around-boresight.
# 区别: 这里 pitch 是从水平面起算 (= elevation), 不是从 boresight 起算.
# 每条弧上所有 ray 的 Z 分量恒等 sin(pitch), 即都在同一水平面 / 同一俯仰角.
# ================================================================================

def single_pitch_arc_pattern(cfg: "SinglePitchArcPatternCfg", device: str):
    """生成 (num_azimuth, 3) ray_starts (全零) + ray_directions.

    所有 ray 在同一俯仰角 cfg.pitch_deg 上 (= 水平面之 Z 分量一致), azimuth 从
    azimuth_range_deg[0] 到 [1] 线性采样 num_azimuth 个方向.
    """
    if cfg.num_azimuth < 1:
        raise ValueError(f"num_azimuth must be >= 1, got {cfg.num_azimuth}")

    pitch = torch.deg2rad(torch.tensor(cfg.pitch_deg, device=device, dtype=torch.float32))
    azim_deg = torch.linspace(cfg.azimuth_range_deg[0], cfg.azimuth_range_deg[1],
                              cfg.num_azimuth, device=device, dtype=torch.float32)
    azim = torch.deg2rad(azim_deg)

    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    dx = cos_p * torch.cos(azim)
    dy = cos_p * torch.sin(azim)
    dz = sin_p * torch.ones_like(azim)
    ray_directions = torch.stack([dx, dy, dz], dim=-1)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions


@configclass
class SinglePitchArcPatternCfg(PatternBaseCfg):
    """单 pitch 水平扇面 lidar pattern (用于 6+6 sensor 架构, 每 sensor 一个 pitch).

    Convention (elevation/azimuth, NOT mujoco polar/azim-from-boresight):
      - pitch:    elevation angle from horizontal plane, [-90°, 90°]
      - azimuth:  horizontal angle from sensor +X (forward), [-180°, 180°]
      - ray_dir = (cos(pitch)*cos(azim), cos(pitch)*sin(azim), sin(pitch))
      - 所有 ray 共用 sensor local 原点, Z 分量恒定 = sin(pitch).

    跟 baseline 12-sensor (forward_scanner_layer0..5 + backward_scanner_layer0..5)
    架构兼容: 每个 sensor 用此 pattern, pitch_deg 取 baseline 的 down_angles_deg 列表
    [-25, -15, -5, 5, 15, 25]. 后向 sensor 通过 sensor offset rot 180°Z 来反向.
    """

    func: Callable = single_pitch_arc_pattern

    pitch_deg: float = 0.0
    """单一俯仰角 (deg from horizontal, 正 = 上, 负 = 下)."""

    num_azimuth: int = 21
    """每弧 azimuth 采样数 (跟 baseline GridPattern lateral resolution 对齐)."""

    azimuth_range_deg: tuple[float, float] = (-90.0, 90.0)
    """Azimuth 范围 (deg). 默认 ±90° = 半圆扇 (跟 baseline GridPattern size 对齐)."""


# ================================================================================
# Multi-pitch arc pattern: 一个 sensor 内封装多个 pitch 弧 (= N 个 SinglePitchArc 打包)
# ================================================================================
# 用法: 替代 N 个独立 sensor 节省 IsaacLab sensor 管理开销, 同时几何完全等价于
# N 个 SinglePitchArcPatternCfg sensor stack 起来.
#
# 几何 (sensor frame, +X 前 +Y 左 +Z 上, elevation/azimuth 约定):
#   每个 pitch 一个 arc, 21 ray 沿 azimuth 均匀分布;
#   ray_dir[i, j] = (cos(p_i)*cos(a_j), cos(p_i)*sin(a_j), sin(p_i))
#   总 ray 数 = len(pitch_angles_deg) × num_azimuth
#   flatten 顺序: pitch-major, azimuth-minor (跟 ScanAE reshape (B, channels, rays) 对齐)
#
# 验证: 取 pitch_angles_deg=[-25,-15,-5,5,15,25] + num_azimuth=21 + azimuth_range=(-90,90),
# 输出的 126 条 ray 跟 6 个 SinglePitchArcPatternCfg sensor 各自输出 21 ray 拼起来
# 字节级等价 (前提是 flatten 顺序一致).
# ================================================================================

def multi_pitch_arc_pattern(cfg: "MultiPitchArcPatternCfg", device: str):
    """N pitches × M azimuth ray pattern (elevation/azimuth 约定, 跟 SinglePitchArc 同).

    flatten 顺序: pitch-major (外层), azimuth-minor (内层).
    ray idx k = pitch_idx * num_azimuth + azim_idx.
    """
    if not cfg.pitch_angles_deg:
        raise ValueError("pitch_angles_deg must be non-empty")
    if cfg.num_azimuth < 1:
        raise ValueError(f"num_azimuth must be >= 1, got {cfg.num_azimuth}")

    pitch_deg = torch.tensor(cfg.pitch_angles_deg, device=device, dtype=torch.float32)
    azim_deg = torch.linspace(cfg.azimuth_range_deg[0], cfg.azimuth_range_deg[1],
                              cfg.num_azimuth, device=device, dtype=torch.float32)
    pitch = torch.deg2rad(pitch_deg)
    azim = torch.deg2rad(azim_deg)

    # (P, A) 网格. indexing='ij' → pitch 外, azim 内. flat = pitch_idx * num_az + azim_idx.
    P, A = torch.meshgrid(pitch, azim, indexing="ij")
    cos_p = torch.cos(P)
    sin_p = torch.sin(P)
    dx = cos_p * torch.cos(A)
    dy = cos_p * torch.sin(A)
    dz = sin_p
    ray_directions = torch.stack([dx.flatten(), dy.flatten(), dz.flatten()], dim=-1)
    n_rays = ray_directions.shape[0]
    ray_starts = torch.zeros((n_rays, 3), device=device, dtype=torch.float32)
    return ray_starts, ray_directions


@configclass
class MultiPitchArcPatternCfg(PatternBaseCfg):
    """多 pitch 水平扇面 lidar pattern (N 个 SinglePitchArc 打包成一个 sensor).

    几何跟 SinglePitchArcPatternCfg 完全一致, 只是把多个 pitch 弧合并进同一个 sensor 节省
    IsaacLab sensor 对象管理开销. 输出 N × M 个 ray, flatten 顺序 pitch-major.

    用法 (2-sensor 取代 12-sensor):
        pattern = MultiPitchArcPatternCfg(
            pitch_angles_deg=[-25, -15, -5, 5, 15, 25],
            num_azimuth=21,
            azimuth_range_deg=(-90.0, 90.0),
        )
        前向 sensor 用此 pattern 不旋转 → 6 × 21 = 126 ray
        后向 sensor 用同一 pattern + offset.rot 绕 Z 180° → 另 126 ray
        NoisyElevationCfg 只 2 个 ObsTerm, ScanAE reshape (B, 12, 21) 字节对齐.
    """

    func: Callable = multi_pitch_arc_pattern

    pitch_angles_deg: list[float] = None
    """N 个俯仰角 (deg from horizontal). 不必均匀, 显式列出."""

    num_azimuth: int = 21
    """每弧 azimuth 采样数."""

    azimuth_range_deg: tuple[float, float] = (-90.0, 90.0)
    """Azimuth 范围 (deg). 默认 ±90° 跟 baseline 对齐."""
