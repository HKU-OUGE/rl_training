"""Verify sim2sim mujoco pipeline depth correctness (no ROS, no MuJoCo needed).

Hypothesis chain:
  Isaac Sim training (multi_layer_scan, BUGGY): depth = ||hit - base_link||
                                                       = sensor_to_hit + 32cm (forward sensor)
  MuJoCo sim2sim (lidar_to_scan.py, CORRECT)    : depth = ||(hit_in_base_frame) - sensor_offset||
                                                       = sensor_to_hit
  Real robot (lidar_to_scan.cpp, CORRECT)       : depth = ||hit_in_base_frame - sensor_offset||
                                                       = sensor_to_hit

We replicate the sim2sim binning math here on synthetic wall points and confirm:
  output bin value == true sensor-to-wall distance (NOT base_link-to-wall).

This confirms sim2sim deployment exposes the SAME 32cm OOD as sim2real
(policy trained on inflated depths sees correct depths at deployment).
"""

import math
import numpy as np

NUM_POLAR = 16
NUM_AZ = 31
NUM_PER_DIR = NUM_POLAR * NUM_AZ
MAX_DIST = 2.5
MIN_DIST = 0.3
POLAR_MAX_RAD = math.radians(80.0)
TWO_PI = 2 * math.pi
FWD_OFFSET = np.array([0.32028, 0.0, -0.013], dtype=np.float32)


def bin_one_direction_replica(pts, boresight_sign):
    """Exact replica of lidar_to_scan.py:_bin_one_direction (deploy code path)."""
    bins = np.full(NUM_PER_DIR, MAX_DIST, dtype=np.float32)
    if pts.size == 0:
        return bins
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if boresight_sign > 0:
        mask = x > 0.0
        x_b, y_b = x, y
    else:
        mask = x < 0.0
        x_b, y_b = -x, -y
    if not np.any(mask):
        return bins
    x_b, y_b, z_b = x_b[mask], y_b[mask], z[mask]
    r = np.sqrt(x_b * x_b + y_b * y_b + z_b * z_b)
    valid = (r >= MIN_DIST) & (r <= MAX_DIST)
    if not np.any(valid):
        return bins
    x_b, y_b, z_b, r = x_b[valid], y_b[valid], z_b[valid], r[valid]
    perp = np.sqrt(y_b * y_b + z_b * z_b)
    polar = np.arctan2(perp, x_b)
    in_fov = polar <= POLAR_MAX_RAD
    if not np.any(in_fov):
        return bins
    polar, r, y_b, z_b = polar[in_fov], r[in_fov], y_b[in_fov], z_b[in_fov]
    az = np.arctan2(y_b, z_b)
    p_idx = np.clip(np.rint(polar * (NUM_POLAR - 1) / POLAR_MAX_RAD).astype(np.int32), 0, NUM_POLAR - 1)
    a_idx = np.clip(np.rint((az + math.pi) * (NUM_AZ - 1) / TWO_PI).astype(np.int32), 0, NUM_AZ - 1)
    flat = p_idx * NUM_AZ + a_idx
    np.minimum.at(bins, flat, r)
    return bins


def run_test(distance_base_link_to_wall):
    """Wall at +X, 'distance_base_link_to_wall' meters from base_link.
    Sensor at +0.32m, so true sensor-to-wall = distance - 0.32."""
    # Build synthetic wall points in base_link frame
    # Wall is a vertical plane at x = distance_base_link_to_wall, spanning y in [-0.5, 0.5], z in [0, 0.6]
    n_per_axis = 50
    ys = np.linspace(-0.5, 0.5, n_per_axis)
    zs = np.linspace(0.0, 0.6, n_per_axis)
    yy, zz = np.meshgrid(ys, zs)
    wall_pts_base = np.stack([
        np.full_like(yy.flatten(), distance_base_link_to_wall),
        yy.flatten(),
        zz.flatten(),
    ], axis=-1).astype(np.float32)

    # Mimic lidar_to_scan.py path: subtract fwd_offset → points now in sensor frame
    rel_fwd = wall_pts_base - FWD_OFFSET
    bins = bin_one_direction_replica(rel_fwd, boresight_sign=+1)

    # The polar=0 bin (azimuth degenerate, all 31 az → forward) should hold the wall hit
    polar0 = bins[0:NUM_AZ]                       # shape (31,)
    polar0_min = polar0.min()                      # min over all azimuths at polar=0

    expected_sensor_to_wall = distance_base_link_to_wall - FWD_OFFSET[0]
    return polar0_min, expected_sensor_to_wall


def main():
    print("=" * 80)
    print("MuJoCo sim2sim lidar_to_scan.py logical verification")
    print("=" * 80)
    print(f"Sensor offset: forward = +{FWD_OFFSET[0]:.5f}m from base_link\n")

    print(f"{'wall→base_link (m)':<22}{'expected (m)':<25}{'actual bin (m)':<18}{'OK?':<8}")
    print(f"{'':22}{'(blind if <0.3)':<25}")
    print("-" * 80)

    all_ok = True
    for d in [2.0, 1.5, 1.0, 0.7, 0.5, 0.4, 0.35, 0.32, 0.20]:
        actual, true_sensor_dist = run_test(d)
        # Deploy 逻辑: sensor→wall < 0.3 → 输出 2.5 (MAX_DIST/blind), 否则输出真实距离
        if true_sensor_dist < MIN_DIST:
            expected = MAX_DIST  # blind fallback
            note = f"{true_sensor_dist:.3f} < 0.3 → blind (2.5)"
        else:
            expected = true_sensor_dist
            note = f"{true_sensor_dist:.3f}"
        ok = abs(actual - expected) < 0.05
        all_ok &= ok
        marker = "✓" if ok else "✗"
        print(f"{d:<22.2f}{note:<25}{actual:<18.3f}{marker:<8}")

    print()
    if all_ok:
        print("✅ sim2sim mujoco pipeline reports CORRECT sensor-to-wall distances.")
        print("   It does NOT have the 32cm bug — it's identical to real robot's lidar_to_scan.cpp logic.")
        print()
        print("⇒ Same conclusion as sim2real: policy was trained on Isaac Sim BUGGY 32cm-inflated")
        print("   depths, then receives CORRECT depths at deployment in BOTH MuJoCo sim2sim and real.")
        print("   Both deployments expose the same scan-OOD; sim2sim gap should be similar to sim2real")
        print("   for perception. If user observed sim2sim gap < sim2real, the difference is not in")
        print("   perception but in other physics: motor/contact/friction (MuJoCo > real fidelity).")
    else:
        print("❌ Some test cases failed.")


if __name__ == "__main__":
    main()
