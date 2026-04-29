"""Unit test for command class redesign — no IsaacSim.

Manually re-implements the resample logic from commands.py and validates the
distribution against expected fractions. No Isaac Sim or env spawning.

Run:
    python scripts/reinforcement_learning/rsl_rl/test_command_distribution.py
"""

import torch


def simulate_parent_uniform_velocity_command(
    n: int,
    rel_standing_envs: float,
    lin_x_range: tuple = (-1.0, 1.0),
    lin_y_range: tuple = (-1.0, 1.0),
    ang_z_range: tuple = (-1.0, 1.0),
):
    """模拟 IsaacLab 的父类 UniformVelocityCommand._resample_command 行为。"""
    # 均匀采样
    cmd = torch.empty(n, 3)
    cmd[:, 0] = torch.empty(n).uniform_(*lin_x_range)
    cmd[:, 1] = torch.empty(n).uniform_(*lin_y_range)
    cmd[:, 2] = torch.empty(n).uniform_(*ang_z_range)
    # standing envs 强制清零
    standing_dice = torch.rand(n)
    standing_envs = standing_dice < rel_standing_envs
    cmd[standing_envs] = 0.0
    return cmd


def my_resample_override(cmd: torch.Tensor, rel_standing_envs: float, rel_pure_turn_envs: float | None,
                          min_cmd_norm: float = 0.5):
    """复刻 commands.py 里 UniformThresholdVelocityCommand._resample_command 的 override 逻辑。"""
    n = cmd.shape[0]
    cmd = cmd.clone()

    # 默认 pure_turn 比例 = standing 比例
    rate_pt = rel_pure_turn_envs if rel_pure_turn_envs is not None else rel_standing_envs

    # 检测 standing（从 parent 输出）
    cur_norm = torch.norm(cmd, dim=1)
    standing_mask = cur_norm < 1e-4

    # 非 standing 中按 rate_pt 抽出 pure_turn
    rand = torch.rand(n)
    pure_turn_mask = (~standing_mask) & (rand < rate_pt)
    n_pt = int(pure_turn_mask.sum())
    if n_pt > 0:
        ang_dirs = torch.randint(0, 2, (n_pt,)) * 2 - 1
        ang_mags = torch.empty(n_pt).uniform_(0.5, 1.0)
        cmd[pure_turn_mask, 0] = 0.0
        cmd[pure_turn_mask, 1] = 0.0
        cmd[pure_turn_mask, 2] = ang_dirs.float() * ang_mags

    # moving envs：weak filter
    moving_mask = (~standing_mask) & (~pure_turn_mask)
    if moving_mask.any():
        mv_ids = torch.nonzero(moving_mask, as_tuple=True)[0]
        mv_norm = torch.norm(cmd[mv_ids], dim=1)
        invalid = (mv_norm < min_cmd_norm) & (mv_norm > 1e-4)
        if invalid.any():
            inv_ids = mv_ids[invalid]
            n_inv = len(inv_ids)
            lin_dirs = torch.randint(0, 2, (n_inv,)) * 2 - 1
            lin_mags = torch.empty(n_inv).uniform_(0.5, 1.5)
            cmd[inv_ids, 0] = lin_dirs.float() * lin_mags
            cmd[inv_ids, 1] = 0.0

    return cmd


def categorize(cmd: torch.Tensor):
    """分类：standing / pure_turn / moving."""
    lin_norm = torch.norm(cmd[:, :2], dim=1)
    ang_abs = torch.abs(cmd[:, 2])
    total_norm = torch.norm(cmd, dim=1)
    standing = total_norm < 1e-4
    pure_turn = (~standing) & (lin_norm < 0.05) & (ang_abs >= 0.4)
    moving = (~standing) & (~pure_turn)
    return standing, pure_turn, moving


def test(rel_standing: float, rel_pure_turn=None, n_per_round=10000, n_rounds=10):
    """跑多轮、累积大量样本统计分布。"""
    torch.manual_seed(42)

    rate_pt_eff = rel_pure_turn if rel_pure_turn is not None else rel_standing

    all_cmds = []
    for _ in range(n_rounds):
        # parent
        cmd = simulate_parent_uniform_velocity_command(n_per_round, rel_standing)
        # 我的 override
        cmd = my_resample_override(cmd, rel_standing, rel_pure_turn)
        all_cmds.append(cmd)
    cmds = torch.cat(all_cmds, dim=0)
    n_total = cmds.shape[0]

    standing, pure_turn, moving = categorize(cmds)
    n_st = int(standing.sum())
    n_pt = int(pure_turn.sum())
    n_mv = int(moving.sum())

    # 期望（解析）
    # standing: rel_standing
    # pure_turn 显式分配: (1 - rel_standing) × rate_pt_eff
    # natural pure_turn from moving (passing weak filter): 由均匀采样几何决定，估算 ~1.5%
    expected_st = rel_standing
    expected_pt_explicit = (1 - rel_standing) * rate_pt_eff
    # 自然纯转向：lin_xy² + ang_z² ≥ 0.25 AND lin_xy < 0.05 AND |ang_z| ≥ 0.4
    # 在 [-1,1]³ 均匀分布上 P(lin_x<0.05, lin_y<0.05) = (0.1/2)² = 0.0025
    # × P(|ang_z| ≥ 0.4 | total_norm ≥ 0.5) ≈ 0.6
    # ≈ 0.15% 量级，太小，归到 moving 即可
    expected_pt_total = expected_pt_explicit + 0.0015  # 加一点点自然贡献

    print(f"\n--- rel_standing={rel_standing}, rel_pure_turn={rel_pure_turn} (effective {rate_pt_eff}) ---")
    print(f"   total samples: {n_total}")
    print(f"   {'category':<12} {'measured':>10} {'expected':>10} {'OK?'}")
    print(f"   {'-'*44}")
    pct_st = n_st / n_total
    pct_pt = n_pt / n_total
    pct_mv = n_mv / n_total
    ok_st = abs(pct_st - expected_st) < 0.01
    ok_pt = abs(pct_pt - expected_pt_total) < 0.01
    print(f"   {'standing':<12} {pct_st:>10.3%} {expected_st:>10.3%}  {'✓' if ok_st else '✗'}")
    print(f"   {'pure_turn':<12} {pct_pt:>10.3%} {expected_pt_total:>10.3%}  {'✓' if ok_pt else '✗'}")
    print(f"   {'moving':<12} {pct_mv:>10.3%}")

    # moving 的 total_norm 是否真的 ≥ 0.5（保证没有 wishy-washy 残留）
    if moving.any():
        mv_norm = torch.norm(cmds[moving], dim=1)
        # 注意：参数 min_cmd_norm 是 3D，filter 阈值是 0.5
        # 没有 remap 的"自然小但 ≥ 0.5"应该没有；remap 后的 lin_x ≥ 0.5 也保证 norm ≥ 0.5
        n_below = int((mv_norm < 0.5).sum())
        n_zero_or_below_eps = int((mv_norm < 1e-4).sum())
        below_but_nonzero = n_below - n_zero_or_below_eps
        print(f"   moving total_norm < 0.5 (excluding zero): {below_but_nonzero}  (理想 0)")
        print(f"   moving total_norm range: [{mv_norm.min():.3f}, {mv_norm.max():.3f}]")


if __name__ == "__main__":
    print("=" * 60)
    print(" Command Distribution Unit Test (no IsaacSim)")
    print("=" * 60)

    # 案例 1: 默认行为 — pure_turn 自动跟随 standing
    test(rel_standing=0.05, rel_pure_turn=None)

    # 案例 2: 显式 5% pure_turn (应等同于案例 1)
    test(rel_standing=0.05, rel_pure_turn=0.05)

    # 案例 3: 不同的 standing/pure_turn 比例
    test(rel_standing=0.10, rel_pure_turn=0.20)

    # 案例 4: pure_turn 关闭
    test(rel_standing=0.05, rel_pure_turn=0.0)

    # 案例 5: 极端 — 全部 pure_turn (除 standing 外)
    test(rel_standing=0.05, rel_pure_turn=1.0)

    print("\n" + "=" * 60)
    print(" 解析说明：")
    print(" - standing rate  = rel_standing_envs (parent 直接控制)")
    print(" - pure_turn rate = (1 - rel_standing_envs) × rate_pt_eff + 自然贡献(~0.15%)")
    print(" - 即默认时 pure_turn ≈ 0.95 × 0.05 = 4.75% (略低于 standing 5%)")
    print(" - 想严格 5%/5% 对齐，需把抽样改成 rate_pt / (1-rel_standing)，")
    print("   但 0.25 个百分点的差异通常无关紧要")
    print("=" * 60)
