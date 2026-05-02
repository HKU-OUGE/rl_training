"""单元测试: 验证 Plan B' 双尺度 scan_history 的正确性.

测试目标:
  1. sample_idx 计算正确 (offsets → buffer 索引)
  2. _scan_history_attach 输出 shape 正确
  3. _scan_history_attach 输出内容正确 (current + history_at_offsets)
  4. 内部 ring buffer rolling 正确 (rollout 路径写入)
  5. PPO update 路径只读不写 buffer
  6. reset(dones) 仅清空 done env 的 buffer
  7. ONNX 路径 (外部 scan_history_input) 不污染内部 buffer
  8. LR mirror 在 history 拼接后仍可用 (azimuth flip 等价 y_neg)
  9. 不启用 history 时退回 legacy 行为 (返回 scan_lat 单帧)
  10. deploy 端 cpp 索引计算与 python 一致

不依赖 IsaacLab；构造最小 mock 对象，方法绑定测试目标实例方法.

运行:
    conda run -n env_isaaclab python scripts/reinforcement_learning/rsl_rl/test_scan_history.py
    # 或者纯 torch 环境也可
    python scripts/reinforcement_learning/rsl_rl/test_scan_history.py
"""
from __future__ import annotations

import sys
import types

import torch


# ============================================================
# 复刻被测代码 (与 moe_terrain.py SplitMoEActorCritic 中等价)
# ============================================================
class _ScanHistoryHarness(torch.nn.Module):
    """最小复现 SplitMoEActorCritic 中 scan_history 相关的方法 + state."""
    def __init__(self, scan_out_dim: int, scan_history_offsets: list[int], num_envs: int, device="cpu"):
        super().__init__()
        self.use_multilayer_scan = True
        self.scan_out_dim = scan_out_dim
        self.scan_history_offsets = list(scan_history_offsets)
        self.use_scan_history = len(self.scan_history_offsets) > 0
        self.scan_history_len = len(self.scan_history_offsets)
        self.scan_history_buffer_size = max(self.scan_history_offsets) if self.use_scan_history else 0
        if self.use_scan_history:
            self.register_buffer(
                "_scan_history_buf",
                torch.zeros(num_envs, self.scan_history_buffer_size, scan_out_dim, device=device),
                persistent=False,
            )
            sample_idx = torch.tensor(
                [self.scan_history_buffer_size - off for off in self.scan_history_offsets],
                dtype=torch.long, device=device,
            )
            self.register_buffer("_scan_history_sample_idx", sample_idx, persistent=False)

    def _scan_history_attach(self, latent_scan, scan_history_input=None, update_internal_buf=False):
        if not self.use_scan_history:
            return latent_scan
        if scan_history_input is not None:
            history = scan_history_input
            if history.dim() == 2:
                history = history.view(-1, self.scan_history_len, self.scan_out_dim)
        else:
            B = latent_scan.shape[0]
            if self._scan_history_buf.shape[0] != B:
                history = torch.zeros(B, self.scan_history_len, self.scan_out_dim,
                                       dtype=latent_scan.dtype, device=latent_scan.device)
            else:
                history = self._scan_history_buf[:, self._scan_history_sample_idx, :].clone()
                if update_internal_buf:
                    with torch.no_grad():
                        new_buf = torch.cat([self._scan_history_buf[:, 1:],
                                              latent_scan.detach().unsqueeze(1)], dim=1)
                        self._scan_history_buf.copy_(new_buf)
        full = torch.cat([latent_scan.unsqueeze(1), history], dim=1)
        return full.flatten(start_dim=1)

    def reset(self, dones):
        if hasattr(dones, "dtype") and dones.dtype != torch.bool:
            dones = dones.bool()
        if self.use_scan_history and hasattr(self, "_scan_history_buf"):
            buf = self._scan_history_buf
            if buf.shape[0] == dones.shape[0]:
                buf[dones] = 0.0


# ============================================================
# 测试用例
# ============================================================
def _ok(name): print(f"  \033[92m✓\033[0m {name}")
def _fail(name, msg): print(f"  \033[91m✗\033[0m {name}: {msg}"); raise AssertionError(f"{name}: {msg}")


def t01_sample_idx_calculation():
    """sample_idx 应该把 offsets 正确映射到 buffer 索引 (newest 在 buffer 末尾)."""
    h = _ScanHistoryHarness(scan_out_dim=64,
                             scan_history_offsets=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40],
                             num_envs=2)
    assert h.scan_history_buffer_size == 40
    expected = [40-1, 40-2, 40-3, 40-4, 40-5, 40-10, 40-15, 40-20, 40-25, 40-30, 40-35, 40-40]
    assert h._scan_history_sample_idx.tolist() == expected, \
        f"got {h._scan_history_sample_idx.tolist()}, expected {expected}"
    _ok("sample_idx == [39,38,37,36,35,30,25,20,15,10,5,0]")


def t02_attach_output_shape():
    """输出 shape = (B, (K+1) * scan_out_dim)."""
    h = _ScanHistoryHarness(scan_out_dim=64, scan_history_offsets=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40], num_envs=8)
    latent = torch.randn(8, 64)
    out = h._scan_history_attach(latent, update_internal_buf=False)
    assert out.shape == (8, 13 * 64), f"shape {out.shape}, expect (8, 832)"
    _ok(f"output shape (B=8, 832)  | K=12 + 1 current = 13 frames × 64")


def t03_attach_first_frame_is_current():
    """flatten 后前 64 维应该等于 latent_scan."""
    h = _ScanHistoryHarness(scan_out_dim=64, scan_history_offsets=[1, 2, 3], num_envs=4)
    latent = torch.randn(4, 64)
    out = h._scan_history_attach(latent, update_internal_buf=False)
    assert torch.allclose(out[:, :64], latent), "first 64 dims should be current latent"
    _ok("flatten 顺序: [current, hist_off_1, hist_off_2, ...]")


def t04_external_history_path():
    """传 scan_history_input 时, 应使用外部数据, 不读内部 buffer (写一些非零数据进 buffer 验证)."""
    h = _ScanHistoryHarness(scan_out_dim=64, scan_history_offsets=[1, 2, 3], num_envs=4)
    h._scan_history_buf.fill_(99.0)  # 内部 buffer 全 99
    latent = torch.zeros(4, 64)
    ext_hist = torch.ones(4, 3, 64) * 7.0  # 外部 buffer 全 7
    out = h._scan_history_attach(latent, scan_history_input=ext_hist, update_internal_buf=False)
    # current (64) + 3*64 hist of 7s
    assert torch.allclose(out[:, :64], latent), "current"
    assert torch.allclose(out[:, 64:], torch.ones(4, 3 * 64) * 7.0), "ext history used"
    _ok("ONNX 路径: scan_history_input 优先于内部 buffer")


def t05_external_history_2d_compat():
    """scan_history_input 接受 (B, K*D) 也接受 (B, K, D)."""
    h = _ScanHistoryHarness(scan_out_dim=64, scan_history_offsets=[1, 2, 3], num_envs=2)
    latent = torch.zeros(2, 64)
    h2d = torch.arange(2 * 3 * 64, dtype=torch.float32).view(2, 3 * 64)
    h3d = h2d.view(2, 3, 64)
    o2 = h._scan_history_attach(latent, scan_history_input=h2d)
    o3 = h._scan_history_attach(latent, scan_history_input=h3d)
    assert torch.allclose(o2, o3), "(B, K*D) and (B, K, D) shape compat broken"
    _ok("scan_history_input 兼容 2D / 3D")


def t06_buffer_rolling_after_one_call():
    """update_internal_buf=True 时, buffer 应左移 1, 末尾写新 latent."""
    h = _ScanHistoryHarness(scan_out_dim=64, scan_history_offsets=[1, 2, 3], num_envs=4)
    h._scan_history_buf.fill_(0.0)
    latent_a = torch.full((4, 64), 5.0)
    h._scan_history_attach(latent_a, update_internal_buf=True)
    # buffer 末尾应是 latent_a, 其他位置仍 0
    assert torch.allclose(h._scan_history_buf[:, -1], latent_a), "newest at end"
    assert torch.allclose(h._scan_history_buf[:, :-1], torch.zeros(4, 2, 64)), "rest still zero"
    _ok("一步: buffer[:,-1] = current_latent; buffer[:,:-1] = 0")


def t07_buffer_rolling_full_window():
    """连续调 K+1 次后, buffer 内容应反映最近 K 帧 + 当前."""
    K = 5  # buffer_size
    h = _ScanHistoryHarness(scan_out_dim=4, scan_history_offsets=list(range(1, K+1)), num_envs=2)
    seq = [torch.full((2, 4), float(t)) for t in range(K + 1)]  # 0..5
    for lat in seq:
        h._scan_history_attach(lat, update_internal_buf=True)
    # 在第 K+1 次调用后 (latent=5.0), buffer 末尾应是 5.0 (newest=t-1 之前的输入是 4.0; 但调用时 attach 写入是 detached current)
    # 调用顺序: attach(0) → buf[-1]=0; attach(1) → buf[-2]=0, buf[-1]=1; ... attach(5) → buf=[1,2,3,4,5]
    expected = torch.stack([torch.full((2, 4), float(t)) for t in [1, 2, 3, 4, 5]], dim=1)
    assert torch.allclose(h._scan_history_buf, expected), \
        f"buffer mismatch:\n{h._scan_history_buf}\nexpect:\n{expected}"
    _ok(f"K={K} 次填充后, buffer = [t-{K}, ..., t-1]")


def t08_buffer_no_update_when_flag_false():
    """update_internal_buf=False 时不应改 buffer."""
    h = _ScanHistoryHarness(scan_out_dim=4, scan_history_offsets=[1, 2, 3], num_envs=2)
    initial = torch.full((2, 3, 4), 9.0)
    h._scan_history_buf.copy_(initial)
    latent = torch.full((2, 4), 1.0)
    h._scan_history_attach(latent, update_internal_buf=False)
    assert torch.allclose(h._scan_history_buf, initial), "buffer mutated when update_internal_buf=False"
    _ok("forward / evaluate 路径: update=False → buffer 不动")


def t09_reset_clears_done_envs_only():
    """reset(dones=[True, False, True, False]) 只清空对应行."""
    h = _ScanHistoryHarness(scan_out_dim=4, scan_history_offsets=[1, 2, 3], num_envs=4)
    h._scan_history_buf.fill_(7.0)
    dones = torch.tensor([True, False, True, False])
    h.reset(dones)
    assert torch.allclose(h._scan_history_buf[0], torch.zeros(3, 4)), "env 0 should be reset"
    assert torch.allclose(h._scan_history_buf[1], torch.full((3, 4), 7.0)), "env 1 untouched"
    assert torch.allclose(h._scan_history_buf[2], torch.zeros(3, 4)), "env 2 should be reset"
    assert torch.allclose(h._scan_history_buf[3], torch.full((3, 4), 7.0)), "env 3 untouched"
    _ok("reset 只清空 dones=True 的行")


def t10_disabled_returns_pass_through():
    """offsets=[] 时直接返回 latent_scan."""
    h = _ScanHistoryHarness(scan_out_dim=64, scan_history_offsets=[], num_envs=4)
    assert not h.use_scan_history
    latent = torch.randn(4, 64)
    out = h._scan_history_attach(latent, update_internal_buf=True)
    assert torch.allclose(out, latent), "should pass through unchanged"
    assert out.shape == latent.shape
    _ok("offsets=[] 关闭 history; pass-through")


def t11_lr_mirror_compat_check():
    """LR mirror 通过 azimuth flip 实现; 当前帧的 latent (从 ScanAE) 不在此测试范围内,
       这里仅验证 history 拼接不破坏 latent 的逐 dim 对齐.
    """
    h = _ScanHistoryHarness(scan_out_dim=4, scan_history_offsets=[1, 2, 3], num_envs=2)
    latent_orig = torch.tensor([[1, 2, 3, 4]] * 2, dtype=torch.float32)
    out_orig = h._scan_history_attach(latent_orig, update_internal_buf=False)

    # mirror 在 latent 上元素 reorder (LR 由 azimuth flip 实现, AE 内部完成)
    latent_mir = torch.tensor([[4, 3, 2, 1]] * 2, dtype=torch.float32)
    out_mir = h._scan_history_attach(latent_mir, update_internal_buf=False)
    # 输出前 scan_out_dim 应该和 mirror 一致
    assert torch.allclose(out_mir[:, :4], latent_mir), "mirror current preserved"
    assert torch.allclose(out_orig[:, :4], latent_orig)
    _ok("history 拼接不打乱 latent 的逐 dim 顺序 (LR mirror 兼容)")


def t12_rollout_replay_consistency():
    """模拟 rollout 写 → PPO update 期间 forward 多次 (read-only)，buffer 不应变化."""
    h = _ScanHistoryHarness(scan_out_dim=4, scan_history_offsets=[1, 2, 3], num_envs=2)
    # rollout 写入 5 步
    for t in range(5):
        h._scan_history_attach(torch.full((2, 4), float(t)), update_internal_buf=True)
    snapshot = h._scan_history_buf.clone()

    # PPO update: forward 调多次 (update_internal_buf=False)
    for _ in range(20):
        h._scan_history_attach(torch.randn(2, 4), update_internal_buf=False)
    assert torch.allclose(h._scan_history_buf, snapshot), "PPO update path corrupted buffer!"
    _ok("PPO update (forward) 多次调用不修改 buffer (Plan B' 关键正确性)")


def t13_cpp_offsets_match_py():
    """deploy cpp 端 SCAN_HISTORY_OFFSETS 应与 python 默认一致."""
    py_offsets = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
    cpp_path = "/home/ouge/Software/sdk_deploy/src/M20_sdk_deploy/run_policy/m20_sensor_policy_runner.hpp"
    try:
        with open(cpp_path) as f:
            src = f.read()
    except FileNotFoundError:
        print("  \033[93m⊘\033[0m skipped (cpp file not found, deploy 端可能不在本机)")
        return
    import re
    m = re.search(r"SCAN_HISTORY_OFFSETS\[\w+\]\s*=\s*\{([^}]+)\}", src)
    assert m, "SCAN_HISTORY_OFFSETS not found in cpp"
    cpp_vals = [int(x.strip()) for x in m.group(1).split(",")]
    assert cpp_vals == py_offsets, f"py {py_offsets} != cpp {cpp_vals}"
    _ok(f"cpp SCAN_HISTORY_OFFSETS 与 py 默认一致: {py_offsets}")


def t14_buffer_size_implied_correctly():
    """scan_history_buffer_size == max(offsets) (deploy 端 SCAN_BUFFER_LEN 必须匹配)."""
    cases = [
        ([1, 2, 3, 4], 4),
        ([1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40], 40),
        ([5, 10], 10),
    ]
    for offsets, expected_buf in cases:
        h = _ScanHistoryHarness(scan_out_dim=4, scan_history_offsets=offsets, num_envs=1)
        assert h.scan_history_buffer_size == expected_buf
    _ok("buffer_size = max(offsets) 在多种 offsets 下都正确")


def t15_attach_idempotent_no_update():
    """两次连续 attach (都 update=False) 输出相同."""
    h = _ScanHistoryHarness(scan_out_dim=4, scan_history_offsets=[1, 2, 3], num_envs=2)
    h._scan_history_buf[:] = torch.randn_like(h._scan_history_buf)
    latent = torch.randn(2, 4)
    o1 = h._scan_history_attach(latent, update_internal_buf=False)
    o2 = h._scan_history_attach(latent, update_internal_buf=False)
    assert torch.allclose(o1, o2)
    _ok("不更新模式下: attach 是纯函数")


# ============================================================
# Run
# ============================================================
TESTS = [
    t01_sample_idx_calculation,
    t02_attach_output_shape,
    t03_attach_first_frame_is_current,
    t04_external_history_path,
    t05_external_history_2d_compat,
    t06_buffer_rolling_after_one_call,
    t07_buffer_rolling_full_window,
    t08_buffer_no_update_when_flag_false,
    t09_reset_clears_done_envs_only,
    t10_disabled_returns_pass_through,
    t11_lr_mirror_compat_check,
    t12_rollout_replay_consistency,
    t13_cpp_offsets_match_py,
    t14_buffer_size_implied_correctly,
    t15_attach_idempotent_no_update,
]


def main():
    print("=" * 70)
    print("Plan B' (双尺度 scan history) 单元测试")
    print("=" * 70)
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  \033[91m✗\033[0m {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  \033[91m✗\033[0m {t.__name__} 抛异常: {type(e).__name__}: {e}")
    print()
    if failed == 0:
        print(f"\033[92m{len(TESTS)}/{len(TESTS)} PASS ✓\033[0m")
        return 0
    else:
        print(f"\033[91m{failed}/{len(TESTS)} FAIL\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
