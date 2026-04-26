"""Integration test: load _mirror_obs and _mirror_obs_fb from the actual source file
and verify their output matches the reference implementation in test_mirror_fb.py.

We avoid importing the full moe_terrain.py module (which would pull in IsaacLab).
Instead, we extract the two methods via AST and exec them in a minimal namespace
with a mock `self` (only needs .device and .actor_critic with a few attributes).
"""
from __future__ import annotations
import ast
import os
import sys
import textwrap
import torch
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mirror_fb as ref
from conftest import MOE_TERRAIN_PATH

SOURCE_FILE = str(MOE_TERRAIN_PATH)


def extract_method_from_class(source_text: str, class_name: str, method_name: str) -> str:
    """Use AST to find the method, then return its source text dedented to top-level."""
    tree = ast.parse(source_text)
    src_lines = source_text.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    start = sub.lineno - 1
                    end = sub.end_lineno  # type: ignore[attr-defined]
                    method_src = "\n".join(src_lines[start:end])
                    return textwrap.dedent(method_src)
    raise RuntimeError(f"method {class_name}.{method_name} not found")


def build_callable(method_src: str):
    """Compile the method as a top-level def, exec into namespace, return function."""
    ns = {"torch": torch}
    exec(method_src, ns)
    fn_name = method_src.split("(")[0].split()[-1]  # 'def NAME(...)' -> NAME
    return ns[fn_name]


def make_mock_self(device):
    """Build a mock self that satisfies the methods' attribute access."""
    model = types.SimpleNamespace(
        use_elevation_ae=True,
        use_multilayer_scan=True,
        elevation_dim=ref.ELEV_DIM,
        scan_dim=ref.SCAN_DIM,
        num_scan_channels=ref.SCAN_CH,
        num_scan_rays=ref.SCAN_RAY,
    )
    return types.SimpleNamespace(
        device=device,
        actor_critic=model,
        policy=model,
    )


def assert_close_dict(a, b, name, atol=1e-6):
    ok = True
    for k in a:
        if not torch.allclose(a[k], b[k], atol=atol):
            diff = (a[k] - b[k]).abs().max().item()
            print(f"  ✗ FAIL {name}/{k}: max_abs_diff={diff:.3e}")
            ok = False
        else:
            print(f"  ✓ {name}/{k}")
    return ok


def main():
    print(f"Loading source from {SOURCE_FILE}")
    with open(SOURCE_FILE) as f:
        src = f.read()

    method_lr_src = extract_method_from_class(src, "SplitMoEPPO", "_mirror_obs")
    method_fb_src = extract_method_from_class(src, "SplitMoEPPO", "_mirror_obs_fb")

    print(f"  _mirror_obs:    {method_lr_src.count(chr(10))} lines")
    print(f"  _mirror_obs_fb: {method_fb_src.count(chr(10))} lines")

    mirror_lr = build_callable(method_lr_src)
    mirror_fb = build_callable(method_fb_src)

    device = torch.device("cpu")
    mock_self = make_mock_self(device)

    # Build a few random obs and compare against reference.
    all_ok = True
    for B, hist, seed in [(4, 5, 0), (1, 1, 100), (8, 3, 42)]:
        obs = ref.make_obs(B=B, hist=hist, seed=seed)

        # ----- L-R: source vs reference -----
        out_src_lr = mirror_lr(mock_self, obs)
        out_ref_lr = ref.reference_mirror_lr(obs)
        print(f"\n[L-R B={B} hist={hist} seed={seed}]")
        all_ok &= assert_close_dict(out_src_lr, out_ref_lr, "LR src=ref")

        # ----- F-B: source vs reference -----
        out_src_fb = mirror_fb(mock_self, obs)
        out_ref_fb = ref.reference_mirror_fb(obs)
        print(f"[F-B B={B} hist={hist} seed={seed}]")
        all_ok &= assert_close_dict(out_src_fb, out_ref_fb, "FB src=ref")

        # ----- Group property: F-B ∘ F-B = id (with source method) -----
        out_double_fb = mirror_fb(mock_self, mirror_fb(mock_self, obs))
        all_ok &= assert_close_dict(out_double_fb, obs, "src FB^2=id")

        # ----- Commutativity (with source methods) -----
        a = mirror_lr(mock_self, mirror_fb(mock_self, obs))
        b = mirror_fb(mock_self, mirror_lr(mock_self, obs))
        all_ok &= assert_close_dict(a, b, "src LR∘FB = FB∘LR")

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL INTEGRATION TESTS PASSED")
        return 0
    else:
        print("SOME INTEGRATION TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
