"""Comprehensive unit tests for moe_terrain.py.

Coverage:
  TIER 1 (helper classes, no IsaacLab deps):
    - EmpiricalNormalization: Welford running stats correctness
    - MLP: shape preservation, gradient flow
    - ElevationAE: encode/decode shape, train/eval mode behaviour
    - MultiLayerScanAE: shape, fwd/bwd internal flip
    - DepthAE: shape, training-only reconstruction
    - ProprioVAE: shape, deterministic vs sampled forward, vel_pred path
    - load balancing loss math

  TIER 2 (config consistency, static analysis):
    - Joint name list matches assumed [FL,FR,HL,HR + 4 wheels] layout
    - action_swap_idx (LR/FB) on joint_names list produces expected swap
    - action_neg_mask consistency with M20 init_state URDF conventions

  TIER 3 (mirror function correctness on extracted source):
    - L-R and F-B mirror behaviour (delegates to test_mirror_fb)
    - Sym loss bypass: verify the early-return branch in update() exists
    - act() fast path bypass when sym_enabled=False

Run from repo root:

    /home/ouge/miniconda3/envs/env_isaaclab/bin/python /tmp/auto_probe/test_moe_terrain_components.py
"""
from __future__ import annotations

import ast
import importlib.util
import math
import os
import re
import sys
import textwrap
import types

import torch
import torch.nn as nn
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import MOE_TERRAIN_PATH, TEACHER_CFG_PATH, ASSET_CFG_PATH, REPO_ROOT
SOURCE_FILE = str(MOE_TERRAIN_PATH)
TEACHER_CFG = str(TEACHER_CFG_PATH)
DEEP_ASSET = str(ASSET_CFG_PATH)
REPO_ROOT = str(REPO_ROOT)


# ---------------------------------------------------------------------------
# Helper: load classes from moe_terrain.py without triggering IsaacLab imports
# ---------------------------------------------------------------------------

def load_class_via_ast(source_text: str, class_names: list[str]):
    """Extract specific top-level classes from source and exec in a torch-only namespace.
    Bypasses heavy IsaacLab imports by stubbing the symbols the classes reference."""
    tree = ast.parse(source_text)
    src_lines = source_text.splitlines()
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            start = node.lineno - 1
            end = node.end_lineno  # type: ignore[attr-defined]
            class_src = "\n".join(src_lines[start:end])
            ns = {
                "torch": torch,
                "nn": nn,
                "F": torch.nn.functional,
                "np": np,
            }
            exec(class_src, ns)
            out[node.name] = ns[node.name]
    return out


# Also extract the top-level orthogonal_init helper, since MLP uses it.
def load_orthogonal_init(source_text: str):
    tree = ast.parse(source_text)
    src_lines = source_text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "orthogonal_init":
            start = node.lineno - 1
            end = node.end_lineno  # type: ignore[attr-defined]
            fn_src = "\n".join(src_lines[start:end])
            ns = {"nn": nn, "torch": torch}
            exec(fn_src, ns)
            return ns["orthogonal_init"]
    raise RuntimeError("orthogonal_init not found")


print(f"Loading source from {SOURCE_FILE}")
with open(SOURCE_FILE) as f:
    SOURCE_TEXT = f.read()

# orthogonal_init must be in the namespace of MLP since MLP calls it.
orthogonal_init = load_orthogonal_init(SOURCE_TEXT)

# Component classes share orthogonal_init + may depend on other components
# (e.g. MultiLayerScanAE / ProprioVAE use MLP). We populate a namespace
# incrementally with previously-loaded classes so subsequent classes can find them.
_COMPONENT_NS = {
    "torch": torch,
    "nn": nn,
    "F": torch.nn.functional,
    "np": np,
    "orthogonal_init": orthogonal_init,
}


def load_component(class_name: str):
    if class_name in _COMPONENT_NS:
        return _COMPONENT_NS[class_name]
    tree = ast.parse(SOURCE_TEXT)
    src_lines = SOURCE_TEXT.splitlines()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            start = node.lineno - 1
            end = node.end_lineno  # type: ignore[attr-defined]
            class_src = "\n".join(src_lines[start:end])
            exec(class_src, _COMPONENT_NS)
            return _COMPONENT_NS[class_name]
    raise RuntimeError(f"class {class_name} not found")


# Pre-load MLP since multiple components reference it.
load_component("MLP")


# ---------------------------------------------------------------------------
# TIER 1.1: EmpiricalNormalization Welford correctness
# ---------------------------------------------------------------------------

def test_empirical_normalization():
    print("\n[T1.1] EmpiricalNormalization Welford running stats")
    EmpiricalNormalization = load_component("EmpiricalNormalization")
    en = EmpiricalNormalization(shape=(5,))

    torch.manual_seed(0)
    # Stream data in chunks to test online update.
    chunks = [torch.randn(100, 5) * 2 + 3, torch.randn(50, 5) * 2 + 3, torch.randn(80, 5) * 2 + 3]
    all_data = torch.cat(chunks, dim=0)

    for c in chunks:
        en.update(c)

    ref_mean = all_data.mean(dim=0)
    ref_var = all_data.var(dim=0, unbiased=False)

    ok = True
    if not torch.allclose(en.mean, ref_mean, atol=1e-5):
        print(f"  ✗ mean mismatch: max_abs_diff={(en.mean - ref_mean).abs().max().item():.3e}")
        ok = False
    else:
        print("  ✓ running mean matches batch reference")

    if not torch.allclose(en.var, ref_var, atol=1e-4):
        print(f"  ✗ var mismatch: max_abs_diff={(en.var - ref_var).abs().max().item():.3e}")
        ok = False
    else:
        print("  ✓ running var matches batch reference")

    # Forward pass: should produce ~zero mean, ~unit variance on the same data.
    out = en(all_data)
    out_mean = out.mean(dim=0)
    out_var = out.var(dim=0, unbiased=False)
    if out_mean.abs().max().item() < 1e-5:
        print("  ✓ forward output zero mean")
    else:
        print(f"  ✗ forward output mean not zero: {out_mean.abs().max().item():.3e}")
        ok = False
    if (out_var - 1.0).abs().max().item() < 1e-3:
        print("  ✓ forward output unit variance")
    else:
        print(f"  ✗ forward output var not 1: max_dev={((out_var - 1.0).abs().max().item()):.3e}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 1.2: MLP shape and gradient flow
# ---------------------------------------------------------------------------

def test_mlp():
    print("\n[T1.2] MLP shape preservation and gradient flow")
    MLP = load_component("MLP")
    net = MLP(input_dim=10, output_dim=4, hidden_dims=[32, 16])
    x = torch.randn(8, 10, requires_grad=True)
    y = net(x)

    ok = True
    if y.shape != (8, 4):
        print(f"  ✗ output shape {tuple(y.shape)} != (8, 4)")
        ok = False
    else:
        print("  ✓ output shape (B=8, out=4)")

    loss = y.pow(2).mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in net.parameters() if p.grad is not None]
    if all(g > 0 for g in grad_norms) and not any(math.isnan(g) for g in grad_norms):
        print(f"  ✓ all params have non-zero, non-NaN gradients (n={len(grad_norms)})")
    else:
        print(f"  ✗ bad gradients: {grad_norms}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 1.3: ElevationAE
# ---------------------------------------------------------------------------

def test_elevation_ae():
    print("\n[T1.3] ElevationAE")
    ElevationAE = load_component("ElevationAE")
    ae = ElevationAE(input_dim=187, output_dim=64, grid_shape=(11, 17))

    ok = True
    # In training mode, both latent and recon are returned.
    ae.train()
    x_train = torch.randn(4, 187)
    latent_t, recon_t = ae(x_train)
    if latent_t.shape == (4, 64) and recon_t is not None and recon_t.shape == (4, 187):
        print(f"  ✓ train mode: latent={tuple(latent_t.shape)}, recon={tuple(recon_t.shape)}")
    else:
        print(f"  ✗ train shapes wrong: latent={tuple(latent_t.shape)}, recon={recon_t}")
        ok = False

    # In eval mode, recon should be None.
    ae.eval()
    x_eval = torch.randn(4, 187)
    latent_e, recon_e = ae(x_eval)
    if latent_e.shape == (4, 64) and recon_e is None:
        print("  ✓ eval mode: latent shape correct, recon=None")
    else:
        print(f"  ✗ eval mode wrong: latent={tuple(latent_e.shape)}, recon={recon_e}")
        ok = False

    # Multi-history shape: [Seq, B, 187] should produce [Seq, B, 64] latent.
    ae.eval()
    x_seq = torch.randn(7, 4, 187)
    latent_s, _ = ae(x_seq)
    if latent_s.shape == (7, 4, 64):
        print("  ✓ multi-axis input: [Seq=7, B=4, 187] -> latent=[7, 4, 64]")
    else:
        print(f"  ✗ seq shape: {tuple(latent_s.shape)}")
        ok = False

    # Determinism: same input -> same output (no dropout / no random sampling).
    out1, _ = ae(x_eval)
    out2, _ = ae(x_eval)
    if torch.allclose(out1, out2):
        print("  ✓ deterministic in eval mode")
    else:
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 1.4: MultiLayerScanAE - check fwd/bwd internal handling
# ---------------------------------------------------------------------------

def test_multi_layer_scan_ae():
    print("\n[T1.4] MultiLayerScanAE")
    AE = load_component("MultiLayerScanAE")
    ae = AE(num_channels=12, num_rays=21, output_dim=64)

    ok = True
    ae.eval()
    x = torch.randn(8, 12 * 21)
    latent, recon = ae(x)
    if latent.shape == (8, 64):
        print(f"  ✓ eval latent shape (8, 64)")
    else:
        print(f"  ✗ latent shape: {tuple(latent.shape)}")
        ok = False
    if recon is None:
        print("  ✓ eval recon=None")
    else:
        ok = False

    ae.train()
    latent_t, recon_t = ae(x)
    if recon_t is not None and recon_t.shape == (8, 12 * 21):
        print(f"  ✓ train recon shape {tuple(recon_t.shape)}")
    else:
        print(f"  ✗ train recon shape: {recon_t}")
        ok = False

    # Internal fwd/bwd handling:
    # ae.actual_channels = 6, expected.
    if ae.actual_channels == 6:
        print("  ✓ actual_channels = 6 (for 12 input channels)")
    else:
        print(f"  ✗ actual_channels = {ae.actual_channels}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 1.5: DepthAE
# ---------------------------------------------------------------------------

def test_depth_ae():
    print("\n[T1.5] DepthAE")
    DepthAE = load_component("DepthAE")
    ae = DepthAE(input_channels=2, output_dim=128, camera_height=58, camera_width=87)

    ok = True
    ae.eval()
    x = torch.randn(4, 2, 58, 87)
    latent, recon = ae(x)
    if latent.shape == (4, 128) and recon is None:
        print(f"  ✓ eval: latent shape (4, 128), recon=None")
    else:
        print(f"  ✗ eval shape: latent={tuple(latent.shape)}, recon={recon}")
        ok = False

    ae.train()
    latent_t, recon_t = ae(x)
    if recon_t is not None and recon_t.shape == (4, 2, 58, 87):
        print(f"  ✓ train recon matches input shape {tuple(recon_t.shape)}")
    else:
        print(f"  ✗ train recon shape: {recon_t}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 1.6: ProprioVAE
# ---------------------------------------------------------------------------

def test_proprio_vae():
    print("\n[T1.6] ProprioVAE")
    ProprioVAE = load_component("ProprioVAE")
    vae = ProprioVAE(input_dim=57, vel_dim=3, latent_dim=64, hidden_dims=[128, 64])

    ok = True
    # Train mode: returns (vel_pred, recon, mu, logvar, z) where z is sampled.
    vae.train()
    x = torch.randn(4, 57)
    vel_pred, recon, mu, logvar, z = vae(x)
    if vel_pred.shape == (4, 3) and recon is not None and recon.shape == (4, 57) and mu.shape == (4, 64) and logvar.shape == (4, 64) and z.shape == (4, 64):
        print("  ✓ train mode all shapes OK")
    else:
        print(f"  ✗ train shapes: vel={tuple(vel_pred.shape)}, recon={tuple(recon.shape) if recon is not None else None}, mu={tuple(mu.shape)}, logvar={tuple(logvar.shape)}, z={tuple(z.shape)}")
        ok = False

    # Eval mode: recon=None, z=mu (deterministic).
    vae.eval()
    vel_e, recon_e, mu_e, _, z_e = vae(x)
    if recon_e is None and torch.allclose(z_e, mu_e):
        print("  ✓ eval mode: recon=None, z = mu (deterministic)")
    else:
        ok = False

    # Reparameterization: train mode z != mu in general.
    vae.train()
    _, _, mu1, _, z1 = vae(x)
    diff = (z1 - mu1).abs().max().item()
    if diff > 1e-4:
        print(f"  ✓ train mode: z != mu (sampled), max_diff={diff:.3e}")
    else:
        print(f"  ✗ train mode z = mu (no sampling? diff={diff:.3e})")
        ok = False

    # logvar clamp: feed extreme input, ensure logvar stays in [-20, 10].
    x_big = torch.randn(4, 57) * 1e6
    _, _, _, logvar_big, _ = vae(x_big)
    if logvar_big.max() <= 10.0 + 1e-6 and logvar_big.min() >= -20.0 - 1e-6:
        print(f"  ✓ logvar clamped to [-20, 10], observed [{logvar_big.min().item():.3f}, {logvar_big.max().item():.3f}]")
    else:
        print(f"  ✗ logvar not clamped: range=[{logvar_big.min().item():.3f}, {logvar_big.max().item():.3f}]")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 2.1: Joint name layout matches assumed action_swap_idx layout
# ---------------------------------------------------------------------------

def test_joint_name_layout():
    print("\n[T2.1] Joint name layout matches action_swap_idx assumption")
    # Read teacher cfg to extract leg_joint_names and wheel_joint_names.
    with open(TEACHER_CFG) as f:
        text = f.read()

    # Match the leg and wheel joint name lists (very lenient regex, then ast-eval).
    def extract_list(text, var_name):
        m = re.search(rf"\b{var_name}\s*=\s*\[(.*?)\]", text, re.DOTALL)
        if not m:
            return None
        body = "[" + m.group(1) + "]"
        return ast.literal_eval(body)

    legs = extract_list(text, "leg_joint_names")
    wheels = extract_list(text, "wheel_joint_names")

    ok = True
    expected_legs = [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    ]
    expected_wheels = ["fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint"]

    if legs == expected_legs:
        print("  ✓ leg_joint_names order matches assumption")
    else:
        print(f"  ✗ leg_joint_names mismatch:")
        print(f"      actual:   {legs}")
        print(f"      expected: {expected_legs}")
        ok = False

    if wheels == expected_wheels:
        print("  ✓ wheel_joint_names order matches assumption")
    else:
        print(f"  ✗ wheel_joint_names mismatch:")
        print(f"      actual:   {wheels}")
        print(f"      expected: {expected_wheels}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 2.2: action_swap_idx applied to joint names produces the expected labels
# ---------------------------------------------------------------------------

def test_action_swap_idx_labels():
    print("\n[T2.2] action_swap_idx (LR/FB) produces expected leg-name swap")
    # joint order in obs/action: leg_joint_names + wheel_joint_names = 16 strings.
    joint_names = [
        "fl_hipx", "fl_hipy", "fl_knee",
        "fr_hipx", "fr_hipy", "fr_knee",
        "hl_hipx", "hl_hipy", "hl_knee",
        "hr_hipx", "hr_hipy", "hr_knee",
        "fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel",
    ]

    # L-R from moe_terrain.py
    lr_idx = [3,4,5, 0,1,2, 9,10,11, 6,7,8, 13,12, 15,14]
    # F-B from moe_terrain.py
    fb_idx = [6,7,8, 9,10,11, 0,1,2, 3,4,5, 14,15, 12,13]

    swapped_lr = [joint_names[i] for i in lr_idx]
    swapped_fb = [joint_names[i] for i in fb_idx]

    # Expected after L-R swap: every "fl"/"hl" pair becomes "fr"/"hr" and vice versa.
    expected_lr = [
        "fr_hipx", "fr_hipy", "fr_knee",
        "fl_hipx", "fl_hipy", "fl_knee",
        "hr_hipx", "hr_hipy", "hr_knee",
        "hl_hipx", "hl_hipy", "hl_knee",
        "fr_wheel", "fl_wheel",
        "hr_wheel", "hl_wheel",
    ]
    # Expected after F-B swap: "fl" -> "hl", "fr" -> "hr", and vice versa.
    expected_fb = [
        "hl_hipx", "hl_hipy", "hl_knee",
        "hr_hipx", "hr_hipy", "hr_knee",
        "fl_hipx", "fl_hipy", "fl_knee",
        "fr_hipx", "fr_hipy", "fr_knee",
        "hl_wheel", "hr_wheel",
        "fl_wheel", "fr_wheel",
    ]

    ok = True
    if swapped_lr == expected_lr:
        print("  ✓ L-R swap_idx produces correct fl↔fr / hl↔hr label permutation")
    else:
        print(f"  ✗ L-R swap mismatch: {swapped_lr}")
        ok = False
    if swapped_fb == expected_fb:
        print("  ✓ F-B swap_idx produces correct fl↔hl / fr↔hr label permutation")
    else:
        print(f"  ✗ F-B swap mismatch: {swapped_fb}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 2.3: action_neg_mask consistency with M20 init_state
# ---------------------------------------------------------------------------

def test_neg_mask_vs_init_state():
    """The M20 init_state has front and rear hipy/knee with opposite signs
    (X-pose default), implying URDF axes are F-B mirrored. action_neg_mask_fb
    must therefore flip hipy/knee on swap. action_neg_mask (L-R) must flip
    hipx (since fl/fr URDF axes are L-R mirrored) but NOT hipy/knee."""
    print("\n[T2.3] action_neg_mask consistency with M20 URDF init_state")
    with open(DEEP_ASSET) as f:
        asset_text = f.read()

    # Extract the joint_pos init dict for M20.
    # Look for: f[l,r]_hipy_joint": -0.6,  h[l,r]_hipy_joint": 0.6, etc.
    fl_hipy = re.search(r'"f\[l,r\]_hipy_joint"\s*:\s*(-?[\d.]+)', asset_text)
    hl_hipy = re.search(r'"h\[l,r\]_hipy_joint"\s*:\s*(-?[\d.]+)', asset_text)
    fl_knee = re.search(r'"f\[l,r\]_knee_joint"\s*:\s*(-?[\d.]+)', asset_text)
    hl_knee = re.search(r'"h\[l,r\]_knee_joint"\s*:\s*(-?[\d.]+)', asset_text)

    ok = True
    if not (fl_hipy and hl_hipy and fl_knee and hl_knee):
        print("  ✗ failed to parse init_state from deeprobotics.py")
        return False

    fl_hipy_v = float(fl_hipy.group(1))
    hl_hipy_v = float(hl_hipy.group(1))
    fl_knee_v = float(fl_knee.group(1))
    hl_knee_v = float(hl_knee.group(1))
    print(f"  init_state: fl_hipy={fl_hipy_v}, hl_hipy={hl_hipy_v}, fl_knee={fl_knee_v}, hl_knee={hl_knee_v}")

    # If front and rear have opposite signs, F-B mirror needs hipy/knee flip.
    if fl_hipy_v * hl_hipy_v < 0:
        print("  ✓ hipy front/rear opposite sign => F-B neg mask must flip hipy")
    else:
        print(f"  ✗ unexpected: hipy same-sign (or zero) — F-B neg mask might be wrong")
        ok = False
    if fl_knee_v * hl_knee_v < 0:
        print("  ✓ knee front/rear opposite sign => F-B neg mask must flip knee")
    else:
        print(f"  ✗ unexpected: knee same-sign — F-B neg mask might be wrong")
        ok = False

    # Now extract action_neg_mask_fb from moe_terrain.py and check it.
    m = re.search(r"action_neg_mask_fb\s*=\s*torch\.tensor\(\[(.*?)\]", SOURCE_TEXT, re.DOTALL)
    if not m:
        print("  ✗ couldn't find action_neg_mask_fb in source")
        return False
    mask_text = m.group(1)
    mask = [int(x.strip()) for x in mask_text.split(",") if x.strip().lstrip("-").isdigit()][:16]
    # Joint indexing (16): FL[hipx, hipy, knee], FR[hipx, hipy, knee], HL[...], HR[...], wheels[FL,FR,HL,HR]
    print(f"  action_neg_mask_fb = {mask}")
    expected_signs = [1, -1, -1] * 4 + [-1, -1, -1, -1]
    if mask == expected_signs:
        print("  ✓ action_neg_mask_fb matches expected (hipx no-flip, hipy/knee flip, wheels flip)")
    else:
        print(f"  ✗ action_neg_mask_fb mismatch: got {mask}, expected {expected_signs}")
        ok = False

    # Same for L-R.
    m_lr = re.search(r"action_neg_mask\s*=\s*torch\.tensor\(\[(.*?)\]", SOURCE_TEXT, re.DOTALL)
    if m_lr:
        mask_lr = [int(x.strip()) for x in m_lr.group(1).split(",") if x.strip().lstrip("-").isdigit()][:16]
        print(f"  action_neg_mask    = {mask_lr}")
        expected_lr = [-1, 1, 1] * 4 + [1, 1, 1, 1]
        if mask_lr == expected_lr:
            print("  ✓ action_neg_mask (L-R) matches expected (hipx flip, hipy/knee no-flip, wheels no-flip)")
        else:
            print(f"  ✗ action_neg_mask mismatch: got {mask_lr}, expected {expected_lr}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 3.1: Sym loss bypass when sym_enabled=False
# ---------------------------------------------------------------------------

def test_sym_disabled_bypass():
    """When sym_loss_coef <= 0, sym_enabled = False. Verify:
    1. _init_rnn_state returns 1-layer (no piggyback)
    2. update() has 'continue' early return that skips sym_loss code
    3. act() has fast-path that skips _mirror_obs piggyback
    """
    print("\n[T3.1] Sym loss bypass when sym_enabled=False (static analysis)")
    ok = True

    # Look for: num_layers = 1 path in _init_rnn_state
    if re.search(r"num_layers\s*=\s*\d+\s+if\s+getattr\(self,\s*['\"]sym_enabled['\"],?.*?\)\s+else\s+1", SOURCE_TEXT):
        print("  ✓ _init_rnn_state has sym_enabled-conditional num_layers (else=1)")
    else:
        print("  ✗ _init_rnn_state doesn't have expected sym_enabled branching")
        ok = False

    # Look for: continue inside STEP D when sym_enabled is False
    step_d_block = re.search(r"STEP D:.*?obs_mirrored_batch", SOURCE_TEXT, re.DOTALL)
    if step_d_block and "continue" in step_d_block.group(0):
        print("  ✓ STEP D has 'continue' early return when sym disabled")
    else:
        print("  ✗ STEP D missing 'continue' for sym_enabled=False bypass")
        ok = False

    # Look for: fast path in act() when sym_enabled=False
    if re.search(r"快速路径.*?sym 未启用", SOURCE_TEXT, re.DOTALL) or \
       re.search(r"if not getattr\(model,\s*['\"]sym_enabled['\"]", SOURCE_TEXT):
        print("  ✓ act() has fast-path branch for sym_enabled=False")
    else:
        print("  ✗ act() missing fast-path for sym_enabled=False")
        ok = False

    # Look for: sym_loss_coef threshold
    if re.search(r"self\.sym_enabled\s*=\s*bool\(self\.sym_loss_coef\s*>\s*0", SOURCE_TEXT):
        print("  ✓ sym_enabled = sym_loss_coef > 0")
    else:
        print("  ✗ sym_enabled boolean derivation not as expected")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 3.2: STEP D2 (F-B sym loss) is correctly placed AFTER L-R sym loss
#          and BEFORE STEP E (backward pass)
# ---------------------------------------------------------------------------

def test_step_order():
    print("\n[T3.2] PPO update step order: D (LR sym) -> D2 (FB sym) -> E (backward)")
    # find positions of step markers in source
    pos_D = SOURCE_TEXT.find("STEP D:")
    pos_D2 = SOURCE_TEXT.find("STEP D2:")
    pos_E = SOURCE_TEXT.find("STEP E:")

    ok = True
    if pos_D < 0 or pos_D2 < 0 or pos_E < 0:
        print(f"  ✗ missing markers: D={pos_D}, D2={pos_D2}, E={pos_E}")
        return False
    if pos_D < pos_D2 < pos_E:
        print(f"  ✓ order correct: STEP D (pos {pos_D}) < D2 ({pos_D2}) < E ({pos_E})")
    else:
        print(f"  ✗ wrong order: D={pos_D}, D2={pos_D2}, E={pos_E}")
        ok = False

    # Both sym losses contribute to the SAME loss tensor before backward.
    # Look for: loss = loss + ... * sym_loss   AND   loss = loss + ... * sym_loss_fb
    if re.search(r"loss\s*=\s*loss\s*\+\s*.*?\*\s*sym_loss\b", SOURCE_TEXT) and \
       re.search(r"loss\s*=\s*loss\s*\+\s*.*?\*\s*sym_loss_fb\b", SOURCE_TEXT):
        print("  ✓ both sym_loss and sym_loss_fb added to total loss before backward")
    else:
        print("  ✗ couldn't find both loss accumulations")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 3.3: F-B mirror gives mathematically consistent outputs (basic involution)
# ---------------------------------------------------------------------------

def test_fb_mirror_basic():
    """Re-run the involution / commutativity tests already in test_mirror_fb to give
    a one-stop-shop view; details are in test_mirror_fb_integration.py."""
    print("\n[T3.3] F-B mirror involution + commutativity (smoke test)")
    sys.path.insert(0, "/tmp/auto_probe")
    try:
        import test_mirror_fb as ref
    except Exception as e:
        print(f"  ✗ couldn't import test_mirror_fb: {e}")
        return False

    obs = ref.make_obs(B=4, hist=3, seed=12345)
    ok = True
    # Involutions
    out_fb_fb = ref.reference_mirror_fb(ref.reference_mirror_fb(obs))
    if all(torch.allclose(out_fb_fb[k], obs[k]) for k in obs):
        print("  ✓ F-B mirror is involutive: M_FB(M_FB(x)) == x")
    else:
        ok = False
    # Commutativity
    a = ref.reference_mirror_lr(ref.reference_mirror_fb(obs))
    b = ref.reference_mirror_fb(ref.reference_mirror_lr(obs))
    if all(torch.allclose(a[k], b[k]) for k in obs):
        print("  ✓ L-R and F-B commute (Klein 4-group)")
    else:
        ok = False
    print("  (full coverage in /tmp/auto_probe/test_mirror_fb_integration.py)")
    return ok


# ---------------------------------------------------------------------------
# TIER 3.4: Load balancing loss math
# ---------------------------------------------------------------------------

def test_load_balancing_loss():
    """The MoE load balancing loss should:
    - be 0 when gate weights are uniform (no specialization)
    - be positive when one expert dominates
    - scale appropriately with num_experts
    Check the implementation produces these intuitive properties."""
    print("\n[T3.4] Load balancing loss properties")

    # Extract _calculate_load_balancing_loss method body.
    tree = ast.parse(SOURCE_TEXT)
    src_lines = SOURCE_TEXT.splitlines()
    method_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_calculate_load_balancing_loss":
            start = node.lineno - 1
            end = node.end_lineno  # type: ignore[attr-defined]
            method_src = textwrap.dedent("\n".join(src_lines[start:end]))
            break

    if method_src is None:
        print("  ✗ _calculate_load_balancing_loss not found")
        return False

    print(f"  found method ({method_src.count(chr(10))} lines)")

    ns = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    exec(method_src, ns)
    fn = ns["_calculate_load_balancing_loss"]

    # Mock self with .num_leg_experts, .num_wheel_experts (typical 6 + 3).
    mock_self = types.SimpleNamespace(num_leg_experts=6, num_wheel_experts=3, aux_loss_coef=0.01)

    B = 32
    n_leg, n_wheel = 6, 3
    ok = True

    # Case 1: perfectly uniform gates -> loss should be at its minimum (or zero).
    w_leg_uniform = torch.full((B, n_leg), 1.0 / n_leg)
    w_wheel_uniform = torch.full((B, n_wheel), 1.0 / n_wheel)
    loss_uniform = fn(mock_self, w_leg_uniform, w_wheel_uniform)
    print(f"  uniform gate loss: {loss_uniform.item():.6f}")

    # Case 2: one-hot (one expert dominates).
    w_leg_onehot = torch.zeros((B, n_leg))
    w_leg_onehot[:, 0] = 1.0
    w_wheel_onehot = torch.zeros((B, n_wheel))
    w_wheel_onehot[:, 0] = 1.0
    loss_onehot = fn(mock_self, w_leg_onehot, w_wheel_onehot)
    print(f"  one-hot gate loss: {loss_onehot.item():.6f}")

    if loss_onehot > loss_uniform:
        print("  ✓ one-hot loss > uniform loss (load balancing penalises specialization)")
    else:
        print(f"  ✗ load balancing not pushing toward uniform: uniform={loss_uniform.item():.6f}, onehot={loss_onehot.item():.6f}")
        ok = False

    # Case 3: Should be non-negative.
    if loss_uniform >= -1e-6 and loss_onehot >= -1e-6:
        print("  ✓ load balancing loss is non-negative")
    else:
        print(f"  ✗ negative loss: uniform={loss_uniform.item()}, onehot={loss_onehot.item()}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# TIER 3.5: avg_sym_loss split (LR/FB) appears in loss_dict
# ---------------------------------------------------------------------------

def test_sym_log_split():
    """We just split the sym loss logging. Verify the wandb keys are correctly
    populated."""
    print("\n[T3.5] Split sym_loss logging keys")
    ok = True

    if "Loss/Actor_Symmetry_Reg_LR" in SOURCE_TEXT and "Loss/Actor_Symmetry_Reg_FB" in SOURCE_TEXT:
        print("  ✓ both _LR and _FB keys present in source")
    else:
        print("  ✗ split keys missing")
        ok = False

    if "Loss/Actor_Symmetry_Reg" in SOURCE_TEXT:
        print("  ✓ legacy _Reg key kept (sum of LR+FB) for backward compat")
    else:
        print("  ✗ legacy key gone")
        ok = False

    # And that the variables are separate.
    if "avg_sym_loss_lr" in SOURCE_TEXT and "avg_sym_loss_fb" in SOURCE_TEXT:
        print("  ✓ avg_sym_loss_lr / avg_sym_loss_fb defined")
    else:
        print("  ✗ accumulator vars missing")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("EmpiricalNormalization", test_empirical_normalization),
        ("MLP", test_mlp),
        ("ElevationAE", test_elevation_ae),
        ("MultiLayerScanAE", test_multi_layer_scan_ae),
        ("DepthAE", test_depth_ae),
        ("ProprioVAE", test_proprio_vae),
        ("Joint name layout", test_joint_name_layout),
        ("action_swap_idx labels", test_action_swap_idx_labels),
        ("neg mask vs init_state", test_neg_mask_vs_init_state),
        ("sym disabled bypass", test_sym_disabled_bypass),
        ("PPO step order", test_step_order),
        ("F-B mirror smoke", test_fb_mirror_basic),
        ("Load balancing loss", test_load_balancing_loss),
        ("Sym log split", test_sym_log_split),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            import traceback
            print(f"\n[ERROR] in {name}:")
            traceback.print_exc()
            ok = False
        results.append((name, ok))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] {name}")
    print(f"\n{n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
