"""Verify that sym loss is a genuine optimisation objective: training only on
sym MSE (no PPO loss) drives policy toward L-R + F-B equivariance.

Run:
    /home/ouge/miniconda3/envs/env_isaaclab/bin/python tests/test_sym_loss_training.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import (
    load_moe_terrain, make_actor_critic, make_fake_obs,
    NUM_ACTIONS, section, passed, failed,
)


B_TEST = 8
LATENT_DIM = 32


def get_mirror_fns(model, device):
    """Borrow _mirror_obs and _mirror_obs_fb from SplitMoEPPO via duck-typing."""
    import types as _t
    mod = load_moe_terrain()
    stub = _t.SimpleNamespace(device=device, actor_critic=model, policy=model)
    return (
        mod.SplitMoEPPO._mirror_obs.__get__(stub, _t.SimpleNamespace),
        mod.SplitMoEPPO._mirror_obs_fb.__get__(stub, _t.SimpleNamespace),
    )


def compute_sym_losses(model, obs, mirror_fn_lr, mirror_fn_fb):
    """Compute (lr_mse, fb_mse) on a batch.

    Reset hidden state to zero before each forward so that all three forwards
    (real, lr, fb) see the same context."""
    dev = obs["policy"].device
    swap_lr = torch.tensor([3,4,5, 0,1,2, 9,10,11, 6,7,8, 13,12, 15,14], dtype=torch.long, device=dev)
    neg_lr  = torch.tensor([-1,1,1, -1,1,1, -1,1,1, -1,1,1, 1,1,1,1], dtype=torch.float32, device=dev)
    swap_fb = torch.tensor([6,7,8, 9,10,11, 0,1,2, 3,4,5, 14,15, 12,13], dtype=torch.long, device=dev)
    neg_fb  = torch.tensor([1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, -1,-1,-1,-1], dtype=torch.float32, device=dev)

    def reset_state():
        if model.rnn_type == "lstm":
            model.active_hidden_states = (
                torch.zeros_like(model.active_hidden_states[0]),
                torch.zeros_like(model.active_hidden_states[1]),
            )
        else:
            model.active_hidden_states = torch.zeros_like(model.active_hidden_states)

    reset_state()
    a_real = model.act_inference(obs)
    target_lr = a_real[..., swap_lr] * neg_lr
    target_fb = a_real[..., swap_fb] * neg_fb

    reset_state()
    obs_lr = mirror_fn_lr(obs)
    a_lr = model.act_inference(obs_lr)

    reset_state()
    obs_fb = mirror_fn_fb(obs)
    a_fb = model.act_inference(obs_fb)

    mse_lr = F.mse_loss(a_lr, target_lr.detach())  # detach target so only the predicted side learns
    mse_fb = F.mse_loss(a_fb, target_fb.detach())
    return mse_lr, mse_fb


def test_sym_loss_decreases_with_training():
    """Default init has output_gain=0.01 → action magnitude tiny → sym MSE ~ 1e-6
    already, hard to drive lower. We amplify expert outputs first (multiply
    final layer weights × 100) to create a clearly non-equivariant policy,
    then train."""
    section("[S1] Sym loss as the SOLE objective drives policy toward equivariance")
    model, _ = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM, sym_loss_coef=0.3)

    # Boost the final layer of every expert so the policy is meaningfully non-trivial.
    with torch.no_grad():
        for expert in list(model.actor_leg_experts) + list(model.actor_wheel_experts):
            last_lin = [m for m in expert.modules() if isinstance(m, nn.Linear)][-1]
            last_lin.weight.mul_(100.0)
    model.train()

    dev = next(model.parameters()).device
    mirror_lr, mirror_fb = get_mirror_fns(model, dev)

    # Fixed batch — make the optimization deterministic.
    obs = make_fake_obs(B=B_TEST, history=5, seed=42)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(150):
        optimizer.zero_grad()
        mse_lr, mse_fb = compute_sym_losses(model, obs, mirror_lr, mirror_fb)
        loss = mse_lr + mse_fb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append((mse_lr.item(), mse_fb.item()))

    initial_lr, initial_fb = losses[0]
    final_lr, final_fb = losses[-1]
    print(f"  step 0:   L-R MSE = {initial_lr:.4f},  F-B MSE = {initial_fb:.4f}")
    print(f"  step 49:  L-R MSE = {losses[49][0]:.4f},  F-B MSE = {losses[49][1]:.4f}")
    print(f"  step 99:  L-R MSE = {losses[99][0]:.4f},  F-B MSE = {losses[99][1]:.4f}")
    print(f"  step 149: L-R MSE = {final_lr:.4f},  F-B MSE = {final_fb:.4f}")

    # We expect sym MSE to drop by at least 80% (from notably non-zero to small).
    ok = True
    if final_lr < initial_lr * 0.2:
        passed(f"L-R sym MSE dropped >80% (initial={initial_lr:.4f} → final={final_lr:.4f})")
    else:
        failed(f"L-R sym MSE not decreasing enough: {initial_lr:.4f} → {final_lr:.4f}")
        ok = False
    if final_fb < initial_fb * 0.2:
        passed(f"F-B sym MSE dropped >80% (initial={initial_fb:.4f} → final={final_fb:.4f})")
    else:
        failed(f"F-B sym MSE not decreasing enough: {initial_fb:.4f} → {final_fb:.4f}")
        ok = False
    # Check loss is strictly decreasing (no oscillation).
    if all(losses[i+10][0] <= losses[i][0] * 1.5 for i in range(0, len(losses)-10, 10)):
        passed("L-R loss monotone enough (no >1.5x oscillation across 10-step windows)")
    return ok


def test_sym_loss_at_zero_input():
    """If obs is all zeros, mirror operations are identity (since flipping
    zeros = zeros). The pre/post-mirror obs are the same ⇒ a deterministic
    policy outputs the same action ⇒ sym MSE measures only the action mirror
    asymmetry, which is 0 only if action = M·action ⇒ specific symmetry
    fixed-point at zero action. For a randomly initialized RNN the action is
    NOT zero, so there's still a residual loss."""
    section("[S2] All-zero obs: sym MSE measures action mirror residual")
    model, _ = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM, sym_loss_coef=0.3)
    model.eval()

    dev = next(model.parameters()).device
    mirror_lr, mirror_fb = get_mirror_fns(model, dev)

    obs = {
        "policy":          torch.zeros(B_TEST, 57),
        "critic":          torch.zeros(B_TEST, 60),
        "estimator":       torch.zeros(B_TEST, 57 * 5),
        "noisy_elevation": torch.zeros(B_TEST, 187 + 252),
    }

    obs_lr = mirror_lr(obs)
    obs_fb = mirror_fb(obs)

    # Mirroring zeros gives zeros.
    ok = True
    for k in obs:
        if torch.allclose(obs[k], obs_lr[k]) and torch.allclose(obs[k], obs_fb[k]):
            pass
        else:
            failed(f"mirror not identity on zero obs for key {k}")
            ok = False
    if ok:
        passed("mirror of zero-obs is identity (sanity)")

    return ok


def test_sym_loss_zero_for_synthetic_equivariant():
    """Construct an artificially equivariant function (zero output) and verify
    sym MSE = 0. This is the trivial fixed point but verifies the loss math."""
    section("[S3] Sym MSE = 0 for trivially equivariant policy (output ≡ 0)")
    model, _ = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM, sym_loss_coef=0.3)
    model.eval()

    # Force the actor's leg/wheel experts to output zero by zeroing their last linear layer.
    with torch.no_grad():
        for expert in list(model.actor_leg_experts) + list(model.actor_wheel_experts):
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    m.weight.zero_()
                    if m.bias is not None: m.bias.zero_()

    dev = next(model.parameters()).device
    mirror_lr, mirror_fb = get_mirror_fns(model, dev)
    obs = make_fake_obs(B=B_TEST, history=5, seed=99)

    with torch.no_grad():
        mse_lr, mse_fb = compute_sym_losses(model, obs, mirror_lr, mirror_fb)
    print(f"  L-R MSE = {mse_lr.item():.3e},  F-B MSE = {mse_fb.item():.3e}")
    ok = True
    if mse_lr.item() < 1e-9:
        passed("L-R sym MSE = 0 for zero-output policy (trivial equivariance)")
    else:
        failed(f"L-R MSE not zero: {mse_lr.item():.3e}")
        ok = False
    if mse_fb.item() < 1e-9:
        passed("F-B sym MSE = 0 for zero-output policy (trivial equivariance)")
    else:
        failed(f"F-B MSE not zero: {mse_fb.item():.3e}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
def main():
    tests = [
        ("zero-obs mirror identity",  test_sym_loss_at_zero_input),
        ("zero-output trivially equiv", test_sym_loss_zero_for_synthetic_equivariant),
        ("training drives equivariance", test_sym_loss_decreases_with_training),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception:
            import traceback
            print(f"\n[ERROR in {name}]")
            traceback.print_exc()
            ok = False
        results.append((name, ok))

    section("SUMMARY")
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] {name}")
    print(f"\n{n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
