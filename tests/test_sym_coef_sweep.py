"""Sweep sym_loss_coef values to inform default selection.

We add a synthetic "PPO-like" pseudo-loss that competes with the sym loss
(reconstruction of a target action), then measure how each coefficient affects:

  - How fast sym MSE decreases
  - How well the pseudo-PPO target is tracked
  - Whether gradients explode or get unstable

This is a heuristic sweep, not a full PPO training. The point is to inform
a reasonable default coefficient — the actual best value depends on the live
RL signal.

Run:
    /home/ouge/miniconda3/envs/env_isaaclab/bin/python tests/test_sym_coef_sweep.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import (
    load_moe_terrain, make_actor_critic, make_fake_obs,
    NUM_ACTIONS, section, passed,
)


def get_mirror_fns(model, device):
    import types as _t
    mod = load_moe_terrain()
    stub = _t.SimpleNamespace(device=device, actor_critic=model, policy=model)
    return (
        mod.SplitMoEPPO._mirror_obs.__get__(stub, _t.SimpleNamespace),
        mod.SplitMoEPPO._mirror_obs_fb.__get__(stub, _t.SimpleNamespace),
    )


def reset_state(model):
    if model.rnn_type == "lstm":
        model.active_hidden_states = (
            torch.zeros_like(model.active_hidden_states[0]),
            torch.zeros_like(model.active_hidden_states[1]),
        )
    else:
        model.active_hidden_states = torch.zeros_like(model.active_hidden_states)


def amplify_experts(model, factor=50.0):
    """Boost expert outputs to make initial non-equivariance non-trivial."""
    with torch.no_grad():
        for expert in list(model.actor_leg_experts) + list(model.actor_wheel_experts):
            last_lin = [m for m in expert.modules() if isinstance(m, nn.Linear)][-1]
            last_lin.weight.mul_(factor)


def run_one_sweep(coef_lr: float, coef_fb: float, n_steps: int = 100, seed: int = 0):
    """Run a synthetic mixed-objective training at given (coef_lr, coef_fb)."""
    torch.manual_seed(seed)
    model, _ = make_actor_critic(B=8, latent_dim=32, sym_loss_coef=0.3, seed=seed)
    amplify_experts(model)
    model.train()

    dev = next(model.parameters()).device
    mirror_lr, mirror_fb = get_mirror_fns(model, dev)

    swap_lr = torch.tensor([3,4,5, 0,1,2, 9,10,11, 6,7,8, 13,12, 15,14], dtype=torch.long, device=dev)
    neg_lr  = torch.tensor([-1,1,1, -1,1,1, -1,1,1, -1,1,1, 1,1,1,1], dtype=torch.float32, device=dev)
    swap_fb = torch.tensor([6,7,8, 9,10,11, 0,1,2, 3,4,5, 14,15, 12,13], dtype=torch.long, device=dev)
    neg_fb  = torch.tensor([1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, -1,-1,-1,-1], dtype=torch.float32, device=dev)

    # Synthetic PPO-like target action (random fixed) — drives policy to track it.
    target_action = torch.randn(8, NUM_ACTIONS, device=dev)
    obs = make_fake_obs(B=8, history=5, seed=seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"step": [], "task_loss": [], "lr_mse": [], "fb_mse": [], "grad_norm": []}

    for step in range(n_steps):
        optimizer.zero_grad()

        # Task: track target (pseudo PPO objective)
        reset_state(model)
        a_real = model.act_inference(obs)
        task_loss = F.mse_loss(a_real, target_action)

        # Sym losses
        target_lr = a_real.detach()[..., swap_lr] * neg_lr
        target_fb = a_real.detach()[..., swap_fb] * neg_fb

        reset_state(model)
        a_lr = model.act_inference(mirror_lr(obs))
        sym_loss_lr = F.mse_loss(a_lr, target_lr)

        reset_state(model)
        a_fb = model.act_inference(mirror_fb(obs))
        sym_loss_fb = F.mse_loss(a_fb, target_fb)

        total = task_loss + coef_lr * sym_loss_lr + coef_fb * sym_loss_fb
        total.backward()

        # Compute total grad norm
        gn = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gn += p.grad.pow(2).sum().item()
        gn = gn ** 0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history["step"].append(step)
        history["task_loss"].append(task_loss.item())
        history["lr_mse"].append(sym_loss_lr.item())
        history["fb_mse"].append(sym_loss_fb.item())
        history["grad_norm"].append(gn)

    return history


def main():
    section("[SWEEP] sym_loss_coef effect on combined objective")

    coefs_to_try = [
        (0.0, 0.0),   # baseline: PPO only
        (0.1, 0.1),   # gentle
        (0.3, 0.3),   # moderate, single-value default
        (0.5, 0.3),   # asymmetric: stronger LR (has piggyback), milder FB (no piggyback)
        (0.5, 0.5),   # balanced moderate
        (1.0, 1.0),   # aggressive
    ]

    print(f"\n{'coef_lr':>8} {'coef_fb':>8} {'task_init':>10} {'task_final':>11} {'lr_init':>10} {'lr_final':>10} {'fb_init':>10} {'fb_final':>10} {'grad_avg':>9}")
    results = []
    for lr_c, fb_c in coefs_to_try:
        h = run_one_sweep(lr_c, fb_c, n_steps=100, seed=0)
        avg_grad = sum(h["grad_norm"]) / len(h["grad_norm"])
        row = (lr_c, fb_c, h["task_loss"][0], h["task_loss"][-1],
               h["lr_mse"][0], h["lr_mse"][-1],
               h["fb_mse"][0], h["fb_mse"][-1], avg_grad)
        results.append(row)
        print(f"{lr_c:>8.2f} {fb_c:>8.2f} {row[2]:>10.4f} {row[3]:>11.4f} "
              f"{row[4]:>10.4f} {row[5]:>10.4f} {row[6]:>10.4f} {row[7]:>10.4f} {row[8]:>9.2f}")

    # Pick the recommendation: balance task tracking + sym constraint satisfaction.
    print("\n[Analysis]")
    for lr_c, fb_c, t0, t1, l0, l1, f0, f1, gn in results:
        sym_satisfied = (l1 < 0.01) and (f1 < 0.01)
        task_done     = (t1 < t0 * 0.5)
        bullet = "✓✓" if (sym_satisfied and task_done) else ("✓" if (sym_satisfied or task_done) else "−")
        print(f"  coef_lr={lr_c}, coef_fb={fb_c}: task_drop={t1/t0:.2f}, "
              f"lr_residual={l1:.4f}, fb_residual={f1:.4f}, grad={gn:.2f}  [{bullet}]")

    # The "best" balance is the one with both flags ✓✓ and the lowest sym residuals
    # without sacrificing task convergence.
    return 0


if __name__ == "__main__":
    sys.exit(main())
