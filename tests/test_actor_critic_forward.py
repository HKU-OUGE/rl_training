"""End-to-end forward-pass tests for SplitMoEActorCritic.

Covers (with a real instance, not mocks):
    - act / act_inference / evaluate shape and dtype
    - Hidden state shape and update during step
    - reset() and reset(dones=...) behaviour
    - get_hidden_states / load consistency
    - Multi-step rollout: action varies across steps for the same input
      (RNN context is being used)

Run:
    /home/ouge/miniconda3/envs/env_isaaclab/bin/python tests/test_actor_critic_forward.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from conftest import (
    load_moe_terrain, make_actor_critic, make_fake_obs, make_ppo,
    NUM_ACTIONS, POLICY_DIM, NOISY_ELEV_DIM,
    section, passed, failed,
)


B_TEST = 4
LATENT_DIM = 32


# ---------------------------------------------------------------------------

def test_act_shape():
    section("[F1] act() shape and dtype")
    model, obs = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.eval()

    with torch.no_grad():
        action = model.act(obs)

    ok = True
    if action.shape == (B_TEST, NUM_ACTIONS):
        passed(f"act() output shape {tuple(action.shape)}")
    else:
        failed(f"act() shape {tuple(action.shape)} != (B={B_TEST}, A={NUM_ACTIONS})")
        ok = False

    if action.dtype == torch.float32 and not torch.isnan(action).any():
        passed(f"action dtype OK, no NaN")
    else:
        failed(f"bad action: dtype={action.dtype}, has NaN={torch.isnan(action).any()}")
        ok = False
    return ok


def test_act_inference_shape():
    section("[F2] act_inference() shape (deterministic, returns mean)")
    model, obs = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.eval()

    with torch.no_grad():
        a1 = model.act_inference(obs)
        a2 = model.act_inference(obs, hidden_states=model.active_hidden_states)
    ok = True
    if a1.shape == (B_TEST, NUM_ACTIONS):
        passed(f"act_inference shape {tuple(a1.shape)}")
    else:
        failed(f"shape: {tuple(a1.shape)}")
        ok = False
    return ok


def test_evaluate_shape():
    section("[F3] evaluate() returns value of shape (B, 1)")
    model, obs = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.eval()

    with torch.no_grad():
        value = model.evaluate(obs)
    ok = True
    if value.shape == (B_TEST, 1):
        passed(f"value shape {tuple(value.shape)}")
    else:
        failed(f"shape: {tuple(value.shape)}")
        ok = False
    if not torch.isnan(value).any():
        passed("value no NaN")
    else:
        failed("NaN in value")
        ok = False
    return ok


def test_initial_hidden_state_shape():
    section("[F4] _init_rnn_state shape: (num_layers=2 if sym_enabled, 1 otherwise) × B × H")
    ok = True

    # sym_enabled=True (default)
    m_sym, _ = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM, sym_loss_coef=0.3)
    if m_sym.sym_enabled and m_sym.active_hidden_states.shape == (2, B_TEST, LATENT_DIM):
        passed(f"sym_enabled=True → 2 layers, shape {tuple(m_sym.active_hidden_states.shape)}")
    else:
        failed(f"sym_enabled state: shape {tuple(m_sym.active_hidden_states.shape)}, sym_enabled={m_sym.sym_enabled}")
        ok = False

    # sym_disabled (sym_loss_coef=0)
    m_no, _ = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM, sym_loss_coef=0.0)
    if (not m_no.sym_enabled) and m_no.active_hidden_states.shape == (1, B_TEST, LATENT_DIM):
        passed(f"sym_enabled=False → 1 layer, shape {tuple(m_no.active_hidden_states.shape)}")
    else:
        failed(f"sym_disabled state: shape {tuple(m_no.active_hidden_states.shape)}, sym_enabled={m_no.sym_enabled}")
        ok = False
    return ok


def test_hidden_state_evolves_with_act():
    """Note: model.act() (SplitMoEActorCritic.act) is the rsl_rl base interface and
    only updates layer 0 (real) hidden state — layer 1 stays untouched by design
    (see _run_rnn line 750-754: 'reassemble: new layer 0 + original layer 1').
    Layer 1 is only touched by SplitMoEPPO.act() — see test_piggyback_via_ppo."""
    section("[F5] model.act() updates layer 0 only (layer 1 piggyback handled by PPO.act)")
    model, obs = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.eval()

    h_before = model.active_hidden_states.clone()
    with torch.no_grad():
        _ = model.act(obs)
    h_after = model.active_hidden_states.clone()

    ok = True
    layer0_diff = (h_before[0] - h_after[0]).abs().max().item()
    layer1_diff = (h_before[1] - h_after[1]).abs().max().item()

    if layer0_diff > 1e-6:
        passed(f"layer 0 (real) changed: max_abs_diff={layer0_diff:.4f}")
    else:
        failed(f"layer 0 didn't change")
        ok = False

    if layer1_diff < 1e-9:
        passed(f"layer 1 unchanged by model.act() (as designed)")
    else:
        failed(f"layer 1 unexpectedly changed: max_abs_diff={layer1_diff:.6f}")
        ok = False
    return ok


def test_piggyback_via_ppo():
    """SplitMoEPPO.act() does the 2B batched forward and updates BOTH layer 0
    (real) and layer 1 (L-R mirror piggyback). Verify layer 1 evolves and differs
    from layer 0 after multiple steps (layer 0 sees obs, layer 1 sees M_LR(obs))."""
    section("[F6] SplitMoEPPO.act() piggyback: layer 0 ≠ layer 1 across rollout")
    ppo = make_ppo()
    model = ppo.policy   # rsl_rl PPO stores model as .policy
    model.eval()

    # Init storage so PPO.act() can write transitions
    # SplitMoEPPO.act writes to self.transition; we just need self.transition to exist
    # (rsl_rl PPO sets it up lazily). We'll write a minimal init.
    import types as _types
    # If transition is not yet a real obj, replace with a SimpleNamespace
    if not hasattr(ppo.transition, "actions"):
        ppo.transition = _types.SimpleNamespace()

    # Reset both layers to zero so any divergence is purely from piggyback.
    if model.rnn_type == "lstm":
        model.active_hidden_states = (
            torch.zeros_like(model.active_hidden_states[0]),
            torch.zeros_like(model.active_hidden_states[1]),
        )
    else:
        model.active_hidden_states = torch.zeros_like(model.active_hidden_states)

    dev = model.active_hidden_states.device if not isinstance(model.active_hidden_states, tuple) else model.active_hidden_states[0].device

    with torch.no_grad():
        for step in range(8):
            obs = make_fake_obs(B=B_TEST, history=5, seed=step + 100, device=dev)
            _ = ppo.act(obs)

    h = model.active_hidden_states
    if isinstance(h, tuple):
        h_check = h[0]
    else:
        h_check = h

    layer0_norm = h_check[0].abs().max().item()
    layer1_norm = h_check[1].abs().max().item()
    diff = (h_check[0] - h_check[1]).abs().max().item()

    ok = True
    if layer0_norm > 1e-3:
        passed(f"layer 0 evolved (max_abs={layer0_norm:.4f})")
    else:
        failed(f"layer 0 stagnant: max_abs={layer0_norm:.2e}")
        ok = False
    if layer1_norm > 1e-3:
        passed(f"layer 1 evolved through L-R mirror obs (max_abs={layer1_norm:.4f})")
    else:
        failed(f"layer 1 stagnant: max_abs={layer1_norm:.2e} — PPO piggyback broken!")
        ok = False
    if diff > 1e-3:
        passed(f"layer 0 ≠ layer 1 (diff={diff:.4f}) — piggyback differentiating real vs mirror")
    else:
        failed(f"layers identical: diff={diff:.2e}")
        ok = False
    return ok


def test_hidden_state_shape_constant_across_rollout():
    section("[F7] hidden state shape is invariant across rollout steps")
    model, _ = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.eval()
    initial_shape = tuple(model.active_hidden_states.shape) if not isinstance(model.active_hidden_states, tuple) else (
        tuple(model.active_hidden_states[0].shape), tuple(model.active_hidden_states[1].shape)
    )

    with torch.no_grad():
        for step in range(5):
            obs = make_fake_obs(B=B_TEST, history=5, seed=step, device=model.active_hidden_states.device if not isinstance(model.active_hidden_states, tuple) else model.active_hidden_states[0].device)
            _ = model.act(obs)

    final_shape = tuple(model.active_hidden_states.shape) if not isinstance(model.active_hidden_states, tuple) else (
        tuple(model.active_hidden_states[0].shape), tuple(model.active_hidden_states[1].shape)
    )

    ok = initial_shape == final_shape
    if ok:
        passed(f"hidden state shape constant: {initial_shape}")
    else:
        failed(f"shape changed: {initial_shape} → {final_shape}")
    return ok


def test_reset_zeroes_hidden_state():
    section("[F8] reset() zeroes hidden state across batch / selectively via dones")
    model, obs = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.eval()

    # Run a few steps to make hidden state non-zero.
    with torch.no_grad():
        _ = model.act(obs)
        _ = model.act(obs)
    ok = True

    # Reset some envs via dones mask, others kept.
    dones = torch.zeros(B_TEST, dtype=torch.bool)
    dones[0] = True
    dones[2] = True

    h_before = model.active_hidden_states.clone()
    model.reset(dones=dones)
    h_after = model.active_hidden_states

    # Reset env (idx 0, 2) hidden state should be zero, others preserved.
    for env_idx in range(B_TEST):
        if dones[env_idx]:
            h_env = h_after[:, env_idx, :].abs().max().item()
            if h_env < 1e-9:
                passed(f"env {env_idx} (reset): hidden state zero")
            else:
                failed(f"env {env_idx} hidden state not zeroed (max_abs={h_env})")
                ok = False
        else:
            diff = (h_before[:, env_idx, :] - h_after[:, env_idx, :]).abs().max().item()
            if diff < 1e-9:
                passed(f"env {env_idx} (preserved): hidden state unchanged")
            else:
                failed(f"env {env_idx} preserved but changed (diff={diff})")
                ok = False
    return ok


def test_action_uses_rnn_context():
    """If the RNN is doing its job, repeated forward passes with the SAME obs should
    give DIFFERENT outputs (because hidden state evolves)."""
    section("[F9] act() output varies across steps (RNN context is used)")
    model, obs = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.eval()
    model.reset()  # ensure clean start

    with torch.no_grad():
        a1 = model.act_inference(obs).clone()
        # advance state
        _ = model.act(obs)
        a2 = model.act_inference(obs).clone()

    diff = (a1 - a2).abs().max().item()
    ok = diff > 1e-6
    if ok:
        passed(f"action changed after RNN step: max_abs_diff={diff:.4e}")
    else:
        failed(f"action unchanged: diff={diff:.2e}")
    return ok


def test_gradient_flow_through_act():
    section("[F10] gradient flow: backward through act() produces non-zero param grads")
    model, obs = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.train()

    # Use act_inference to get a deterministic mean, then compute a synthetic loss.
    mean = model.act_inference(obs)
    loss = mean.pow(2).sum()
    loss.backward()

    grads = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads.append((n, p.grad.norm().item()))

    ok = True
    nonzero = [g for _, g in grads if g > 0]
    if len(nonzero) > 5:
        passed(f"backward produced {len(nonzero)}/{len(grads)} non-zero grad norms")
    else:
        failed(f"too few non-zero grads: {len(nonzero)}/{len(grads)}")
        ok = False

    if all(not torch.isnan(p.grad).any() for _, p in model.named_parameters() if p.grad is not None):
        passed("no NaN in gradients")
    else:
        failed("NaN found in gradients")
        ok = False
    return ok


def test_with_aux_loss_path():
    """The act() method has a return_aux_loss path producing aux outputs (load balancing,
    AE recons) used during PPO update."""
    section("[F11] act(return_aux_loss=True) path produces aux dict")
    model, obs = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.train()

    out = model.act(obs, return_aux_loss=True)
    if isinstance(out, tuple) and len(out) == 2:
        action, aux = out
        ok = True
        if action.shape == (B_TEST, NUM_ACTIONS):
            passed(f"action shape OK: {tuple(action.shape)}")
        else:
            failed(f"action shape: {tuple(action.shape)}")
            ok = False
        if isinstance(aux, dict):
            passed(f"aux is dict with keys: {sorted(aux.keys())}")
        else:
            failed(f"aux not dict: {type(aux)}")
            ok = False
        if "lb_loss" in aux:
            lb = aux["lb_loss"]
            if torch.is_tensor(lb) and lb.numel() == 1 and lb.item() >= 0:
                passed(f"lb_loss = {lb.item():.4f} (non-negative scalar)")
            else:
                failed(f"lb_loss bad: {lb}")
                ok = False
        return ok
    else:
        failed(f"return_aux_loss=True gave: {type(out)}")
        return False


def test_sym_loss_nonzero_at_init():
    """For random init weights, the policy is NOT yet equivariant. Sym loss MSE
    between π(M·obs) and M·π(obs) should be non-trivial."""
    section("[F12] At random init, π is NOT equivariant — sym loss MSE > 0")
    model, obs = make_actor_critic(B=B_TEST, latent_dim=LATENT_DIM)
    model.eval()

    # Action mirror tensors (L-R)
    dev = obs["policy"].device
    swap_lr = torch.tensor([3,4,5, 0,1,2, 9,10,11, 6,7,8, 13,12, 15,14], dtype=torch.long, device=dev)
    neg_lr  = torch.tensor([-1,1,1, -1,1,1, -1,1,1, -1,1,1, 1,1,1,1], dtype=torch.float32, device=dev)
    swap_fb = torch.tensor([6,7,8, 9,10,11, 0,1,2, 3,4,5, 14,15, 12,13], dtype=torch.long, device=dev)
    neg_fb  = torch.tensor([1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, -1,-1,-1,-1], dtype=torch.float32, device=dev)

    # We need a SplitMoEPPO instance to call _mirror_obs / _mirror_obs_fb.
    # Build a minimal one — only need self.device, self.actor_critic.
    import types as _types
    ppo_stub = _types.SimpleNamespace(
        device=dev,
        actor_critic=model,
        policy=model,
    )
    mod = load_moe_terrain()
    mirror_obs = mod.SplitMoEPPO._mirror_obs.__get__(ppo_stub, _types.SimpleNamespace)
    mirror_obs_fb = mod.SplitMoEPPO._mirror_obs_fb.__get__(ppo_stub, _types.SimpleNamespace)

    with torch.no_grad():
        a_real = model.act_inference(obs).clone()
        # Reset state to ensure same starting context for mirror eval.
        h_real = model.active_hidden_states
        # For sym test, just compute act_inference on mirrored obs from same state.
        obs_lr = mirror_obs(obs)
        obs_fb = mirror_obs_fb(obs)

        # Reset to same hidden state before each mirrored forward
        if model.rnn_type == "gru":
            model.active_hidden_states = h_real.clone()
        a_lr = model.act_inference(obs_lr).clone()
        if model.rnn_type == "gru":
            model.active_hidden_states = h_real.clone()
        a_fb = model.act_inference(obs_fb).clone()

    # Targets: M · a_real
    target_lr = a_real[..., swap_lr] * neg_lr
    target_fb = a_real[..., swap_fb] * neg_fb

    mse_lr = (a_lr - target_lr).pow(2).mean().item()
    mse_fb = (a_fb - target_fb).pow(2).mean().item()
    print(f"  MSE(π(M_LR·obs), M_LR·π(obs)) = {mse_lr:.6f}")
    print(f"  MSE(π(M_FB·obs), M_FB·π(obs)) = {mse_fb:.6f}")

    ok = True
    if mse_lr > 1e-6:
        passed("L-R sym MSE > 0 at random init (constraint has work to do)")
    else:
        failed(f"L-R sym MSE near zero at random init: {mse_lr}")
        ok = False
    if mse_fb > 1e-6:
        passed("F-B sym MSE > 0 at random init (constraint has work to do)")
    else:
        failed(f"F-B sym MSE near zero at random init: {mse_fb}")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tests = [
        ("act() shape",                  test_act_shape),
        ("act_inference() shape",        test_act_inference_shape),
        ("evaluate() shape",             test_evaluate_shape),
        ("init hidden state shape",      test_initial_hidden_state_shape),
        ("hidden evolves with act",      test_hidden_state_evolves_with_act),
        ("PPO piggyback layer 1",        test_piggyback_via_ppo),
        ("hidden shape stable",          test_hidden_state_shape_constant_across_rollout),
        ("reset(dones) selective zero",  test_reset_zeroes_hidden_state),
        ("action varies across steps",   test_action_uses_rnn_context),
        ("gradient flow",                test_gradient_flow_through_act),
        ("aux loss path",                test_with_aux_loss_path),
        ("sym MSE > 0 at random init",   test_sym_loss_nonzero_at_init),
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
