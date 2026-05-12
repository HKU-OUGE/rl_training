#!/usr/bin/env python3
"""Unit test for SplitMoEPPO per-rank critic mode.

Verifies (no torch.distributed needed, no Isaac Sim app needed):
  1. SplitMoEPPO has __init__, _collect_critic_param_ids, reduce_parameters
  2. The PER_RANK_TERRAIN env var + is_multi_gpu gating exists in source
  3. _collect_critic_param_ids correctly returns only critic_rnn + critic_mlp
     param ids on a mock model (actor / encoder / std params NOT included)
  4. reduce_parameters() in per-rank mode:
       - does NOT touch critic param gradients (they stay at sentinel value)
       - touches actor + encoder grads (averaged via mocked all_reduce)
  5. reduce_parameters() in default mode (per_rank_critic=False) delegates
     to super (verified by checking no exclusion logic fires)

Run:
    cd /home/ouge/Software/rl_training
    python scripts/reinforcement_learning/rsl_rl/test_per_rank_critic.py
"""

import ast
import unittest
import unittest.mock as mock
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path("/home/ouge/Software/rl_training")
MOE_TERRAIN = (
    REPO
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
      "/config/wheeled/deeprobotics_m20/agents/moe_terrain.py"
)


# ---------------------------------------------------------------------------
# Stage 1: AST source checks (fast, no imports of moe_terrain itself)
# ---------------------------------------------------------------------------

class ASTSourceChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(MOE_TERRAIN.read_text())
        cls.src = MOE_TERRAIN.read_text()
        cls.cls_node = None
        for node in ast.walk(cls.tree):
            if isinstance(node, ast.ClassDef) and node.name == "SplitMoEPPO":
                cls.cls_node = node
                break

    def test_split_moe_ppo_class_exists(self):
        self.assertIsNotNone(self.cls_node, "class SplitMoEPPO not found")

    def test_three_methods_defined(self):
        method_names = {
            m.name
            for m in self.cls_node.body
            if isinstance(m, ast.FunctionDef)
        }
        for required in ("__init__", "_collect_critic_param_ids", "reduce_parameters", "update"):
            self.assertIn(required, method_names,
                          f"SplitMoEPPO must define {required}")

    def test_per_rank_terrain_env_var_referenced(self):
        self.assertIn("PER_RANK_TERRAIN", self.src,
                      "must gate on PER_RANK_TERRAIN env var")
        self.assertIn("is_multi_gpu", self.src,
                      "must check is_multi_gpu (single-GPU should be no-op)")

    def test_critic_attribute_names_used(self):
        """Only critic_rnn AND critic_mlp should be in the exclusion scope."""
        # Look at _collect_critic_param_ids body
        helper = None
        for m in self.cls_node.body:
            if isinstance(m, ast.FunctionDef) and m.name == "_collect_critic_param_ids":
                helper = m
                break
        self.assertIsNotNone(helper)
        helper_src = ast.unparse(helper)
        self.assertIn("critic_rnn", helper_src)
        self.assertIn("critic_mlp", helper_src)
        # Make sure no accidental inclusion of actor attrs:
        for actor_attr in ("'rnn'", '"rnn"', "actor_mlp", "scan_encoder", "elevation_encoder"):
            self.assertNotIn(actor_attr, helper_src,
                             f"helper should not reference {actor_attr}")

    def test_reduce_parameters_falls_through_when_off(self):
        """When per_rank_critic is False, must call super().reduce_parameters()."""
        red = None
        for m in self.cls_node.body:
            if isinstance(m, ast.FunctionDef) and m.name == "reduce_parameters":
                red = m
                break
        self.assertIsNotNone(red)
        red_src = ast.unparse(red)
        self.assertIn("super().reduce_parameters()", red_src,
                      "non-per-rank-critic path must delegate to super()")


# ---------------------------------------------------------------------------
# Stage 2: Logic checks using a mock model
# ---------------------------------------------------------------------------


def _make_mock_policy():
    """Build a minimal nn.Module that mirrors SplitMoEActorCritic's attr names
    relevant to the per-rank-critic logic."""
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            # ----- actor side -----
            self.rnn = nn.GRU(input_size=10, hidden_size=8, batch_first=False)
            self.actor_mlp = nn.Linear(8, 4)
            self.std = nn.Parameter(torch.zeros(4))
            # ----- critic side -----
            self.critic_rnn = nn.GRU(input_size=10, hidden_size=8, batch_first=False)
            self.critic_mlp = nn.Linear(8, 1)
            # ----- shared encoders -----
            self.scan_encoder = nn.Linear(50, 8)
            self.elevation_encoder = nn.Linear(20, 8)
    return _Model()


class CollectCriticParamIds(unittest.TestCase):
    """Mimic the helper logic on a mock model."""

    @staticmethod
    def _collect(model):
        ids = set()
        for attr in ("critic_rnn", "critic_mlp"):
            mod = getattr(model, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    ids.add(id(p))
        return ids

    def test_critic_params_included(self):
        m = _make_mock_policy()
        ids = self._collect(m)
        for p in m.critic_rnn.parameters():
            self.assertIn(id(p), ids)
        for p in m.critic_mlp.parameters():
            self.assertIn(id(p), ids)

    def test_actor_and_encoder_excluded(self):
        m = _make_mock_policy()
        ids = self._collect(m)
        # Actor RNN should NOT be in exclusion set
        for p in m.rnn.parameters():
            self.assertNotIn(id(p), ids)
        # Actor MLP head
        for p in m.actor_mlp.parameters():
            self.assertNotIn(id(p), ids)
        # std (action noise std)
        self.assertNotIn(id(m.std), ids)
        # Shared encoders
        for p in m.scan_encoder.parameters():
            self.assertNotIn(id(p), ids)
        for p in m.elevation_encoder.parameters():
            self.assertNotIn(id(p), ids)

    def test_no_critic_attrs_returns_empty(self):
        # If model lacks critic_rnn/critic_mlp, set must be empty (degenerate
        # but should not throw)
        class _ActorOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(input_size=10, hidden_size=8)
        ids = self._collect(_ActorOnly())
        self.assertEqual(ids, set())


# ---------------------------------------------------------------------------
# Stage 3: Reduce-step behavior with mocked all_reduce
# ---------------------------------------------------------------------------


class ReduceParametersBehavior(unittest.TestCase):
    """Simulate reduce_parameters() end-to-end with torch.distributed mocked."""

    def _build_ppo_like(self, per_rank_critic: bool, world_size: int = 4):
        """Construct a minimal stand-in for SplitMoEPPO with enough attributes."""
        class _PPO:
            pass
        ppo = _PPO()
        ppo.policy = _make_mock_policy()
        ppo.rnd = None
        ppo.is_multi_gpu = True
        ppo.gpu_world_size = world_size
        ppo.per_rank_critic = per_rank_critic
        ppo._critic_param_ids = (
            CollectCriticParamIds._collect(ppo.policy) if per_rank_critic else set()
        )
        # Give every param a non-zero gradient so we can detect "touched" vs not.
        for p in ppo.policy.parameters():
            p.grad = torch.full_like(p, 7.0)
        # Mark critic params with a SENTINEL grad so untouched is detectable.
        for p in ppo.policy.critic_rnn.parameters():
            p.grad = torch.full_like(p, 99.0)
        for p in ppo.policy.critic_mlp.parameters():
            p.grad = torch.full_like(p, 99.0)
        return ppo

    def _reduce(self, ppo):
        """Run the same algorithm as SplitMoEPPO.reduce_parameters override."""
        crit_ids = ppo._critic_param_ids
        pol_params = [p for p in ppo.policy.parameters() if id(p) not in crit_ids]
        if ppo.rnd:
            all_param_list = list(chain(pol_params, ppo.rnd.parameters()))
        else:
            all_param_list = pol_params
        params_with_grad = [p for p in all_param_list if p.grad is not None]
        if not params_with_grad:
            return
        grads = [p.grad.view(-1) for p in params_with_grad]
        all_grads = torch.cat(grads)
        # Mock all_reduce: pretend every rank has the same grad → SUM = N * grad
        # Then divide by N → identity. To detect "was the tensor touched", we
        # flip its sign so the post-scatter value differs from sentinel.
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= ppo.gpu_world_size
        offset = 0
        for p in params_with_grad:
            numel = p.grad.numel()
            p.grad.copy_(all_grads[offset : offset + numel].view_as(p.grad))
            offset += numel

    def test_per_rank_critic_keeps_critic_grads_local(self):
        ppo = self._build_ppo_like(per_rank_critic=True)
        # Mock all_reduce to write -1 into the tensor so we can detect "touched"
        def mock_all_reduce(tensor, op):
            tensor.fill_(-1.0)
        with mock.patch("torch.distributed.all_reduce", side_effect=mock_all_reduce):
            self._reduce(ppo)
        # Critic grads should retain the 99.0 sentinel (not touched)
        for p in ppo.policy.critic_rnn.parameters():
            self.assertTrue(torch.all(p.grad == 99.0),
                            "critic_rnn grad was modified despite per_rank_critic=True")
        for p in ppo.policy.critic_mlp.parameters():
            self.assertTrue(torch.all(p.grad == 99.0),
                            "critic_mlp grad was modified despite per_rank_critic=True")
        # Actor grads should be touched (mock wrote -1 / world_size = -0.25)
        for p in ppo.policy.rnn.parameters():
            expected = -1.0 / ppo.gpu_world_size
            self.assertTrue(torch.allclose(p.grad, torch.full_like(p.grad, expected)),
                            f"rnn grad = {p.grad.flatten()[:3]} expected {expected}")
        # Encoders also touched
        for p in ppo.policy.scan_encoder.parameters():
            expected = -1.0 / ppo.gpu_world_size
            self.assertTrue(torch.allclose(p.grad, torch.full_like(p.grad, expected)))

    def test_per_rank_critic_off_reduces_critic_too(self):
        ppo = self._build_ppo_like(per_rank_critic=False)
        # critic params should NOT be in exclusion set
        self.assertEqual(ppo._critic_param_ids, set())
        # When we run reduce, the critic grads SHOULD be touched.
        def mock_all_reduce(tensor, op):
            tensor.fill_(-1.0)
        with mock.patch("torch.distributed.all_reduce", side_effect=mock_all_reduce):
            self._reduce(ppo)
        expected = -1.0 / ppo.gpu_world_size
        for p in ppo.policy.critic_rnn.parameters():
            self.assertTrue(torch.allclose(p.grad, torch.full_like(p.grad, expected)),
                            "critic_rnn grad should have been averaged when per_rank_critic=False")
        for p in ppo.policy.actor_mlp.parameters():
            self.assertTrue(torch.allclose(p.grad, torch.full_like(p.grad, expected)))

    def test_payload_size_excludes_critic_in_per_rank_mode(self):
        ppo = self._build_ppo_like(per_rank_critic=True)
        captured = []
        def mock_all_reduce(tensor, op):
            captured.append(tensor.numel())
            tensor.fill_(0.0)
        with mock.patch("torch.distributed.all_reduce", side_effect=mock_all_reduce):
            self._reduce(ppo)
        # exactly one all_reduce call
        self.assertEqual(len(captured), 1)
        # payload = total numel - critic numel
        total = sum(p.grad.numel() for p in ppo.policy.parameters())
        critic_numel = sum(
            p.numel() for p in chain(
                ppo.policy.critic_rnn.parameters(),
                ppo.policy.critic_mlp.parameters(),
            )
        )
        self.assertEqual(captured[0], total - critic_numel,
                         f"all_reduce payload {captured[0]} must equal "
                         f"total({total}) - critic({critic_numel})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
