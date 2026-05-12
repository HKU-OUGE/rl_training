#!/usr/bin/env python3
"""Unit test for the PER_RANK_TERRAIN + per-rank is_terminated override.

Validates that train_moe.py and inspect_terrain.py are wired so that:
  1. rank 4 maps to STEPPING_STONES_TEACHER_TERRAINS_CFG (was FLAT after revert)
  2. PER_RANK_NAMES default contains "STONES" at index 4 (was "FLAT2")
  3. _RANK_IS_TERMINATED_WEIGHT_OVERRIDE maps rank 4 → 0.0 (so jumping the
     gap and falling doesn't get the -100 termination penalty that makes the
     policy "afraid to try")
  4. The override is actually applied to env_cfg.rewards.is_terminated.weight
     inside the dispatch block
  5. STEPPING_STONES_TEACHER_TERRAINS_CFG actually contains stepping-stones
     sub_terrains (only attempted if the Isaac Lab import works without sim app)

Run:
    cd /home/ouge/Software/rl_training
    python scripts/reinforcement_learning/rsl_rl/test_per_rank_terrain.py
"""

import ast
import sys
import unittest
from pathlib import Path

REPO = Path("/home/ouge/Software/rl_training")
TRAIN_MOE = REPO / "scripts/reinforcement_learning/rsl_rl/train_moe.py"
INSPECT = REPO / "scripts/reinforcement_learning/rsl_rl/inspect_terrain.py"


def load_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def find_assignment_value(tree: ast.Module, name: str):
    """Return the rhs ast node of `name = ...` (first occurrence)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return node.value
    return None


class ASTSourceChecks(unittest.TestCase):
    """Static checks against source AST — fast, no Isaac Lab needed."""

    def test_train_moe_rank4_is_stepping_stones(self):
        tree = load_ast(TRAIN_MOE)
        list_node = find_assignment_value(tree, "_RANK_TERRAIN_MAP")
        self.assertIsNotNone(list_node, "_RANK_TERRAIN_MAP not found in train_moe.py")
        self.assertIsInstance(list_node, ast.List)
        self.assertGreaterEqual(len(list_node.elts), 5, "_RANK_TERRAIN_MAP has < 5 entries")
        rank4 = list_node.elts[4]
        self.assertIsInstance(rank4, ast.Name,
                              f"rank 4 is {ast.dump(rank4)}, expected Name")
        self.assertEqual(
            rank4.id, "STEPPING_STONES_TEACHER_TERRAINS_CFG",
            f"rank 4 = {rank4.id}, expected STEPPING_STONES_TEACHER_TERRAINS_CFG"
        )

    def test_train_moe_per_rank_names_default(self):
        src = TRAIN_MOE.read_text()
        self.assertIn(
            '"FLAT,STAIR_SLOPE,PLATFORM,SCAN,STONES,RAIL,NOISE,GRID"',
            src,
            "Default PER_RANK_NAMES should have STONES at idx 4"
        )
        self.assertNotIn('"FLAT,STAIR_SLOPE,PLATFORM,SCAN,FLAT2,RAIL,NOISE,GRID"', src,
                         "Stale FLAT2 default must be replaced")

    def test_train_moe_termination_override_dict_present(self):
        tree = load_ast(TRAIN_MOE)
        dict_node = find_assignment_value(tree, "_RANK_IS_TERMINATED_WEIGHT_OVERRIDE")
        self.assertIsNotNone(dict_node,
                             "_RANK_IS_TERMINATED_WEIGHT_OVERRIDE not found in train_moe.py")
        self.assertIsInstance(dict_node, ast.Dict)
        found_4 = False
        for k, v in zip(dict_node.keys, dict_node.values):
            self.assertIsInstance(k, ast.Constant, f"non-Constant key {ast.dump(k)}")
            if k.value == 4:
                found_4 = True
                self.assertIsInstance(v, ast.Constant,
                                      f"override value for rank 4 is {ast.dump(v)}, expected Constant")
                self.assertEqual(v.value, 0.0,
                                 f"rank 4 override = {v.value}, expected 0.0")
        self.assertTrue(found_4,
                        "rank 4 missing from _RANK_IS_TERMINATED_WEIGHT_OVERRIDE")

    def test_train_moe_override_is_actually_applied(self):
        """The dict alone is dead code unless it's read and assigned. Verify both sides."""
        src = TRAIN_MOE.read_text()
        self.assertIn(
            "_RANK_IS_TERMINATED_WEIGHT_OVERRIDE.get(local_rank)",
            src,
            "override dict declared but never looked up"
        )
        self.assertIn(
            "env_cfg.rewards.is_terminated.weight = _term_weight_override",
            src,
            "override value never assigned to env_cfg.rewards.is_terminated.weight"
        )

    def test_inspect_terrain_rank4_is_stepping_stones(self):
        tree = load_ast(INSPECT)
        list_node = find_assignment_value(tree, "RANK_TERRAIN_MAP")
        self.assertIsNotNone(list_node, "RANK_TERRAIN_MAP not found in inspect_terrain.py")
        self.assertIsInstance(list_node, ast.List)
        rank4 = list_node.elts[4]
        self.assertIsInstance(rank4, ast.Name)
        self.assertEqual(rank4.id, "STEPPING_STONES_TEACHER_TERRAINS_CFG")

    def test_inspect_terrain_rank_names_have_stones(self):
        tree = load_ast(INSPECT)
        list_node = find_assignment_value(tree, "RANK_NAMES")
        self.assertIsNotNone(list_node)
        self.assertIsInstance(list_node, ast.List)
        names = [n.value for n in list_node.elts if isinstance(n, ast.Constant)]
        self.assertEqual(
            names[4], "STONES",
            f"RANK_NAMES[4] = {names[4]!r}, expected 'STONES'"
        )
        self.assertNotIn("FLAT2", names, "stale 'FLAT2' must be replaced")


class TerrainCfgImportChecks(unittest.TestCase):
    """Tests that need importing the actual rough.py cfg.
    Skipped if Isaac Lab import requires sim app.
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO / "source/rl_training"))
        try:
            from rl_training.terrains.config.rough import (
                STEPPING_STONES_TEACHER_TERRAINS_CFG,
            )
            cls.cfg = STEPPING_STONES_TEACHER_TERRAINS_CFG
        except Exception as e:
            raise unittest.SkipTest(
                f"terrain cfg import failed (likely needs sim app): {type(e).__name__}: {e}"
            )

    def test_stepping_stones_cfg_has_terrain_generator_fields(self):
        self.assertTrue(hasattr(self.cfg, "sub_terrains"))
        self.assertTrue(hasattr(self.cfg, "num_rows"))
        self.assertTrue(hasattr(self.cfg, "num_cols"))
        self.assertTrue(hasattr(self.cfg, "size"))

    def test_stepping_stones_cfg_sub_terrains_nonempty(self):
        sub = self.cfg.sub_terrains
        self.assertGreater(len(sub), 0, "sub_terrains is empty")

    def test_stepping_stones_cfg_actually_has_stones(self):
        sub_names = list(self.cfg.sub_terrains.keys())
        has_stones = any(
            ("step" in n.lower()) or ("stone" in n.lower()) for n in sub_names
        )
        self.assertTrue(
            has_stones,
            f"None of sub_terrains look like stepping-stones: {sub_names}"
        )

    def test_proportions_sum_to_one_with_tolerance(self):
        # Each sub_terrain has a `proportion` attr (TerrainGeneratorCfg convention).
        total = sum(t.proportion for t in self.cfg.sub_terrains.values())
        self.assertAlmostEqual(
            total, 1.0, places=2,
            msg=f"Proportions sum to {total:.4f}, expected ~1.0"
        )


class SimulatedOverrideBehavior(unittest.TestCase):
    """Replicate the dispatch logic on a mock env_cfg, verify behavior."""

    def _make_mock_env_cfg(self):
        # Minimal mock with the attributes that the override touches
        class _IsTerm:
            weight = -100.0
        class _Rewards:
            is_terminated = _IsTerm()
        class _Terrain:
            terrain_generator = None
        class _Scene:
            terrain = _Terrain()
        class _EnvCfg:
            scene = _Scene()
            rewards = _Rewards()
        return _EnvCfg()

    def _simulate_dispatch(self, local_rank: int, num_ranks: int):
        """Run a minimal port of the dispatch logic. Returns (env_cfg, chosen_marker)."""
        env_cfg = self._make_mock_env_cfg()
        terrain_map = ["FLAT", "STAIR", "PLAT", "SCAN", "STONES", "RAIL", "NOISE", "GRID"]
        override_map = {4: 0.0}

        if local_rank < num_ranks:
            chosen = terrain_map[local_rank]
            env_cfg.scene.terrain.terrain_generator = chosen
            override = override_map.get(local_rank)
            if override is not None:
                env_cfg.rewards.is_terminated.weight = override
        return env_cfg

    def test_rank_4_gets_zero_termination_weight(self):
        env_cfg = self._simulate_dispatch(local_rank=4, num_ranks=8)
        self.assertEqual(env_cfg.rewards.is_terminated.weight, 0.0,
                         "rank 4 should have is_terminated.weight overridden to 0.0")
        self.assertEqual(env_cfg.scene.terrain.terrain_generator, "STONES")

    def test_other_ranks_keep_default_termination_weight(self):
        for r in [0, 1, 2, 3, 5, 6, 7]:
            env_cfg = self._simulate_dispatch(local_rank=r, num_ranks=8)
            self.assertEqual(
                env_cfg.rewards.is_terminated.weight, -100.0,
                f"rank {r} should keep default -100.0 (got {env_cfg.rewards.is_terminated.weight})"
            )

    def test_out_of_range_rank_does_nothing(self):
        # rank 8+ on a single-node setup: no terrain swap, no override.
        env_cfg = self._simulate_dispatch(local_rank=8, num_ranks=8)
        self.assertIsNone(env_cfg.scene.terrain.terrain_generator)
        self.assertEqual(env_cfg.rewards.is_terminated.weight, -100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
