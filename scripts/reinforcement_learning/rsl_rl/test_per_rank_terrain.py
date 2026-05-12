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

    def test_train_moe_rank4_is_gap_stones_mix(self):
        tree = load_ast(TRAIN_MOE)
        list_node = find_assignment_value(tree, "_RANK_TERRAIN_MAP")
        self.assertIsNotNone(list_node, "_RANK_TERRAIN_MAP not found in train_moe.py")
        self.assertIsInstance(list_node, ast.List)
        self.assertGreaterEqual(len(list_node.elts), 5, "_RANK_TERRAIN_MAP has < 5 entries")
        rank4 = list_node.elts[4]
        self.assertIsInstance(rank4, ast.Name,
                              f"rank 4 is {ast.dump(rank4)}, expected Name")
        self.assertEqual(
            rank4.id, "GAP_STONES_MIX_TEACHER_TERRAINS_CFG",
            f"rank 4 = {rank4.id}, expected GAP_STONES_MIX_TEACHER_TERRAINS_CFG "
            f"(MeshGap + HfSteppingStones 50/50)"
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

    @staticmethod
    def _ast_numeric_value(node):
        """Unwrap an ast node into a Python numeric value.

        Negative literals like `-0.01` parse as UnaryOp(USub, Constant(0.01)),
        NOT as Constant(-0.01). This helper handles both.
        """
        if isinstance(node, ast.Constant):
            return node.value
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)):
            return -node.operand.value
        raise TypeError(f"Expected Constant or -Constant, got {ast.dump(node)}")

    def test_train_moe_reward_override_dict_present(self):
        """Nested dict _RANK_REWARD_WEIGHT_OVERRIDES with rank 2/3/4 must exist."""
        tree = load_ast(TRAIN_MOE)
        dict_node = find_assignment_value(tree, "_RANK_REWARD_WEIGHT_OVERRIDES")
        self.assertIsNotNone(dict_node,
                             "_RANK_REWARD_WEIGHT_OVERRIDES not found in train_moe.py")
        self.assertIsInstance(dict_node, ast.Dict,
                              "_RANK_REWARD_WEIGHT_OVERRIDES must be a dict literal")

        # Build {rank: inner_dict} for assertions
        rank_to_inner = {}
        for k, v in zip(dict_node.keys, dict_node.values):
            rank_val = self._ast_numeric_value(k)
            self.assertIsInstance(v, ast.Dict,
                                  f"rank {rank_val} value must be a dict, got {ast.dump(v)}")
            inner = {}
            for ik, iv in zip(v.keys, v.values):
                self.assertIsInstance(ik, ast.Constant)
                inner[ik.value] = self._ast_numeric_value(iv)
            rank_to_inner[rank_val] = inner

        # rank 4 (STONES): preserves is_terminated=0 from commit 46238ac
        self.assertIn(4, rank_to_inner, "rank 4 missing from override dict")
        self.assertEqual(rank_to_inner[4].get("is_terminated"), 0.0,
                         f"rank 4 is_terminated = {rank_to_inner[4].get('is_terminated')}, expected 0.0")

        # rank 2 (PLATFORM): must contain platform-specific overrides
        self.assertIn(2, rank_to_inner, "rank 2 (PLATFORM) missing")
        self.assertEqual(rank_to_inner[2].get("ang_vel_xy_l2"), -0.01)
        self.assertEqual(rank_to_inner[2].get("base_height_l2"), 0.0)
        self.assertEqual(rank_to_inner[2].get("feet_air_time"), 1.5)
        self.assertEqual(rank_to_inner[2].get("feet_height_body"), -0.5)
        self.assertEqual(rank_to_inner[2].get("upward"), 0.0)

        # rank 3 (SCAN): must contain scan-specific overrides
        self.assertIn(3, rank_to_inner, "rank 3 (SCAN) missing")
        self.assertEqual(rank_to_inner[3].get("lin_vel_z_l2"), -2.0)
        self.assertEqual(rank_to_inner[3].get("base_height_l2"), -0.5)
        self.assertEqual(rank_to_inner[3].get("feet_air_time"), 0.0)
        self.assertEqual(rank_to_inner[3].get("feet_height_body"), 0.0)
        self.assertEqual(rank_to_inner[3].get("upward"), 0.08)

    def test_train_moe_override_is_actually_applied(self):
        """Dict declared but unused = dead code. Verify both the lookup and the assignment."""
        src = TRAIN_MOE.read_text()
        self.assertIn(
            "_RANK_REWARD_WEIGHT_OVERRIDES.get(local_rank",
            src,
            "override dict declared but never looked up"
        )
        # Must iterate over the inner dict and assign on env_cfg.rewards.<name>.weight
        self.assertIn(
            "_term.weight = _new_w",
            src,
            "loop body never writes back the new weight"
        )
        # Must use getattr to look up the term by name (safe against disable_zero_weight_rewards)
        self.assertIn(
            "getattr(env_cfg.rewards, _term_name, None)",
            src,
            "must use getattr (term may have been pruned by disable_zero_weight_rewards)"
        )

    def test_train_moe_command_range_overrides_present(self):
        """_RANK_COMMAND_RANGE_OVERRIDES must exist and gate rank 0 vs rest."""
        src = TRAIN_MOE.read_text()
        self.assertIn("_RANK_COMMAND_RANGE_OVERRIDES", src,
                      "_RANK_COMMAND_RANGE_OVERRIDES dict missing in train_moe.py")
        self.assertIn("_LATERAL_ON", src,  "lateral ON shortcut missing")
        self.assertIn("_LATERAL_OFF", src, "lateral OFF shortcut missing")
        # Must look up the dict
        self.assertIn("_RANK_COMMAND_RANGE_OVERRIDES.get(local_rank", src,
                      "command override dict never read")
        # Must assign onto env_cfg.commands.base_velocity.ranges
        self.assertIn("env_cfg.commands.base_velocity.ranges", src,
                      "command ranges never modified")

    def test_train_moe_reset_pose_overrides_present(self):
        """_RANK_RESET_POSE_OVERRIDES must exist for FLAT vs non-FLAT yaw."""
        src = TRAIN_MOE.read_text()
        self.assertIn("_RANK_RESET_POSE_OVERRIDES", src,
                      "_RANK_RESET_POSE_OVERRIDES dict missing")
        self.assertIn("_YAW_RANDOM", src)
        self.assertIn("_YAW_FIXED",  src)
        self.assertIn("_RANK_RESET_POSE_OVERRIDES.get(local_rank", src,
                      "reset pose override dict never read")
        # Must write into pose_range
        self.assertIn("randomize_reset_base.params", src,
                      "reset event params never accessed")

    def test_train_moe_reward_func_override_present(self):
        """_RANK_REWARD_FUNC_OVERRIDES must wire feet_air_time → including_ang_z on FLAT."""
        src = TRAIN_MOE.read_text()
        self.assertIn("_RANK_REWARD_FUNC_OVERRIDES", src,
                      "_RANK_REWARD_FUNC_OVERRIDES dict missing")
        # The function name must be referenced (we expect rank 0 → feet_air_time_including_ang_z)
        self.assertIn("feet_air_time_including_ang_z", src,
                      "the ang_z variant of feet_air_time must be referenced for FLAT rank")
        # Dispatch must look it up and write back the func attr
        self.assertIn("_RANK_REWARD_FUNC_OVERRIDES.get(local_rank", src,
                      "func override dict never read")
        self.assertIn("_term.func = _new_func", src,
                      "func override never assigned to the reward term")

    def test_inspect_terrain_rank4_is_gap_stones_mix(self):
        tree = load_ast(INSPECT)
        list_node = find_assignment_value(tree, "RANK_TERRAIN_MAP")
        self.assertIsNotNone(list_node, "RANK_TERRAIN_MAP not found in inspect_terrain.py")
        self.assertIsInstance(list_node, ast.List)
        rank4 = list_node.elts[4]
        self.assertIsInstance(rank4, ast.Name)
        self.assertEqual(rank4.id, "GAP_STONES_MIX_TEACHER_TERRAINS_CFG")

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
    """Replicate the nested-dict dispatch logic on a mock env_cfg, verify behavior."""

    # Mirror the same dict that lives in train_moe.py. If this drifts,
    # test_train_moe_reward_override_dict_present will catch the source side;
    # this dict here is only for testing the dispatch *behavior*.
    _OVERRIDES = {
        2: {
            "ang_vel_xy_l2":          -0.01,
            "base_height_l2":          0.0,
            "feet_air_time":           1.5,
            "feet_height_body":       -0.5,
            "upward":                  0.0,
        },
        3: {
            "lin_vel_z_l2":           -2.0,
            "base_height_l2":         -0.5,
            "hipx_joint_pos_penalty": -0.6,
            "hipy_joint_pos_penalty": -0.3,
            "feet_air_time":           0.0,
            "feet_height_body":        0.0,
            "upward":                  0.08,
            "undesired_contacts":     -0.1,
        },
        4: {"is_terminated": 0.0},
    }

    # Base weights (mirrored from moe_teacher_env_cfg.py) — used to verify
    # un-overridden ranks keep these.
    _BASE_WEIGHTS = {
        "is_terminated":          -100.0,
        "lin_vel_z_l2":           -0.03,
        "ang_vel_xy_l2":          -0.05,
        "base_height_l2":         -0.3,
        "base_roll_l2":           -10.0,
        "hipx_joint_pos_penalty": -0.5,
        "hipy_joint_pos_penalty": -0.25,
        "knee_joint_pos_penalty": -0.1,
        "feet_air_time":           1.0,
        "feet_height_body":       -0.2,
        "upward":                  0.05,
        "undesired_contacts":     -0.3,
    }

    # ---- command range + reset yaw overrides (mirror train_moe.py) ----
    import math as _math_mod
    _LATERAL_ON  = {"lin_vel_y": (-1.0, 1.0), "heading": (-_math_mod.pi, _math_mod.pi)}
    _LATERAL_OFF = {"lin_vel_y": ( 0.0, 0.0), "heading": ( 0.0,           0.0)}
    _CMD_OVERRIDES = {
        0: _LATERAL_ON,
        1: _LATERAL_OFF, 2: _LATERAL_OFF, 3: _LATERAL_OFF, 4: _LATERAL_OFF,
        5: _LATERAL_OFF, 6: _LATERAL_OFF, 7: _LATERAL_OFF,
    }
    _YAW_RANDOM = {"yaw": (-_math_mod.pi, _math_mod.pi)}
    _YAW_FIXED  = {"yaw": ( 0.0,           0.0)}
    _RESET_OVERRIDES = {
        0: _YAW_RANDOM,
        1: _YAW_FIXED, 2: _YAW_FIXED, 3: _YAW_FIXED, 4: _YAW_FIXED,
        5: _YAW_FIXED, 6: _YAW_FIXED, 7: _YAW_FIXED,
    }

    # Base command + reset values (mirror moe_teacher_env_cfg.py)
    _BASE_CMD_RANGES = {
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_z": (-1.0, 1.0),
        "heading":   ( 0.0,  0.0),
    }
    _BASE_RESET_POSE = {
        "x":     (-0.5, 0.5),
        "y":     (-0.2, 0.2),
        "z":     ( 0.0, 0.0),
        "roll":  (-0.3, 0.3),
        "pitch": (-0.3, 0.3),
        "yaw":   (-_math_mod.pi, _math_mod.pi),
    }

    # Sentinel funcs live at MODULE level (see bottom of file). Class-level
    # function attrs auto-bind to instances which breaks `is` identity checks.

    def _make_mock_env_cfg(self):
        """Build a mock env_cfg with rewards + commands + reset events."""
        class _Term:
            def __init__(self, w, func=None):
                self.weight = w
                self.func = func
        class _Rewards:
            pass
        class _Terrain:
            terrain_generator = None
        class _Scene:
            terrain = _Terrain()
        class _Ranges:
            pass
        class _BaseVel:
            ranges = _Ranges()
        class _Commands:
            base_velocity = _BaseVel()
        class _RandReset:
            params = {"pose_range": dict(SimulatedOverrideBehavior._BASE_RESET_POSE)}
        class _Events:
            randomize_reset_base = _RandReset()
        class _EnvCfg:
            pass
        env_cfg = _EnvCfg()
        env_cfg.scene = _Scene()
        env_cfg.rewards = _Rewards()
        env_cfg.commands = _Commands()
        env_cfg.events = _Events()
        for name, w in self._BASE_WEIGHTS.items():
            # Only feet_air_time carries the func attr we care about for func-override tests
            base_func = _base_feet_air_func if name == "feet_air_time" else None
            setattr(env_cfg.rewards, name, _Term(w, func=base_func))
        # Reset ranges fresh (avoid sharing the dict reference across tests)
        env_cfg.events.randomize_reset_base = _RandReset()
        env_cfg.commands.base_velocity = _BaseVel()
        # Wire fresh _Ranges
        new_ranges = type("R", (), {})()
        for k, v in self._BASE_CMD_RANGES.items():
            setattr(new_ranges, k, v)
        env_cfg.commands.base_velocity.ranges = new_ranges
        return env_cfg

    def _simulate_dispatch(self, local_rank: int):
        """Run a faithful port of the nested-dispatch logic (rewards + funcs + cmd + reset)."""
        env_cfg = self._make_mock_env_cfg()
        terrain_map = ["FLAT", "STAIR", "PLAT", "SCAN", "STONES", "RAIL", "NOISE", "GRID"]
        if local_rank < len(terrain_map):
            env_cfg.scene.terrain.terrain_generator = terrain_map[local_rank]
            # Reward weights
            overrides = self._OVERRIDES.get(local_rank, {})
            for term_name, new_w in overrides.items():
                term = getattr(env_cfg.rewards, term_name, None)
                if term is not None and hasattr(term, "weight"):
                    term.weight = new_w
            # Reward funcs (FUNC_OVERRIDES is a module-level dict, not class attr)
            func_overrides = _FUNC_OVERRIDES.get(local_rank, {})
            for term_name, new_func in func_overrides.items():
                term = getattr(env_cfg.rewards, term_name, None)
                if term is not None and hasattr(term, "func"):
                    term.func = new_func
            # Command ranges
            cmd_overrides = self._CMD_OVERRIDES.get(local_rank, {})
            for k, v in cmd_overrides.items():
                if hasattr(env_cfg.commands.base_velocity.ranges, k):
                    setattr(env_cfg.commands.base_velocity.ranges, k, v)
            # Reset pose
            reset_overrides = self._RESET_OVERRIDES.get(local_rank, {})
            pose_range = env_cfg.events.randomize_reset_base.params.get("pose_range", {})
            for k, v in reset_overrides.items():
                if k in pose_range:
                    pose_range[k] = v
        return env_cfg

    def test_rank_2_platform_overrides_applied(self):
        env_cfg = self._simulate_dispatch(local_rank=2)
        self.assertEqual(env_cfg.scene.terrain.terrain_generator, "PLAT")
        # Overridden:
        self.assertEqual(env_cfg.rewards.ang_vel_xy_l2.weight, -0.01)
        self.assertEqual(env_cfg.rewards.base_height_l2.weight, 0.0)
        self.assertEqual(env_cfg.rewards.feet_air_time.weight, 1.5)
        self.assertEqual(env_cfg.rewards.feet_height_body.weight, -0.5)
        self.assertEqual(env_cfg.rewards.upward.weight, 0.0)
        # Non-platform overrides kept at base:
        self.assertEqual(env_cfg.rewards.is_terminated.weight, -100.0)
        self.assertEqual(env_cfg.rewards.lin_vel_z_l2.weight, -0.03)

    def test_rank_3_scan_overrides_applied(self):
        env_cfg = self._simulate_dispatch(local_rank=3)
        self.assertEqual(env_cfg.scene.terrain.terrain_generator, "SCAN")
        self.assertEqual(env_cfg.rewards.lin_vel_z_l2.weight, -2.0)
        self.assertEqual(env_cfg.rewards.base_height_l2.weight, -0.5)
        self.assertEqual(env_cfg.rewards.hipx_joint_pos_penalty.weight, -0.6)
        self.assertEqual(env_cfg.rewards.hipy_joint_pos_penalty.weight, -0.3)
        self.assertEqual(env_cfg.rewards.feet_air_time.weight, 0.0)
        self.assertEqual(env_cfg.rewards.feet_height_body.weight, 0.0)
        self.assertEqual(env_cfg.rewards.upward.weight, 0.08)
        self.assertEqual(env_cfg.rewards.undesired_contacts.weight, -0.1)
        # is_terminated NOT in scan overrides → base value
        self.assertEqual(env_cfg.rewards.is_terminated.weight, -100.0)

    def test_rank_4_keeps_only_is_terminated_override(self):
        env_cfg = self._simulate_dispatch(local_rank=4)
        self.assertEqual(env_cfg.scene.terrain.terrain_generator, "STONES")
        self.assertEqual(env_cfg.rewards.is_terminated.weight, 0.0)
        # All other weights stay at base
        for name, base_w in self._BASE_WEIGHTS.items():
            if name == "is_terminated":
                continue
            self.assertEqual(
                getattr(env_cfg.rewards, name).weight, base_w,
                f"rank 4: rewards.{name}.weight = {getattr(env_cfg.rewards, name).weight}, expected {base_w}"
            )

    def test_untouched_ranks_keep_full_base_weights(self):
        for rank in [0, 1, 5, 6, 7]:
            env_cfg = self._simulate_dispatch(local_rank=rank)
            for name, base_w in self._BASE_WEIGHTS.items():
                self.assertEqual(
                    getattr(env_cfg.rewards, name).weight, base_w,
                    f"rank {rank}: rewards.{name}.weight = {getattr(env_cfg.rewards, name).weight}, expected base {base_w}"
                )

    def test_missing_term_is_skipped_safely(self):
        """A reward term pruned by disable_zero_weight_rewards must not crash dispatch."""
        env_cfg = self._make_mock_env_cfg()
        # Pretend `upward` was pruned (e.g. by disable_zero_weight_rewards)
        delattr(env_cfg.rewards, "upward")
        # Apply rank 2 overrides (which include "upward") — should not raise.
        overrides = self._OVERRIDES[2]
        for term_name, new_w in overrides.items():
            term = getattr(env_cfg.rewards, term_name, None)
            if term is not None and hasattr(term, "weight"):
                term.weight = new_w
        # Confirm other overrides still applied (upward absent doesn't block the rest)
        self.assertEqual(env_cfg.rewards.feet_air_time.weight, 1.5)
        self.assertEqual(env_cfg.rewards.feet_height_body.weight, -0.5)
        # Confirm upward truly doesn't exist (i.e. the override didn't accidentally create it)
        self.assertFalse(hasattr(env_cfg.rewards, "upward"))

    # ---- New: per-rank command + reset overrides ----

    def test_rank_0_flat_gets_lateral_and_random_yaw(self):
        env_cfg = self._simulate_dispatch(local_rank=0)
        # Commands: lateral ON, heading full
        self.assertEqual(env_cfg.commands.base_velocity.ranges.lin_vel_y, (-1.0, 1.0))
        self.assertEqual(env_cfg.commands.base_velocity.ranges.heading[0], -self._math_mod.pi)
        self.assertEqual(env_cfg.commands.base_velocity.ranges.heading[1],  self._math_mod.pi)
        # Reset: random yaw
        yaw = env_cfg.events.randomize_reset_base.params["pose_range"]["yaw"]
        self.assertEqual(yaw[0], -self._math_mod.pi)
        self.assertEqual(yaw[1],  self._math_mod.pi)

    def test_non_flat_ranks_get_no_lateral_no_yaw(self):
        for rank in [1, 2, 3, 4, 5, 6, 7]:
            env_cfg = self._simulate_dispatch(local_rank=rank)
            with self.subTest(rank=rank):
                # Commands locked
                self.assertEqual(
                    env_cfg.commands.base_velocity.ranges.lin_vel_y, (0.0, 0.0),
                    f"rank {rank} lin_vel_y not zeroed"
                )
                self.assertEqual(
                    env_cfg.commands.base_velocity.ranges.heading, (0.0, 0.0),
                    f"rank {rank} heading not zeroed"
                )
                # Reset yaw locked
                self.assertEqual(
                    env_cfg.events.randomize_reset_base.params["pose_range"]["yaw"],
                    (0.0, 0.0),
                    f"rank {rank} reset yaw not zeroed"
                )

    def test_non_yaw_pose_range_keys_untouched(self):
        """Other pose_range keys (x, y, z, roll, pitch) should be unchanged."""
        for rank in [0, 4]:
            env_cfg = self._simulate_dispatch(local_rank=rank)
            pose = env_cfg.events.randomize_reset_base.params["pose_range"]
            self.assertEqual(pose["x"],     self._BASE_RESET_POSE["x"])
            self.assertEqual(pose["y"],     self._BASE_RESET_POSE["y"])
            self.assertEqual(pose["z"],     self._BASE_RESET_POSE["z"])
            self.assertEqual(pose["roll"],  self._BASE_RESET_POSE["roll"])
            self.assertEqual(pose["pitch"], self._BASE_RESET_POSE["pitch"])

    def test_other_command_ranges_untouched(self):
        """lin_vel_x and ang_vel_z stay at base for all ranks."""
        for rank in range(8):
            env_cfg = self._simulate_dispatch(local_rank=rank)
            self.assertEqual(
                env_cfg.commands.base_velocity.ranges.lin_vel_x,
                self._BASE_CMD_RANGES["lin_vel_x"],
                f"rank {rank} lin_vel_x changed unexpectedly"
            )
            self.assertEqual(
                env_cfg.commands.base_velocity.ranges.ang_vel_z,
                self._BASE_CMD_RANGES["ang_vel_z"],
                f"rank {rank} ang_vel_z changed unexpectedly"
            )

    # ---- New: per-rank reward function override ----

    def test_rank_0_flat_swaps_feet_air_time_func(self):
        env_cfg = self._simulate_dispatch(local_rank=0)
        self.assertIs(env_cfg.rewards.feet_air_time.func, _ang_z_feet_air_func,
                      "rank 0: feet_air_time.func should be the including_ang_z variant")
        self.assertEqual(env_cfg.rewards.feet_air_time.func.__name__,
                         "feet_air_time_including_ang_z")

    def test_non_flat_ranks_keep_feet_air_time_func(self):
        for rank in [1, 2, 3, 4, 5, 6, 7]:
            env_cfg = self._simulate_dispatch(local_rank=rank)
            with self.subTest(rank=rank):
                self.assertIs(env_cfg.rewards.feet_air_time.func, _base_feet_air_func,
                              f"rank {rank}: feet_air_time.func should stay at base (curriculum-gated)")

    def test_out_of_range_rank_does_nothing(self):
        """rank 8+ → terrain_map index OOB: no terrain swap, no override applied."""
        env_cfg = self._simulate_dispatch(local_rank=8)
        self.assertIsNone(env_cfg.scene.terrain.terrain_generator,
                          "rank 8 should not have a terrain assigned")
        # All weights stay at base since the override dict has no entry for rank 8
        # (and even if it did, the terrain-map index would fail first)
        for name, base_w in self._BASE_WEIGHTS.items():
            self.assertEqual(getattr(env_cfg.rewards, name).weight, base_w,
                             f"rank 8: rewards.{name}.weight changed unexpectedly")


# Module-level sentinel funcs (avoid bound-method weirdness when used in `is` checks)
def _base_feet_air_func(*a, **kw):
    """Mock for mdp.feet_air_time_curriculum (curriculum-gated, zeros on flat)."""
    return 0.0
_base_feet_air_func.__name__ = "feet_air_time_curriculum"


def _ang_z_feet_air_func(*a, **kw):
    """Mock for mdp.feet_air_time_including_ang_z (no curriculum gate)."""
    return 0.0
_ang_z_feet_air_func.__name__ = "feet_air_time_including_ang_z"


# Module-level dispatch dict (referenced by SimulatedOverrideBehavior._simulate_dispatch)
_FUNC_OVERRIDES = {
    0: {"feet_air_time": _ang_z_feet_air_func},
}


if __name__ == "__main__":
    unittest.main(verbosity=2)
