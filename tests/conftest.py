"""Shared test infrastructure: lightweight imports of moe_terrain.py without
loading the full IsaacLab Omniverse stack.

Provides:
    - REPO_ROOT, MOE_TERRAIN_PATH, TEACHER_CFG_PATH, ASSET_CFG_PATH constants
    - install_isaaclab_stubs(): patch sys.modules to fake out heavy IsaacLab imports
    - load_moe_terrain(): import moe_terrain.py and return the module
    - make_fake_obs(): build a TensorDict-like dict matching the env's obs structure
    - make_actor_critic(): instantiate SplitMoEActorCritic with synthetic obs
"""
from __future__ import annotations

import os
import sys
import types
import importlib.util
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Path discovery (works no matter where pytest is invoked from)
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
MOE_TERRAIN_PATH = (
    REPO_ROOT
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/moe_terrain.py"
)
TEACHER_CFG_PATH = (
    REPO_ROOT
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/moe_teacher_env_cfg.py"
)
ASSET_CFG_PATH = REPO_ROOT / "source/rl_training/rl_training/assets/deeprobotics.py"


# ---------------------------------------------------------------------------
# IsaacLab stub injection (avoids `pxr` / Omniverse Kit boot)
# ---------------------------------------------------------------------------
def install_isaaclab_stubs() -> None:
    """Inject stub modules for isaaclab.utils and isaaclab_rl.rsl_rl so that
    moe_terrain.py can be imported without launching SimulationApp."""
    if "isaaclab.utils" in sys.modules and getattr(
        sys.modules["isaaclab.utils"], "__is_test_stub__", False
    ):
        return  # already installed

    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__is_test_stub__ = True
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_utils.__is_test_stub__ = True
    # @configclass is a no-op (just returns the class).
    isaaclab_utils.configclass = lambda cls: cls

    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.utils"] = isaaclab_utils

    isaaclab_rl = types.ModuleType("isaaclab_rl")
    isaaclab_rl.__is_test_stub__ = True
    isaaclab_rl_rsl = types.ModuleType("isaaclab_rl.rsl_rl")
    isaaclab_rl_rsl.__is_test_stub__ = True

    class _StubBaseCfg:
        """Permissive base-class — swallows kwargs."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    for name in [
        "RslRlOnPolicyRunnerCfg",
        "RslRlPpoActorCriticCfg",
        "RslRlPpoAlgorithmCfg",
        "RslRlDistillationRunnerCfg",
        "RslRlDistillationAlgorithmCfg",
    ]:
        setattr(isaaclab_rl_rsl, name, _StubBaseCfg)

    sys.modules["isaaclab_rl"] = isaaclab_rl
    sys.modules["isaaclab_rl.rsl_rl"] = isaaclab_rl_rsl


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------
_loaded_module = None


def load_moe_terrain():
    """Return the imported moe_terrain.py module (singleton)."""
    global _loaded_module
    if _loaded_module is not None:
        return _loaded_module
    install_isaaclab_stubs()
    sys.path.insert(0, str(REPO_ROOT / "source/rl_training"))
    spec = importlib.util.spec_from_file_location(
        "moe_terrain_under_test", str(MOE_TERRAIN_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _loaded_module = mod
    return mod


# ---------------------------------------------------------------------------
# Synthetic obs builder
# ---------------------------------------------------------------------------
# Layout (from moe_teacher_env_cfg.py PolicyCfg / NoisyElevationCfg / CriticCfg):
#   - "policy"          : (B, 57)  -- ang_vel(3) + gravity(3) + cmd(3) + jp(16) + jv(16) + last_a(16)
#   - "critic"          : (B, 60)  -- base_lin_vel(3) prefix + the 57 above
#   - "estimator"       : (B, 57*H) -- policy obs over history H
#   - "noisy_elevation" : (B, 187 + 12*21) = (B, 439) -- elevation grid(187) + scan(252)
POLICY_DIM = 57
CRITIC_DIM = 60
ELEV_DIM = 187   # 11 * 17
SCAN_CH = 12     # 6 fwd + 6 bwd
SCAN_RAY = 21
SCAN_DIM = SCAN_CH * SCAN_RAY  # 252
NOISY_ELEV_DIM = ELEV_DIM + SCAN_DIM  # 439
NUM_ACTIONS = 16  # 12 leg + 4 wheel


def make_fake_obs(B: int = 4, history: int = 5, device: str = "cpu", seed: int = 0):
    """Build a TensorDict-like dict matching the env's obs structure."""
    g = torch.Generator(device=device).manual_seed(seed)
    return {
        "policy":         torch.randn(B, POLICY_DIM,  generator=g, device=device),
        "critic":         torch.randn(B, CRITIC_DIM,  generator=g, device=device),
        "estimator":      torch.randn(B, POLICY_DIM * history, generator=g, device=device),
        "noisy_elevation": torch.randn(B, NOISY_ELEV_DIM, generator=g, device=device),
    }


DEFAULT_OBS_GROUPS = {
    "policy":          ["policy"],
    "critic":          ["critic"],
    "estimator":       ["estimator"],
    "noisy_elevation": ["noisy_elevation"],
}


def make_actor_critic(
    *,
    B: int = 4,
    latent_dim: int = 32,             # small for fast tests
    actor_hidden_dims=(64, 32),
    critic_hidden_dims=(64, 32),
    sym_loss_coef: float = 0.3,        # enable sym path (sym_enabled=True)
    rnn_type: str = "gru",
    seed: int = 0,
    **extra_kwargs,
):
    """Instantiate SplitMoEActorCritic with synthetic obs and small dims for fast unit tests."""
    mod = load_moe_terrain()
    torch.manual_seed(seed)
    obs = make_fake_obs(B=B, history=5, seed=seed)

    model = mod.SplitMoEActorCritic(
        obs,
        DEFAULT_OBS_GROUPS,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=list(actor_hidden_dims),
        critic_hidden_dims=list(critic_hidden_dims),
        activation="elu",
        init_noise_std=1.0,
        num_wheel_experts=3,
        num_leg_experts=6,
        num_leg_actions=12,
        latent_dim=latent_dim,
        rnn_type=rnn_type,
        aux_loss_coef=0.01,
        sym_loss_coef=sym_loss_coef,
        # AE / scan flags
        use_elevation_ae=True,
        elevation_dim=ELEV_DIM,
        use_multilayer_scan=True,
        num_scan_channels=SCAN_CH,
        num_scan_rays=SCAN_RAY,
        # Estimator (3-D vel pred)
        estimator_output_dim=3,
        estimator_hidden_dims=[64, 32],
        estimator_obs_normalization=True,
        # Other
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        feed_estimator_to_policy=False,
        feed_ae_to_policy=False,
        blind_vision=False,
        is_student_mode=False,
        init_noise_legs=0.2,
        init_noise_wheels=1.5,
        **extra_kwargs,
    )
    return model, obs


def make_ppo(model=None, **extra_kwargs):
    """Instantiate SplitMoEPPO around a SplitMoEActorCritic. If model is None,
    builds one with default test settings."""
    mod = load_moe_terrain()
    if model is None:
        model, _ = make_actor_critic()
    ppo = mod.SplitMoEPPO(
        model,
        num_learning_epochs=2,
        num_mini_batches=2,
        device="cpu",
        **extra_kwargs,
    )
    return ppo


# ---------------------------------------------------------------------------
# Pretty printing helpers (used by the standalone scripts)
# ---------------------------------------------------------------------------
def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def passed(msg: str):
    print(f"  ✓ {msg}")


def failed(msg: str):
    print(f"  ✗ {msg}")
