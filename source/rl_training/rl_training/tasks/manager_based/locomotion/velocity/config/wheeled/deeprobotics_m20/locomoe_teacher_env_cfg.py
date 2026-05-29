# source/.../deeprobotics_m20/locomoe_teacher_env_cfg.py
"""Environment config wrapper for the LocoMoE baseline.

LocoMoE shares the *exact* environment (terrain, command, observation, reward,
events, terminations) with the SplitMoE teacher. This wrapper exists only so
that registering ``Rough-LocoMoE-Teacher-Deeprobotics-M20-v0`` has its own
``env_cfg_entry_point`` for clarity, and so the head-to-head comparison can
later swap env-cfg knobs (e.g. fewer iterations, ablations) without touching
the SplitMoE config.

Reference: MoE-Loco (Huang et al., IROS 2025), arXiv:2503.08564.
"""

from isaaclab.utils import configclass

from .moe_teacher_env_cfg import DeeproboticsM20MoETeacherEnvCfg


@configclass
class DeeproboticsM20LocoMoETeacherEnvCfg(DeeproboticsM20MoETeacherEnvCfg):
    """Same env as SplitMoE — variable-isolated: only the policy architecture changes."""

    pass
