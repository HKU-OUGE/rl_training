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

    def __post_init__(self):
        super().__post_init__()
        # parent only calls disable_zero_weight_rewards when class name matches
        # exactly, so subclasses miss it; do it explicitly here.
        self.disable_zero_weight_rewards()
        # belt-and-suspenders: nuke specific weight=0 terms with broken
        # body_names defaults that disable_zero_weight_rewards may have missed
        # (e.g. because a per-rank apply path resets them after disable runs).
        self.rewards.feet_air_time_variance = None
        self.rewards.feet_distance_y_exp = None
