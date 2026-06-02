# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Flat-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20FlatTrainerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Rough-MoE-Teacher-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:SplitMoEPPOCfg",
    },
)

# LocoMoE baseline (MoE-Loco, Huang et al., IROS 2025, arXiv:2503.08564).
# Head-to-head comparison vs SplitMoE: same env, same single-stage PPO loss,
# only the policy architecture differs (single shared gate over 6 full-action
# experts instead of split leg/wheel gates).
gym.register(
    id="Rough-LocoMoE-Teacher-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomoe_teacher_env_cfg:DeeproboticsM20LocoMoETeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.locomoe_terrain:LocoMoEPPOCfg",
    },
)

# v1: reduced experts (4 leg + 2 wheel) to force routing specialization.
# Shares env cfg with v0; only policy class capacity differs.
gym.register(
    id="Rough-MoE-Teacher-Deeprobotics-M20-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:SplitMoEReducedPPOCfg",
    },
)


gym.register(
    id="Rough-EleMoE-Teacher-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg_EleOnly",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:EleMoEPPOCfg",
    },
)

gym.register(
    id="Rough-ScanMoE-Teacher-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg_ScanOnly",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:ScanMoEPPOCfg",
    },
)

gym.register(
    id="Rough-MoE-Student-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:SplitMoEDistillationCfg",
    },
)

gym.register(
    id="Rough-MoE-SenseStudent-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:SplitMoESenseDistillationCfg",
    },
)


gym.register(
    id="Rough-MoE-Blind-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:BlindMoECfg",
    },
)


gym.register(
    id="Rough-MlpBaseline-Teacher-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:MlpBaselinePPOCfg",
    },
)


gym.register(
    id="Flat-MLP2MoE-Student-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:MlpToMoeDistillationCfg",
    },
)


# Obstacle-course eval tasks (single linear course, 6 patches in fixed order,
# 4 difficulty levels). Used by scripts/reinforcement_learning/rsl_rl/eval_course.py.
gym.register(
    id="Course-MoE-Teacher-Deeprobotics-M20-easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.course_env_cfg:DeeproboticsM20CourseEasyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:SplitMoEPPOCfg",
    },
)

gym.register(
    id="Course-MoE-Teacher-Deeprobotics-M20-med-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.course_env_cfg:DeeproboticsM20CourseMedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:SplitMoEPPOCfg",
    },
)

gym.register(
    id="Course-MoE-Teacher-Deeprobotics-M20-hard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.course_env_cfg:DeeproboticsM20CourseHardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:SplitMoEPPOCfg",
    },
)

gym.register(
    id="Course-MoE-Teacher-Deeprobotics-M20-extreme-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.course_env_cfg:DeeproboticsM20CourseExtremeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:SplitMoEPPOCfg",
    },
)


