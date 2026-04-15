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
    id="Flat-MLP2MoE-Student-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_teacher_env_cfg:DeeproboticsM20MoETeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:MlpToMoeDistillationCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-M20-Teacher-Acrobatic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_acrobatic_env_cfg:DeeproboticsM20TeacherAcrobaticEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_acrobatic_cfg:DeeproboticsM20AcrobaticPPORunnerCfg",
    },
)


gym.register(
    id="MoE-Scan-Teacher-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_scan_env_cfg:DeeproboticsM20TeacherScanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:ScanMoEPPOCfg",
    },
)

gym.register(
    id="MoE-Elevation-Teacher-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_elevation_env_cfg:DeeproboticsM20TeacherElevationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:EleMoEPPOCfg",
    },
)