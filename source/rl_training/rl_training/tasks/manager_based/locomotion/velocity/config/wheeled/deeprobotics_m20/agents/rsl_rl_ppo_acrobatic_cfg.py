from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class AcrobaticPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO Algorithm configuration for Acrobatic Teacher."""
    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

@configclass
class AcrobaticPPOActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-Critic configuration for Acrobatic Teacher (Blind, RNN)."""
    class_name = "ActorCriticRecurrent" # 使用自带的 RNN 架构
    rnn_type = "gru"
    rnn_hidden_size = 256
    rnn_num_layers = 1
    actor_hidden_dims = [256, 128, 128]
    critic_hidden_dims = [256, 128, 128]
    activation = "elu"
    init_noise_std = 1.0
    empirical_normalization = True # 由于没有视觉，必须开启输入归一化

@configclass
class DeeproboticsM20AcrobaticPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for Acrobatic Teacher."""
    num_steps_per_env = 48 # 收集轨迹长度 (100Hz下大约 0.48s)
    max_iterations = 3000
    save_interval = 100
    experiment_name = "m20_teacher_acrobatic"
    empirical_normalization = False
    policy = AcrobaticPPOActorCriticCfg()
    algorithm = AcrobaticPPOAlgorithmCfg()