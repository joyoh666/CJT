from dataclasses import dataclass

@dataclass
class DQNConfig:
    total_env_steps: int = 1_500_000
    num_envs: int = 16
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 50_000
    learning_starts: int = 2_000
    train_every: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 2_000
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 150_000
