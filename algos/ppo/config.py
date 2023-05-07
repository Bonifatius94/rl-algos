from dataclasses import dataclass
from typing import List


@dataclass
class PPOTrainingSettings:
    obs_shape: List[int]
    num_actions: int
    total_steps: int = 10_000_000
    learn_rate: float = 3e-4
    reward_discount: float = 0.99
    gae_discount: float = 0.95
    prob_clip: float = 0.2
    value_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    norm_advantages: bool = True
    clip_values: bool = True
    batch_size: int = 64
    n_envs: int = 32
    steps_per_update: int = 512
    update_epochs: int = 4
    num_model_snapshots: int = 20

    def __post_init__(self):
        if self.n_envs * self.steps_per_update % self.batch_size != 0:
            print((f"WARNING: training examples will be cut because "
                f"{self.n_envs} environments * {self.steps_per_update} steps per update interval "
                f"isn't divisible by a mini-batch size of {self.batch_size}!"))

    @property
    def train_steps(self) -> int:
        return self.total_steps // self.n_envs

    @property
    def model_snapshot_interval(self) -> int:
        return self.train_steps // self.num_model_snapshots
