from typing import List
from dataclasses import dataclass


@dataclass
class DreamerSettings:
    action_dims: List[int]
    obs_dims: List[int]
    repr_dims: List[int]
    hidden_dims: List[int]
    enc_dims: List[int]
    dropout_rate: float = 0.2

    @property
    def repr_dims_flat(self) -> int:
        return self.repr_dims[0] * self.repr_dims[1]

    @property
    def repr_out_dims_flat(self) -> int:
        return self.repr_dims[0] * self.repr_dims[1] + self.hidden_dims[0]


@dataclass
class DreamerTrainSettings:
    num_world_trajs: int=1 # 10
    dream_steps: int=16
    batch_size: int=64
    epochs: int=1000
    world_epochs: int=2 # 10
    agent_epochs: int=5
    n_envs: int=32
    steps_per_update: int=512

    @property
    def agent_timesteps(self) -> int:
        return self.agent_epochs * self.steps_per_update + 1
