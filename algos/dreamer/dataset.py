from typing import Callable, Any
from dataclasses import dataclass

import numpy as np

from algos.dreamer.env import \
    DreamerEnvWrapper, play_episode


@dataclass
class Trajectory:
    states: np.ndarray
    zs: np.ndarray
    hs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terms: np.ndarray

    def __post_init__(self):
        if self.actions.shape[0] != self.rewards.shape[0]:
            raise ValueError(
                "Invalid trajectory! The amount of actions and rewards needs to be the same!")
        if self.actions.shape[0] + 1 != self.states.shape[0] \
                or self.states.shape[0] != self.zs.shape[0] \
                or self.states.shape[0] != self.hs.shape[0]:
            raise ValueError(
                f"Invalid trajectory! Expected {self.rewards.shape[0] + 1} states!")

    @property
    def timesteps(self) -> int:
        return self.actions.shape[0]


def sample_trajectory(
        env: DreamerEnvWrapper,
        actor: Callable[[Any], Any]) -> Trajectory:
    traj_states = []
    traj_zs = []
    traj_hs = []
    traj_actions = []
    traj_rewards = []

    def append_timestep(s1, z1, h1, a, r):
        traj_states.append(s1)
        traj_zs.append(z1)
        traj_hs.append(h1)
        if a is not None:
            traj_actions.append(a)
            traj_rewards.append(r)

    env.collect_step = append_timestep
    play_episode(env, actor, max_steps=100)
    env.collect_step = lambda s1, z1, h1, a, r: None

    terms = np.zeros((len(traj_rewards)))
    terms[-1] = 1.0

    return Trajectory(
        np.array(traj_states), np.array(traj_zs), np.array(traj_hs),
        np.array(traj_actions), np.array(traj_rewards), terms)
