from typing import Callable, Any, List
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

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
        if self.actions.shape[0] != self.rewards.shape[0] \
                and self.actions.shape[0] != self.zs.shape[0] \
                and self.actions.shape[0] != self.hs.shape[0] \
                and self.actions.shape[0] != self.actions.shape[0] \
                and self.actions.shape[0] != self.rewards.shape[0] \
                and self.actions.shape[0] != self.terms.shape[0]:
            raise ValueError(
                "Invalid trajectory! The amount of tuples needs to be the same for all arrays!")

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
        traj_zs.append(z1)
        traj_hs.append(np.squeeze(h1))
        if a is not None:
            traj_states.append(s1)
            traj_actions.append(a)
            traj_rewards.append(r)

    env.collect_step = append_timestep
    play_episode(env, actor, max_steps=100)
    env.collect_step = lambda s1, z1, h1, a, r: None

    terms = np.zeros((len(traj_rewards)))
    terms[-1] = 1.0

    return Trajectory(
        np.array(traj_states), np.array(traj_zs[:-1]), np.array(traj_hs[:-1]),
        np.array(traj_actions), np.array(traj_rewards), terms)


def generate_dataset(trajs: List[Trajectory]) -> tf.data.Dataset:

    def traj_to_dataset(traj: Trajectory) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(tensors=(
            traj.states, traj.zs, traj.hs,
            traj.actions, traj.rewards, traj.terms))

    datasets = [traj_to_dataset(t) for t in trajs]
    ds = datasets[0]
    for i in range(len(trajs) - 1):
        ds = ds.concatenate(datasets[i+1])
    return ds
