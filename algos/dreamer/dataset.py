from typing import Callable, Any, List

import numpy as np
import tensorflow as tf

from algos.dreamer.env import DreamerEnvWrapper, play_episode


def sample_trajectory(
        env: DreamerEnvWrapper,
        actor: Callable[[Any], Any]) -> tf.data.Dataset:
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
    play_episode(env, actor)
    env.collect_step = lambda s1, z1, h1, a, r: None

    terms = np.zeros((len(traj_rewards)))
    terms[-1] = 1.0

    return tf.data.Dataset.from_tensor_slices(tensors=(
        np.array(traj_states), np.array(traj_zs[:-1]), np.array(traj_hs[:-1]),
        np.array(traj_actions), np.array(traj_rewards), terms))


def concat_datasets(datasets: List[tf.data.Dataset]) -> tf.data.Dataset:
    ds = datasets[0]
    for i in range(len(datasets) - 1):
        ds = ds.concatenate(datasets[i+1])
    return ds
