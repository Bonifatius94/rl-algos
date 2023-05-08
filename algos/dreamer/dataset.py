from typing import Callable, Any, List

import numpy as np
import tensorflow as tf

from algos.dreamer.env import DreamerEnvWrapper, play_episode


def sample_trajectory(
        env: DreamerEnvWrapper,
        actor: Callable[[Any], Any],
        init_dt: int=5) -> tf.data.Dataset:
    traj_states = []
    traj_actions = []
    traj_rewards = []
    step = 0

    def reset_trajectory():
        nonlocal traj_states, traj_actions, traj_rewards, step
        traj_states = []
        traj_actions = []
        traj_rewards = []
        step = 0

    # info: append_timestep() yields a trajectory with
    #       t steps and t rewards; for bootstrapping,
    #       it records initial +5 states and +4 actions
    def append_timestep(s1, a, r):
        nonlocal step
        if step < init_dt:
            traj_states.append(s1)
            if step >= 1:
                traj_actions.append(a)
        else:
            traj_states.append(s1)
            traj_actions.append(a)
            traj_rewards.append(r)
        step += 1

    # info: handle the case when the trajectory
    #      terminated during the initial steps
    env.collect_step = append_timestep
    valid_traj = False
    while not valid_traj:
        play_episode(env, actor)
        valid_traj = step > init_dt
        if not valid_traj:
            reset_trajectory()
    env.collect_step = lambda s1, a, r: None

    ep_dt = len(traj_rewards)
    terms = np.zeros(ep_dt)
    terms[-1] = 1.0
    s0_init = np.array([traj_states[t:t+init_dt] for t in range(ep_dt)])
    a0_init = np.array([traj_actions[t:t+init_dt-1] for t in range(ep_dt)])

    s1 = np.array([traj_states[t+init_dt] for t in range(ep_dt)])
    a0 = np.array([traj_actions[t+init_dt-1] for t in range(ep_dt)])
    r1 = np.array(traj_rewards)

    return tf.data.Dataset.from_tensor_slices(tensors=(
        s0_init, a0_init, s1, a0, r1, terms))


def concat_datasets(datasets: List[tf.data.Dataset]) -> tf.data.Dataset:
    ds = datasets[0]
    for i in range(len(datasets) - 1):
        ds = ds.concatenate(datasets[i+1])
    return ds
