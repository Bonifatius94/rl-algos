from typing import Callable, Any, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from algos.dreamer.env import DreamerEnvWrapper, play_episode


def sample_trajectory(
        env: DreamerEnvWrapper,
        actor: Callable[[Any], Any],
        init_dt: int=5) -> Tuple[int, tf.data.Dataset]:
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

    return ep_dt, tf.data.Dataset.from_tensor_slices(tensors=(
        s0_init, a0_init, s1, a0, r1, terms))


def concat_datasets(datasets: List[tf.data.Dataset]) -> tf.data.Dataset:
    ds = datasets[0]
    for i in range(len(datasets) - 1):
        ds = ds.concatenate(datasets[i+1])
    return ds


@dataclass
class MixedDatasetGenerator:
    sample_traj: Callable[[], tf.data.Dataset]
    batch_size: int
    sample_batches: int
    max_cached_trajs: int=20
    resample_probs: List[float] = field(default_factory=lambda: [0.1, 0.45, 0.25, 0.15, 0.05])
    trajectories: List[Tuple[int, tf.data.Dataset]] = field(default_factory=list)

    @property
    def collected_batches(self) -> int:
        return sum([n for (n, ds) in self.trajectories]) // self.batch_size

    def sample(self) -> tf.data.Dataset:
        # sample initial trajectories to generate
        # at least one dataset of unique experiences
        while self.collected_batches < self.sample_batches:
            self.trajectories.append(self.sample_traj())

        # sample new trajectories
        num_new_trajs = int(np.random.choice(
            np.arange(len(self.resample_probs)), p=self.resample_probs, size=1))
        for _ in tqdm(range(num_new_trajs)):
            self.trajectories.append(self.sample_traj())

        # replace old trajectories with new ones
        if len(self.trajectories) > self.max_cached_trajs:
            num_trajs_to_discard = len(self.trajectories) - self.max_cached_trajs
            discarded_trajs = self.trajectories[:num_trajs_to_discard]
            self.trajectories = self.trajectories[num_trajs_to_discard:]
            for traj in discarded_trajs:
                del traj # TODO: check if this works

        # sample from all trajectories according to a random distribution
        num_examples = self.sample_batches * self.batch_size
        sample_dist = self._softmax(np.random.normal(-1, 1, size=(len(self.trajectories))))
        batches_to_take = np.around(sample_dist * num_examples).astype(dtype=np.int64)
        shuffled_datasets = [ds.repeat().shuffle(100) for (n, ds) in self.trajectories]
        fetched_datasets = [ds.take(n) for (n, ds) in zip(batches_to_take, shuffled_datasets)]
        return concat_datasets(fetched_datasets).shuffle(100).batch(self.batch_size)

    def _softmax(self, inputs: np.ndarray) -> np.ndarray:
        exps = np.exp(np.clip(inputs, -10, 10))
        exp_row_sums = np.sum(exps)
        return exps / (exp_row_sums + 1e-8)
