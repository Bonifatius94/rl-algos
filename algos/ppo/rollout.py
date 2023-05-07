from typing import Callable, Tuple, List
from dataclasses import dataclass, field

import numpy as np

from algos.ppo.config import PPOTrainingSettings


Observation = np.ndarray
Action = int
SarsExperience = Tuple[Observation, Action, float, Observation, bool]

BatchedObservations = np.ndarray
BatchedActions = np.ndarray
BatchedProbDists = np.ndarray
BatchedBaselines = np.ndarray
BatchedPredictions = Tuple[BatchedProbDists, BatchedBaselines]
BatchedAdvantages = np.ndarray
BatchedRewards = np.ndarray
BatchedReturns = np.ndarray
BatchedDones = np.ndarray
BatchedSarsExps = Tuple[BatchedObservations, BatchedActions,
    BatchedRewards, BatchedObservations, BatchedDones]
TrainingBatch = Tuple[BatchedObservations, BatchedActions, BatchedReturns,
    BatchedAdvantages, BatchedProbDists, BatchedBaselines]


@dataclass
class PPOVecEnvRolloutBuffer:
    config: PPOTrainingSettings
    consume_batch: Callable[[List[TrainingBatch]], None]
    prob_dist_cache: np.ndarray = field(init=False)
    baseline_cache: np.ndarray = field(init=False)
    s0_cache: np.ndarray = field(init=False)
    actinon_cache: np.ndarray = field(init=False)
    reward_cache: np.ndarray = field(init=False)
    done_cache: np.ndarray = field(init=False)
    step: int = 0

    def __post_init__(self):
        steps, n_envs = self.config.steps_per_update + 1, self.config.n_envs
        obs_shape = [steps, n_envs] + [d for d in self.config.obs_shape]
        self.prob_dist_cache = np.zeros((steps, n_envs, self.config.num_actions), dtype=np.float32)
        self.baseline_cache = np.zeros((steps, n_envs), dtype=np.float32)
        self.s0_cache = np.zeros(obs_shape, dtype=np.float32)
        self.actinon_cache = np.zeros((steps, n_envs), dtype=np.int64)
        self.reward_cache = np.zeros((steps, n_envs), dtype=np.float32)
        self.done_cache = np.zeros((steps, n_envs), dtype=np.bool8)

    def append_step(self, preds: BatchedPredictions, exps: BatchedSarsExps):
        prob_dists, baselines = preds
        states_before, actions, rewards, _, dones = exps
        t = self.step

        self.prob_dist_cache[t] = prob_dists
        self.baseline_cache[t] = baselines
        self.s0_cache[t] = states_before
        self.actinon_cache[t] = np.squeeze(actions) # TODO: add support for multi-dim actions
        self.reward_cache[t] = rewards
        self.done_cache[t] = dones
        self.step += 1

        if self.step == self.config.steps_per_update + 1:
            batch = self._sample_batch()
            mini_batches = self._partition_minibatches(batch)
            self.consume_batch(mini_batches)

            # info: sampling requires the baseline of last s1
            #       -> cache 1 more step, assign it to beginning of the next batch
            self.prob_dist_cache[0] = self.prob_dist_cache[t]
            self.baseline_cache[0] = self.baseline_cache[t]
            self.s0_cache[0] = self.s0_cache[t]
            self.actinon_cache[0] = self.actinon_cache[t]
            self.reward_cache[0] = self.reward_cache[t]
            self.done_cache[0] = self.done_cache[t]
            self.step = 1

    def _sample_batch(self) -> TrainingBatch:
        batch_steps = self.config.steps_per_update
        no_t_dim = lambda old_shape: [-1] + [d for d in old_shape[2:]]
        states = self.s0_cache[:batch_steps].reshape(no_t_dim(self.s0_cache.shape))
        actions = self.actinon_cache[:batch_steps].reshape(-1)
        advantages = self._compute_advantages()[:batch_steps].reshape(-1)
        returns = self.baseline_cache[:batch_steps].reshape(-1) + advantages
        prob_dists_old = self.prob_dist_cache[:batch_steps].reshape(no_t_dim(self.prob_dist_cache.shape))
        baselines_old = self.baseline_cache[:batch_steps].reshape(-1)
        return states, actions, returns, advantages, prob_dists_old, baselines_old

    def _compute_advantages(self) -> BatchedAdvantages:
        baselines, rewards = self.baseline_cache, self.reward_cache
        reward_discount, gae_discount = self.config.reward_discount, self.config.gae_discount
        advantages, last_gae_lam = np.zeros_like(rewards), 0
        for t in reversed(range(self.config.steps_per_update)):
            next_nonterminal = 1.0 - self.done_cache[t+1]
            delta = rewards[t] + baselines[t+1] * next_nonterminal * reward_discount - baselines[t]
            last_gae_lam = delta + next_nonterminal * reward_discount * gae_discount * last_gae_lam
            advantages[t] = last_gae_lam
        return advantages

    def _partition_minibatches(self, big_batch: TrainingBatch) -> List[TrainingBatch]:
        num_examples = big_batch[0].shape[0]
        batch_size = self.config.batch_size
        states, actions, returns, advantages, prob_dists_old, baselines_old = big_batch
        ranges = [(i * batch_size, (i+1) * batch_size) for i in range(num_examples // batch_size)]
        mini_batches = [(states[i:j], actions[i:j], returns[i:j], advantages[i:j],
                         prob_dists_old[i:j], baselines_old[i:j])
                        for i, j in ranges]
        return mini_batches