from typing import Callable, Tuple, Any
from dataclasses import dataclass, field

import gym
from tqdm import tqdm
import numpy as np


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
class VecEnvTrainingSession:
    vec_env: gym.vector.vector_env.VectorEnv
    model: Callable[[BatchedObservations], BatchedPredictions]
    encode_obs: Callable[[Any], BatchedObservations]
    sample_actions: Callable[[BatchedPredictions], BatchedActions]
    exps_consumer: Callable[[BatchedPredictions, BatchedSarsExps], None]
    log_timestep: Callable[[int, BatchedRewards, BatchedDones], None]
    snapshot_model: Callable[[int], None]
    epoch: int=0
    model_snapshot_interval: int=None

    def training(self, steps: int):
        obs = self.encode_obs(self.vec_env.reset())

        for step in tqdm(range(steps)):
            predictions = self.model(obs)
            actions = self.sample_actions(predictions)
            next_obs, rewards, dones, _ = self.vec_env.step(actions)
            next_obs = self.encode_obs(next_obs)
            sars_exps = (obs, actions, rewards, next_obs, dones)
            self.exps_consumer(predictions, sars_exps)
            self.log_timestep(steps * self.epoch + step, rewards, dones)
            obs = next_obs
            if self.model_snapshot_interval and (step + 1) % self.model_snapshot_interval == 0:
                self.snapshot_model(step)

        self.epoch += 1


@dataclass
class VecEnvEpisodeStats:
    n_envs: int
    log_episode: Callable[[int, float, float], None]
    ep_steps: np.ndarray = field(init=False)
    ep_rewards: np.ndarray = field(init=False)

    def __post_init__(self):
        self.ep_steps = np.zeros((self.n_envs), dtype=np.int64)
        self.ep_rewards = np.zeros((self.n_envs), dtype=np.float32)

    def log_step(self, step: int, rewards: BatchedRewards, dones: BatchedDones):
        self.ep_steps += np.ones((self.n_envs), dtype=np.int64)
        self.ep_rewards += rewards
        envs_in_final_state = np.where(dones)[0]
        num_dones = envs_in_final_state.shape[0]

        if num_dones > 0:
            avg_steps = sum(self.ep_steps[envs_in_final_state]) / num_dones
            avg_rewards = sum(self.ep_rewards[envs_in_final_state]) / num_dones
            self.log_episode(step, avg_rewards, avg_steps)
            self.ep_steps[envs_in_final_state] = 0
            self.ep_rewards[envs_in_final_state] = 0.0
