from typing import Any

import gym
import numpy as np

from algos.common.vec_env import VecEnvEpisodeStats, VecEnvTrainingSession
from algos.ppo.config import PPOTrainingSettings
from algos.ppo.model import PPOModel
from algos.ppo.rollout import PPOVecEnvRolloutBuffer
from algos.ppo.logging import PPOTensorboardLogging


class PPOAgent:
    def __init__(self, config: PPOTrainingSettings):
        self.config = config

        tb_logger = PPOTensorboardLogging()
        self.model = PPOModel(config, tb_logger.log_training_loss, tb_logger.flush_losses)
        exp_buffer = PPOVecEnvRolloutBuffer(config, self.model.train)
        episode_logger = VecEnvEpisodeStats(config.n_envs, tb_logger.log_episode)

        encode_obs = lambda x: x
        sample_actions = lambda pred: \
            np.expand_dims(np.array([int(np.random.choice(config.num_actions, 1, p=prob_dist))
                      for prob_dist in pred[0]]), axis=1)

        self.predict_action = lambda obs: self.model.predict(np.expand_dims(obs, axis=0))
        self.session = VecEnvTrainingSession(
            None, self.model.predict, encode_obs, sample_actions,
            exp_buffer.append_step, episode_logger.log_step, lambda ep: None)

    def act(self, obs: Any) -> Any:
        prob_dist, _ = self.predict_action(obs)
        prob_dist = np.squeeze(prob_dist)
        return int(np.random.choice(self.config.num_actions, 1, p=prob_dist))

    def train(self, env: gym.vector.VectorEnv, steps: int):
        self.session.vec_env = env
        self.session.training(steps)

    def save(self, directory: str):
        self.model.save(directory)
