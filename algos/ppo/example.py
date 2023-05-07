from typing import Tuple

import gym
import numpy as np

from algos.common.vec_env import VecEnvEpisodeStats, VecEnvTrainingSession
from algos.ppo.config import PPOTrainingSettings
from algos.ppo.model import PPOModel
from algos.ppo.rollout import PPOVecEnvRolloutBuffer
from algos.ppo.logging import PPOTensorboardLogging


def train_pong():
    # obs space: (210, 160, 3), uint8)
    # action space: Discrete(6)
    config = PPOTrainingSettings([210, 160, 3], 6)
    make_env = lambda: gym.make("ALE/Pong-v5")
    vec_env = gym.vector.AsyncVectorEnv([make_env for _ in range(config.n_envs)])

    tb_logger = PPOTensorboardLogging()
    model = PPOModel(config, tb_logger.log_training_loss, tb_logger.flush_losses)
    exp_buffer = PPOVecEnvRolloutBuffer(config, model.train)
    episode_logger = VecEnvEpisodeStats(config.n_envs, tb_logger.log_episode)

    sample_actions = lambda pred: \
        [int(np.random.choice(config.num_actions, 1, p=prob_dist)) for prob_dist in pred[0]]
    encode_obs = lambda obs: obs / 127.5 - 1.0
    snapshot_model = lambda step: model.save(f"model/ppo_{step}")
    session = VecEnvTrainingSession(
        vec_env, model.predict, encode_obs, sample_actions, exp_buffer.append_step,
        episode_logger.log_step, snapshot_model, config.model_snapshot_interval)

    session.training(config.train_steps)
    model.save("model/ppo_final")


if __name__ == '__main__':
    train_pong()
