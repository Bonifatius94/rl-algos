from time import time
import gym

from algos.dreamer.config import DreamerSettings, DreamerTrainSettings
from algos.dreamer.env import DreamerEnvWrapper, play_episode
from algos.dreamer.model import DreamerModel
from algos.dreamer.logging import DreamerTensorboardLogger
from algos.dreamer.training import train

from algos.ppo.config import PPOTrainingSettings
from algos.ppo.agent import PPOAgent


def train_headless():
    orig_env = gym.make("ALE/MsPacman-v5")
    settings = DreamerSettings([1], [64, 64, 3], [4, 4, 64], [32, 32], [512])
    tb_logger = DreamerTensorboardLogger()
    model = DreamerModel(settings, loss_logger=tb_logger.log_loss)
    env = DreamerEnvWrapper(orig_env, settings, model)

    ppo_config = PPOTrainingSettings(
        obs_shape=settings.repr_dims, num_actions=9, n_envs=128)
    train_config = DreamerTrainSettings(
        n_envs=ppo_config.n_envs,
        steps_per_update=ppo_config.steps_per_update)
    ppo_agent = PPOAgent(ppo_config)

    def eval_model(epoch: int):
        log_interval = 100
        max_steps = 500
        step = 0

        def render_step(frame_orig, frame_hall):
            nonlocal step
            log_step = epoch * max_steps + step
            if (step + 1) % log_interval == 0:
                tb_logger.log_frames(log_step, frame_orig, frame_hall)
            step += 1

        env.render_output = render_step
        env.seed(42)
        play_episode(env, render=True, max_steps=max_steps)
        env.render_output = lambda f1, f2: None
        env.seed(int(time()))

    train(train_config, env, ppo_agent, tb_logger, eval_model)


if __name__ == "__main__":
    train_headless()
