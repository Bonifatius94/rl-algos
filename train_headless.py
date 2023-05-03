import os
import gym

from algos.dreamer.config import DreamerSettings, DreamerTrainSettings
from algos.dreamer.env import DreamerEnvWrapper
from algos.dreamer.logging import record_episode
from algos.dreamer.training import train
from algos.ppo import PPOTrainingSettings, PPOAgent


def train_headless():
    VIDEOS_ROOTDIR = "./videos"
    if os.path.exists(VIDEOS_ROOTDIR):
        os.system(f"rm -rf {VIDEOS_ROOTDIR}")
    os.mkdir(VIDEOS_ROOTDIR)

    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [64, 64, 3], [32, 32], [512], [128])
    train_config = DreamerTrainSettings()
    env = DreamerEnvWrapper(orig_env, settings)
    render = lambda ep: record_episode(env, os.path.join(VIDEOS_ROOTDIR, f"ep_{ep+1}.avi"))

    ppo_config = PPOTrainingSettings(obs_shape=settings.repr_dims, num_actions=6)
    ppo_agent = PPOAgent(ppo_config)
    train(train_config, env, ppo_agent, render)


if __name__ == "__main__":
    train_headless()
