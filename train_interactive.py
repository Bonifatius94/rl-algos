import gym

from algos.dreamer.config import DreamerSettings, DreamerTrainSettings
from algos.dreamer.env import DreamerEnvWrapper, play_episode
from algos.dreamer.training import train
from algos.dreamer.display import DreamerDebugDisplay
from algos.dreamer.model import DreamerModel
from algos.ppo import PPOAgent, PPOTrainingSettings


def train_interactive():
    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [64, 64, 3], [32, 32], [512], [128])
    model = DreamerModel(settings)
    env = DreamerEnvWrapper(orig_env, settings, model=model)

    ui_env = DreamerEnvWrapper(orig_env, settings, model=model)
    display = DreamerDebugDisplay(settings.obs_dims[1], settings.obs_dims[0], 8)
    ui_env.render_output = display.next_frame
    render = lambda ep: play_episode(ui_env, render=True, max_steps=100)

    ppo_config = PPOTrainingSettings(
        obs_shape=settings.repr_dims, num_actions=6, n_envs=128)
    train_config = DreamerTrainSettings(
        n_envs=ppo_config.n_envs,
        steps_per_update=ppo_config.steps_per_update)
    ppo_agent = PPOAgent(ppo_config)

    train(train_config, env, ppo_agent, render)


if __name__ == "__main__":
    train_interactive()
