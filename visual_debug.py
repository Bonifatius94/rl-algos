import gym
from algos.dreamer.config import DreamerSettings
from algos.dreamer.model import DreamerModel
from algos.dreamer.env import DreamerEnvWrapper, play_episode
from algos.dreamer.display import DreamerDebugDisplay
from algos.ppo.agent import PPOAgent
from algos.ppo.config import PPOTrainingSettings


def visual_debugging():
    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [64, 64, 3], [32, 32], [512], [64])
    model = DreamerModel(settings)
    model.load("model/dreamer_final")

    agent_config = PPOTrainingSettings(settings.repr_dims, 6)
    agent = PPOAgent(agent_config)
    agent.load("model/agent_final")

    env = DreamerEnvWrapper(orig_env, settings, debug=True, debug_scaling=8)
    display = DreamerDebugDisplay(settings.obs_dims[1], settings.obs_dims[0], 8)
    env.render_output = display.next_frame
    while True:
        play_episode(env, actor=agent.act, render=True)


if __name__ == "__main__":
    visual_debugging()
