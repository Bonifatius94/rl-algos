import gym
from algos.dreamer.config import DreamerSettings
from algos.dreamer.env import DreamerEnvWrapper, play_episode
from algos.dreamer.display import DreamerDebugDisplay


def visual_debugging():
    env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [32, 32, 3], [32, 32], [512], [64])
    env = DreamerEnvWrapper(env, settings, debug=True, debug_scaling=8)
    display = DreamerDebugDisplay(settings.obs_dims[1], settings.obs_dims[0], 8)
    env.render_output = display.next_frame
    while True:
        play_episode(env, render=True)


if __name__ == "__main__":
    visual_debugging()
