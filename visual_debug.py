import gym
from algos.dreamer import DreamerSettings, DreamerEnvWrapper


def visual_debugging():
    env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [32, 32, 3], [32, 32], [512], [64])
    dream_env = DreamerEnvWrapper(env, settings, debug=True, debug_scaling=8)

    dream_env.reset()
    dream_env.render()

    for _ in range(10000):
        action = dream_env.action_space.sample()
        _, __, done, ___ = dream_env.step(action)
        dream_env.render()
        if done:
            dream_env.reset()
            dream_env.render()


if __name__ == "__main__":
    visual_debugging()
