import gym
from algos.dreamer import \
    DreamerSettings, DreamerModel, \
    DreamerDebugDisplay, DreamerEnvWrapper


def visual_debugging():
    env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [256, 256, 3], [32, 32], [512], [64])
    model = DreamerModel(settings)
    display_factory = lambda: DreamerDebugDisplay(settings.obs_dims[0], settings.obs_dims[1])
    dream_env = DreamerEnvWrapper(env, model, debug=True, display_factory=display_factory)

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
