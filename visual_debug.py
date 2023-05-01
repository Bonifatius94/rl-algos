from typing import Callable, Any
import gym

from algos.dreamer import \
    DreamerSettings, DreamerEnvWrapper, DreamerDebugDisplay


def play_episode(
        env: gym.Env,
        actor: Callable[[Any], Any]=None,
        render: bool=False,
        max_steps: int=float("inf")):
    actor = actor if actor else lambda x: env.action_space.sample()
    render_frame = env.render if render else lambda: None
    state = env.reset()
    render_frame()
    done = False
    i = 0
    while not done:
        action = actor(state)
        state, __, done, ___ = env.step(action)
        render_frame()
        if done or i >= max_steps:
            break
        i += 1


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
