from typing import Callable, Any
import gym
import numpy as np
from algos.dreamer import \
    DreamerEnvWrapper, Trajectory, DreamerModel, \
    DreamerSettings, DreamerDebugDisplay


def sample_trajectory(env: gym.Env, actor: Callable[[], Any]) -> Trajectory:
    done = False
    (state, z) = env.reset()

    traj_states = [state]
    traj_zs = [z]
    traj_actions = []
    traj_rewards = []

    i = 0
    while not done and i < 100:
        action = actor(state)
        (state, z), reward, done, _ = env.step(action)
        traj_states.append(state)
        traj_zs.append(z)
        traj_actions.append(action)
        traj_rewards.append(reward)
        i += 1

    return Trajectory(
        np.array(traj_states), np.array(traj_zs),
        np.array(traj_actions), np.array(traj_rewards))


def train(num_epochs: int, num_trajectories: int, debug_interval: int):
    env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [32, 32, 3], [32, 32], [512], [64])
    model = DreamerModel(settings)
    dream_env = DreamerEnvWrapper(env, model)
    display_factory = lambda: DreamerDebugDisplay(settings.obs_dims[0], settings.obs_dims[1], 8)
    debug_env = DreamerEnvWrapper(env, model, debug=True, display_factory=display_factory)

    def visual_debug():
        debug_env.reset()
        debug_env.render()
        done = False
        while not done:
            action = debug_env.action_space.sample()
            _, __, done, ___ = debug_env.step(action)
            debug_env.render()
            if done:
                debug_env.reset()
                debug_env.render()

    trajs = [sample_trajectory(dream_env, lambda x: dream_env.action_space.sample())
            for _ in range(4)]

    for ep in range(num_epochs):
        print(f"starting epoch {ep+1}")

        for traj in trajs:
            model.train(traj.states, traj.actions, traj.rewards)

        if (ep+1) % debug_interval == 0:
            visual_debug()


if __name__ == "__main__":
    train(1000000, 1, 100)
