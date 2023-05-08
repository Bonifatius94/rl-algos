from typing import Iterable

import gym
import numpy as np

from algos.dreamer.config import DreamerSettings
from algos.dreamer.env import DreamEnv, DreamVecEnv, DreamerEnvWrapper, play_episode


def shape_of(array: np.ndarray, exp_shape: Iterable[int]) -> bool:
    return all([array.shape[i] == exp_shape[i] for i in range(len(array.shape))])


def test_can_play_episode_with_wrapper_env():
    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [64, 64, 3], [32, 32], [512], [128])

    wrapper_env = DreamerEnvWrapper(orig_env, settings)
    play_episode(wrapper_env)


def test_can_collect_sars_experiences_with_wrapper_env():
    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [64, 64, 3], [32, 32], [512], [128])

    def assert_collect_step_has_correct_shapes(s1, a, r):
        assert shape_of(s1, settings.obs_dims)
        assert shape_of(np.array(a), [1])
        assert shape_of(np.array(r), [1])

    wrapper_env = DreamerEnvWrapper(orig_env, settings)
    wrapper_env.collect_step = assert_collect_step_has_correct_shapes
    play_episode(wrapper_env)


def test_can_render_frames_with_wrapper_env():
    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [64, 64, 3], [32, 32], [512], [128])

    def assert_render_frames_has_correct_shapes(f1, f2):
        assert shape_of(f1, settings.obs_dims)
        assert shape_of(f2, settings.obs_dims)

    wrapper_env = DreamerEnvWrapper(orig_env, settings)
    wrapper_env.render_output = assert_render_frames_has_correct_shapes
    play_episode(wrapper_env, render=True)


def test_can_play_episode_with_simple_dream_env():
    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [64, 64, 3], [32, 32], [512], [128])
    wrapper_env = DreamerEnvWrapper(orig_env, settings)

    init_state_provider = lambda: (
        np.zeros([5] + settings.obs_dims),
        np.zeros([4] + settings.action_dims))
    dream_env = DreamEnv(
        wrapper_env.model, wrapper_env.observation_space,
        wrapper_env.action_space, init_state_provider)
    play_episode(dream_env)


def test_can_step_with_vectorized_dream_env():
    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [64, 64, 3], [32, 32], [512], [128])
    wrapper_env = DreamerEnvWrapper(orig_env, settings)

    n_envs = 64
    init_state_provider = lambda: (
        np.zeros([5] + settings.obs_dims),
        np.zeros([4] + settings.action_dims))
    vec_dream_env = DreamVecEnv(
        n_envs, wrapper_env.observation_space, wrapper_env.action_space,
        wrapper_env.model, init_state_provider)

    def play_episode_vecenv(env: gym.vector.VectorEnv):
        env.reset()
        dones = 0
        while not np.any(dones):
            actions = np.array(env.action_space.sample())
            obs, rewards, dones, _ = env.step(actions)

    play_episode_vecenv(vec_dream_env)
