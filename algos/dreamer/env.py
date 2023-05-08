from typing import Callable, Any
from dataclasses import dataclass, field

import gym
import numpy as np
from PIL import Image

from algos.dreamer.config import DreamerSettings
from algos.dreamer.model import DreamerModel


RenderSubscriber = Callable[[np.ndarray, np.ndarray], None]
TrajectorySubscriber = Callable[[np.ndarray, np.ndarray, np.ndarray], None]


def play_episode(
        env: gym.Env,
        actor: Callable[[Any], Any]=None,
        render: bool=False,
        max_steps: int=float("inf")):
    actor = actor if actor else lambda x: env.action_space.sample()
    render_frame = env.render if render else lambda: None
    state = env.reset()
    render_frame()
    done, i = False, 0
    while True:
        action = actor(state)
        state, __, done, ___ = env.step(action)
        render_frame()
        if done or i >= max_steps:
            break
        i += 1


@dataclass
class DreamEnv(gym.Env):
    model: DreamerModel
    observation_space: gym.Space
    action_space: gym.Space
    fetch_initial_state: Callable[[], Any] = None
    h0: np.ndarray = field(init=False)
    z0: np.ndarray = field(init=False)

    def reset(self):
        s0_init, a0_init = self.fetch_initial_state()
        h0, z0 = self.model.dream_bootstrap(
            self._batch(s0_init), self._batch(a0_init))
        self.h0, self.z0 = h0.numpy(), z0.numpy()
        return self._unbatch(self.z0)

    def step(self, action):
        (r1, term), h1, z1 = self.model.dream_step(
            self._batch(action), self.h0, self.z0)
        self.h0, self.z0 = h1, z1
        term = np.round(term)
        return self._unbatch(z1), self._unbatch(r1), self._unbatch(term), None

    def seed(self, seed: int):
        self.model.seed(seed)

    def _batch(self, arr: np.ndarray) -> np.ndarray:
        return np.expand_dims(arr, axis=0)

    def _unbatch(self, arr: np.ndarray) -> np.ndarray:
        return np.squeeze(arr, axis=0)


class DreamVecEnv(gym.vector.VectorEnv):
    def __init__(
            self, num_envs: int, obs_space: gym.Space, action_space: gym.Space,
            model: DreamerModel, fetch_initial_state: Callable[[], Any]):
        super(DreamVecEnv, self).__init__(num_envs, obs_space, action_space)
        self.model = model
        self.fetch_initial_state = fetch_initial_state
        self.h0: np.ndarray = None
        self.z0: np.ndarray = None
        self.closed = True

    def reset(self):
        h0, z0 = self._bootstrap(self.num_envs)
        self.h0, self.z0 = h0.numpy(), z0.numpy()
        return self.z0

    def step(self, actions):
        (r1, term), h1, z1 = self.model.dream_step(actions, self.h0, self.z0)
        self.h0, self.z0 = h1.numpy(), z1.numpy()
        r1, term = np.squeeze(r1), np.squeeze(term.numpy())
        term = np.round(term)

        done_envs = np.where(term)[0]
        num_dones = done_envs.shape[0]
        if num_dones > 0:
            h_temp, z_temp = self._bootstrap(num_dones)
            self.h0[done_envs] = h_temp
            self.z0[done_envs] = z_temp

        return self.z0, r1, term, None

    def seed(self, seed: int):
        self.model.seed(seed)

    def _bootstrap(self, batch_size: int):
        init_data = [self.fetch_initial_state() for _ in range(batch_size)]
        s0_init = np.array([s for (s, a) in init_data])
        a0_init = np.array([a for (s, a) in init_data])
        return self.model.dream_bootstrap(s0_init, a0_init)


@dataclass
class DreamerEnvWrapper(gym.Env):
    orig_env: gym.Env
    settings: DreamerSettings
    model: DreamerModel = field(default=None)
    render_output: RenderSubscriber = field(default=lambda frame_orig, frame_hall: None)
    collect_step: TrajectorySubscriber = field(default=lambda s1, a0, r1: None)
    # TODO: inject encode_obs as callable to support obs other than images

    action_space: gym.Space = field(init=False)
    observation_space: gym.Space = field(init=False)
    h0: np.ndarray = field(init=False)
    z0: np.ndarray = field(init=False)
    frame_orig: np.ndarray = field(init=False)

    def __post_init__(self):
        if not self.model:
            self.model = DreamerModel(self.settings)

        low = np.zeros(self.settings.repr_dims, dtype=np.float32)
        high = np.ones(self.settings.repr_dims, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = self.orig_env.action_space

    def reset(self):
        state = self.orig_env.reset()
        state = self._resize_image(state)
        self.h0, self.z0 = self.model.bootstrap(self._batch(state))
        self.frame_orig, repr = state, self._unbatch(self.z0)
        self.collect_step(state, None, None)
        return repr

    def step(self, action):
        state, reward, done, meta = self.orig_env.step(action)
        state = self._resize_image(state)
        self.h0, self.z0 = self.model.step(
            self._batch(state), self._batch(action), self.h0, self.z0)
        self.frame_orig, repr = state, self._unbatch(self.z0)
        self.collect_step(state, action, reward)
        return repr, reward, done, meta

    def render(self, mode="human"):
        frame_hall = self.model.render(self.h0, self.z0)
        self.render_output(self.frame_orig, self._unbatch(frame_hall))

    def seed(self, seed: int):
        self.orig_env.seed(seed)
        self.model.seed(seed)

    def _batch(self, arr: np.ndarray) -> np.ndarray:
        return np.expand_dims(arr, axis=0)

    def _unbatch(self, arr: np.ndarray) -> np.ndarray:
        return np.squeeze(arr, axis=0)

    def _resize_image(self, orig_image: np.ndarray) -> np.ndarray:
        width, height, _ = self.settings.obs_dims
        img = Image.fromarray(orig_image)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(img)
