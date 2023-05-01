from typing import Callable
from dataclasses import dataclass, field

import gym
import numpy as np
from PIL import Image

from algos.dreamer.config import DreamerSettings
from algos.dreamer.model import DreamerModel


RenderSubscriber = Callable[[np.ndarray, np.ndarray], None]
TrajectorySubscriber = Callable[[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray], None]


@dataclass
class DreamerEnvWrapper(gym.Env):
    orig_env: gym.Env
    settings: DreamerSettings
    model: DreamerModel = field(default=None)
    render_output: RenderSubscriber = field(default=lambda frame_orig, frame_hall: None)
    collect_step: TrajectorySubscriber = field(default=lambda s1, z1, h1, a, r: None)
    debug: bool = False
    debug_scaling: float = 1.0

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
        self.collect_step(state, repr, self.h0, None, None)
        return repr

    def step(self, action):
        state, reward, done, meta = self.orig_env.step(action)
        state = self._resize_image(state)
        inputs = (self._batch(state), self._batch(action), self.h0, self.z0)
        self.h0, self.z0 = self.model.step_model(inputs)
        self.frame_orig, repr = state, self._unbatch(self.z0)
        self.collect_step(state, repr, self.h0, action, reward)
        return repr, reward, done, meta

    def render(self, mode="human"):
        frame_hall = self.model.render_model((self.z0, self.h0))
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
