import os
from typing import Callable, Any, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import cv2
import gym

from algos.dreamer import DreamerEnvWrapper, DreamerSettings


@dataclass
class Trajectory:
    states: np.ndarray
    zs: np.ndarray
    hs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    def __post_init__(self):
        if self.actions.shape[0] != self.rewards.shape[0]:
            raise ValueError(
                "Invalid trajectory! The amount of actions and rewards needs to be the same!")
        if self.actions.shape[0] + 1 != self.states.shape[0] \
                or self.states.shape[0] != self.zs.shape[0] \
                or self.states.shape[0] != self.hs.shape[0]:
            raise ValueError(
                f"Invalid trajectory! Expected {self.rewards.shape[0] + 1} states!")

    @property
    def timesteps(self) -> int:
        return self.actions.shape[0]


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


def sample_trajectory(
        env: DreamerEnvWrapper,
        actor: Callable[[Any], Any]) -> Trajectory:
    traj_states = []
    traj_zs = []
    traj_hs = []
    traj_actions = []
    traj_rewards = []

    def append_timestep(s1, z1, h1, a, r):
        traj_states.append(s1)
        traj_zs.append(z1)
        traj_hs.append(h1)
        if a is not None:
            traj_actions.append(a)
            traj_rewards.append(r)

    env.collect_step = append_timestep
    play_episode(env, actor, max_steps=100)
    env.collect_step = lambda s1, z1, h1, a, r: None

    return Trajectory(
        np.array(traj_states), np.array(traj_zs), np.array(traj_hs),
        np.array(traj_actions), np.array(traj_rewards))


class Mp4VideoWriter:
    def __init__(self, output_file: str, resolution: Tuple[int, int]):
        codec, fps = cv2.VideoWriter_fourcc('M','J','P','G'), 25
        frame_shape = (resolution[1], resolution[0])
        self.video = cv2.VideoWriter(output_file, codec, fps, frame_shape, False)

    def next_frame(self, frame: np.ndarray):
        frame = frame[:, :, ::-1]
        self.video.write(frame)

    def finalize(self):
        self.video.release()


def record_episode(env: DreamerEnvWrapper, video_file: str):
    resolution = env.settings.obs_dims[:2]
    resolution = (resolution[1] * 2, resolution[0])
    video_writer = Mp4VideoWriter(video_file, resolution)

    def resize_image(orig_image: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
        img = Image.fromarray(orig_image)
        img = img.resize(resolution, Image.Resampling.LANCZOS)
        return np.array(img)

    def prepare_frame(frame: np.ndarray, clip: bool) -> np.ndarray:
        if clip:
            frame = np.clip(frame, 0, 255)
        frame = np.rot90(np.fliplr(frame)).astype(dtype=np.uint8)
        frame = resize_image(frame, (800, 600))
        return frame

    def concat_frames_horizontal(
            orig_frame: np.ndarray, hall_frame: np.ndarray) -> np.ndarray:
        # TODO: do something with the hallucinated frame
        return orig_frame

    def next_frame(orig_frame: np.ndarray, hall_frame: np.ndarray):
        orig_frame = prepare_frame(orig_frame, clip=False)
        hall_frame = prepare_frame(hall_frame, clip=True)
        union_frame = concat_frames_horizontal(orig_frame, hall_frame)
        video_writer.next_frame(union_frame)

    env.render_output = next_frame
    play_episode(env, render=True)
    env.render_output = lambda x, y: None

    video_writer.finalize()


def train(num_epochs: int, num_trajectories: int, debug_interval: int):
    VIDEOS_ROOTDIR = "./videos"
    if os.path.exists(VIDEOS_ROOTDIR):
        os.system(f"rm -rf {VIDEOS_ROOTDIR}")
    os.mkdir(VIDEOS_ROOTDIR)

    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [32, 32, 3], [32, 32], [512], [64])
    env = DreamerEnvWrapper(orig_env, settings)

    actor = lambda x: env.action_space.sample()
    trajs = [sample_trajectory(env, actor) for _ in range(num_trajectories)]

    for ep in range(num_epochs):
        print(f"starting epoch {ep+1}")

        for traj in trajs:
            env.model.train(traj.states, traj.actions, traj.rewards)

        if (ep+1) % debug_interval == 0:
            video_file = os.path.join(VIDEOS_ROOTDIR, f"ep_{ep+1}.avi")
            # TODO: figure out why the recorded files are empty
            record_episode(env, video_file)


if __name__ == "__main__":
    train(1000000, 4, 1)
