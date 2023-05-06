from typing import Tuple

import numpy as np
import cv2
from PIL import Image

from algos.dreamer.env import DreamerEnvWrapper, play_episode


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
    # TODO: figure out why the recorded files are empty
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
