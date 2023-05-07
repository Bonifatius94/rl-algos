from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.metrics import Mean


@dataclass
class DreamerTensorboardLogger:
    obs_loss_metric: Mean = Mean()
    repr_loss_metric: Mean = Mean()
    reward_loss_metric: Mean = Mean()
    term_loss_metric: Mean = Mean()
    summary_writer: tf.summary.SummaryWriter = field(init=False)

    def __post_init__(self):
        train_log_dir = "logs/dreamer"
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

    def log_loss(self, obs_loss, repr_loss, reward_loss, term_loss):
        self.obs_loss_metric(obs_loss)
        self.repr_loss_metric(repr_loss)
        self.reward_loss_metric(reward_loss)
        self.term_loss_metric(term_loss)

    def log_frames(self, step: int, frame_orig: np.ndarray, frame_hall: np.ndarray):
        frame_hall = np.clip(frame_hall, 0, 255).astype(dtype=np.uint8)
        with self.summary_writer.as_default():
            tf.summary.image(f"image/snapshot_{step}", np.array([frame_orig, frame_hall]), step)

    def flush_losses(self, step: int):
        with self.summary_writer.as_default():
            tf.summary.scalar("losses/obs_loss", self.obs_loss_metric.result(), step=step)
            tf.summary.scalar("losses/repr_loss", self.repr_loss_metric.result(), step=step)
            tf.summary.scalar("losses/reward_loss", self.reward_loss_metric.result(), step=step)
            tf.summary.scalar("losses/term_loss", self.term_loss_metric.result(), step=step)
        self.obs_loss_metric.reset_states()
        self.repr_loss_metric.reset_states()
        self.reward_loss_metric.reset_states()
        self.term_loss_metric.reset_states()
