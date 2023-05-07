from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from keras.metrics import Metric, Mean


@dataclass
class PPOTensorboardLogging:
    step: int = 0
    policy_loss: Metric = Mean()
    value_loss: Metric = Mean()
    entropy_loss: Metric = Mean()

    def __post_init__(self):
        train_log_dir = 'logs/ppo'
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

    def log_training_loss(self, policy_loss: float, value_loss: float, entropy_loss: float):
        self.policy_loss(policy_loss)
        self.value_loss(value_loss)
        self.entropy_loss(entropy_loss)

    def flush_losses(self):
        with self.summary_writer.as_default():
            tf.summary.scalar('losses/policy_loss', self.policy_loss.result(), step=self.step)
            tf.summary.scalar('losses/value_loss', self.value_loss.result(), step=self.step)
            tf.summary.scalar('losses/entropy_loss', self.entropy_loss.result(), step=self.step)
        self.policy_loss.reset_states()
        self.value_loss.reset_states()
        self.entropy_loss.reset_states()

    def log_episode(self, train_step: int, avg_rewards: float, avg_steps: float):
        self.step = train_step
        with self.summary_writer.as_default():
            tf.summary.scalar('env/ep_rewards', avg_rewards, step=train_step)
            tf.summary.scalar('env/ep_steps', avg_steps, step=train_step)
