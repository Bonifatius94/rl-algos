import os
from typing import Callable, Tuple, List, Iterator

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from keras.optimizers import Optimizer, Adam

from algos.ppo.config import PPOTrainingSettings


Observation = np.ndarray
Action = int
SarsExperience = Tuple[Observation, Action, float, Observation, bool]

BatchedObservations = np.ndarray
BatchedActions = np.ndarray
BatchedProbDists = np.ndarray
BatchedBaselines = np.ndarray
BatchedPredictions = Tuple[BatchedProbDists, BatchedBaselines]
BatchedAdvantages = np.ndarray
BatchedRewards = np.ndarray
BatchedReturns = np.ndarray
BatchedDones = np.ndarray
BatchedSarsExps = Tuple[BatchedObservations, BatchedActions,
    BatchedRewards, BatchedObservations, BatchedDones]
TrainingBatch = Tuple[BatchedObservations, BatchedActions, BatchedReturns,
    BatchedAdvantages, BatchedProbDists, BatchedBaselines]


class PPOModel:
    def __init__(
            self, config: PPOTrainingSettings,
            log_losses: Callable[[float, float, float], None],
            on_training_done: Callable[[], None]):
        self.config = config
        self.log_losses = log_losses
        self.on_training_done = on_training_done
        self.optimizer: Optimizer = Adam(learning_rate=self.config.learn_rate, epsilon=1e-5)
        self.model = PPOModel.create_model(config.obs_shape, config.num_actions)

    @staticmethod
    def create_model(obs_shape: List[int], num_actions: int) -> Model:
        model_in = Input(obs_shape)

        use_feature_extractor = False
        if use_feature_extractor:
            prep_cnn_1 = Conv2D(16, (5, 5), strides=(2, 2), activation="relu")
            prep_drop_1 = Dropout(rate=0.2)
            prep_cnn_2 = Conv2D(16, (3, 3), strides=(2, 2), activation="relu")
            prep_drop_2 = Dropout(rate=0.2)
            prep_cnn_3 = Conv2D(16, (3, 3), strides=(2, 2), activation="relu")
            prep_drop_3 = Dropout(rate=0.2)
            prep_cnn_4 = Conv2D(8, (3, 3), strides=(2, 2), activation="relu")
            prep_drop_4 = Dropout(rate=0.2)
            prep_flatten = Flatten()
            prep_out = Dense(256, activation="linear")

            prep_model_convs = \
                prep_drop_4(prep_cnn_4(prep_drop_3(prep_cnn_3(
                    prep_drop_2(prep_cnn_2(prep_drop_1(prep_cnn_1(model_in))))))))
            prep_model = prep_out(prep_flatten(prep_model_convs))
        else:
            flatten = Flatten()
            prep_model = flatten(model_in)

        actor_fc_1 = Dense(64, activation="relu")
        actor_fc_2 = Dense(64, activation="relu")
        critic_fc_1 = Dense(64, activation="relu")
        critic_fc_2 = Dense(64, activation="relu")
        actor_out = Dense(num_actions, activation="softmax")
        critic_out = Dense(1, activation="linear")

        actor = actor_out(actor_fc_2(actor_fc_1(prep_model)))
        critic = critic_out(critic_fc_2(critic_fc_1(prep_model)))
        return Model(inputs=model_in, outputs=[actor, critic])

    def predict(self, states: BatchedObservations) -> BatchedPredictions:
        prob_dists, baselines = self.model(states)
        prob_dists, baselines = prob_dists.numpy(), baselines.numpy()
        return prob_dists, np.squeeze(baselines)

    def train(self, shuffled_batch_iter: Iterator[TrainingBatch]):
        for _ in range(self.config.update_epochs):
            for batch in shuffled_batch_iter:
                self.train_step(batch)
        self.on_training_done()

    @tf.function
    def train_step(self, batch: TrainingBatch):
        states, actions, returns, advantages, prob_dists_old, baselines_old = batch
        clip_lower, clip_upper = 1.0 - self.config.prob_clip, 1.0 + self.config.prob_clip
        clip_value = self.config.value_clip
        vf_coef, ent_coef = self.config.vf_coef, self.config.ent_coef

        if self.config.norm_advantages:
            std_dev = tf.math.reduce_std(advantages) + 1e-8
            advantages = (advantages - tf.reduce_mean(advantages)) / std_dev

        with tf.GradientTape() as tape:
            prob_dists_new, baselines = self.model(states)
            baselines = tf.squeeze(baselines)

            # info: pick probabilities of selected actions (ignore other probs)
            actions_onehot = tf.one_hot(actions, depth=self.config.num_actions)
            probs_new = tf.reduce_sum(tf.multiply(prob_dists_new, actions_onehot), axis=-1)
            probs_old = tf.reduce_sum(tf.multiply(prob_dists_old, actions_onehot), axis=-1)

            # info: p_new / p_old = exp(log(p_new) - log(p_old))
            policy_ratio = (tf.math.exp(tf.math.log(probs_new + 1e-8) - tf.math.log(probs_old + 1e-8)))
            policy_ratio_clipped = tf.clip_by_value(policy_ratio, clip_lower, clip_upper)
            policy_loss = -1.0 * tf.reduce_mean(tf.minimum(
                policy_ratio * advantages, policy_ratio_clipped * advantages))

            value_loss = tf.reduce_mean(tf.pow(returns - baselines, 2))
            if self.config.clip_values:
                value_clipped = baselines_old + tf.clip_by_value(
                    baselines - baselines_old, -clip_value, clip_value)
                value_loss_clipped = tf.reduce_mean(tf.pow(returns - value_clipped, 2))
                value_loss = tf.reduce_mean(tf.maximum(value_loss, value_loss_clipped))

            entropy = tf.reduce_sum(-1.0 * prob_dists_new * tf.math.log(prob_dists_new + 1e-8), axis=1)
            entropy_loss = tf.reduce_mean(entropy)

            loss = policy_loss + value_loss * vf_coef + entropy_loss * ent_coef

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.log_losses(policy_loss, value_loss, entropy_loss)

    @staticmethod
    def shuffle_batch(batch: TrainingBatch) -> TrainingBatch:
        states, actions, returns, advantages, prob_dists_old, baselines_old = batch
        ids = np.arange(states.shape[0])
        np.random.shuffle(ids)
        return states[ids], actions[ids], returns[ids], \
            advantages[ids], prob_dists_old[ids], baselines_old[ids]

    def save(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, "ppo_model.h5")
        self.model.save_weights(file_path)

    def load(self, directory: str):
        if not os.path.exists(directory):
            raise FileNotFoundError()
        file_path = os.path.join(directory, "ppo_model.h5")
        self.model.load_weights(file_path)
