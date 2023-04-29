from typing import List
from dataclasses import dataclass

import gym
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import Model, Input
from keras.layers import \
    Dense, Conv2D, Conv2DTranspose, Dropout, Flatten, \
    Reshape, Softmax, Lambda, Concatenate, GRU
from keras.optimizers import Adam
from keras.losses import MSE, kullback_leibler_divergence


@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    @property
    def steps(self) -> int:
        return self.actions.shape[0]


@dataclass
class DreamerSettings:
    action_dims: List[int]
    obs_dims: List[int]
    repr_dims: List[int]
    hidden_dims: int
    enc_dims: int


class DreamerModel:
    def __init__(self, settings: DreamerSettings):
        self.settings = settings
        self.optimizer = Adam()
        self.env_model, self.dream_model = DreamerModel.create_models(settings)

    @staticmethod
    def create_models(settings: DreamerSettings):

        history_model = DreamerModel.create_recurrent_model(settings)
        trans_model = DreamerModel.create_transition_model(settings)
        repr_model = DreamerModel.create_representation_model(settings)
        repr_out_model = DreamerModel.create_repr_output_model(settings)
        encoder_model = DreamerModel.create_state_encoder_model(settings)
        decoder_model = DreamerModel.create_state_decoder_model(settings)
        reward_model = DreamerModel.create_reward_model(settings)

        s1 = Input(settings.obs_dims)
        a0 = Input(settings.action_dims)
        h0 = Input(settings.hidden_dims)
        z0 = Input(settings.repr_dims)

        # TODO: compose models to higher-level model with fluent api
        s1_enc = encoder_model(s1)
        h1 = history_model(a0, z0, h0)
        z1 = repr_model(s1_enc, h1)
        z1_hat = trans_model(h1)
        # TODO: for performance reasons, leave z1_hat prediction out during simulation
        out = repr_out_model(z1, h1)
        r1_hat = reward_model(out)
        s1_hat = decoder_model(out)

        env_model = Model(inputs=[s1, a0, h0, z0], outputs=[z1_hat, s1_hat, r1_hat, h1, z1])

        h1 = history_model(a0, z0, h0)
        z1_hat = trans_model(h1)
        out = repr_out_model(z1_hat, h1)
        r1_hat = reward_model(out)
        s1_hat = decoder_model(out)

        dream_model = Model(inputs=[a0, h0, z0], outputs=[z1_hat, s1_hat, r1_hat, h1])
        return env_model, dream_model


    @tf.function
    def train(self, traj: Trajectory):
        BETA = 0.1

        # bootstrap first timestep
        a_in = tf.zeros(self.settings.action_dims)
        h_in = tf.zeros(self.settings.hidden_dims)
        z_in = tf.zeros(self.settings.repr_dims)
        _, __, ___, h0, z0 = self.env_model(traj.states[0], a_in, h_in, z_in)

        loss = 0.0
        with tf.GradientTape() as tape:

            # unroll trajectory
            for t in range(traj.steps):
                term = 1.0 if t == traj.steps - 1 else 0.0
                a0, r1, s1 = traj.actions[t], traj.rewards[t], traj.states[t+1]
                z1_hat, s1_hat, (r1_hat, term_hat), h1, z1 = \
                    self.env_model(s1, a0, tf.stop_gradient(z0), tf.stop_gradient(h0))

                reward_loss = MSE(r1, r1_hat)
                obs_loss = MSE(s1, s1_hat)
                term_loss = MSE(term, term_hat)
                repr_loss = BETA * tf.reduce_mean(kullback_leibler_divergence(z1, z1_hat))
                loss += reward_loss + obs_loss + term_loss + repr_loss

                h0, z0 = h1, z1

        grads = tape.gradient(loss, self.env_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.env_model.trainable_variables))

    @staticmethod
    def create_recurrent_model(settings: DreamerSettings) -> Model:
        # (hidden state t, action t, representation t) -> hidden state t+1

        h0_in = Input(settings.hidden_dims)
        a0_in = Input(settings.action_dims)
        z0_in = Input((settings.repr_dims[0], settings.repr_dims[1]))

        concat_in = Concatenate()
        flatten_repr = Flatten()
        expand_timeseries = Lambda(lambda x: tf.expand_dims(x, axis=0))
        dense_in = Dense(settings.hidden_dims, activation="linear")
        rnn = GRU(settings.hidden_dims, return_state=True)

        gru_in = dense_in(concat_in([flatten_repr(z0_in), a0_in]))
        _, h1_out = rnn(expand_timeseries(gru_in), initial_state=h0_in)
        return Model(inputs=[a0_in, z0_in, h0_in], outputs=h1_out)

    @staticmethod
    def create_transition_model(settings: DreamerSettings) -> Model:
        # hidden state t+1 -> representation t+1
        model_in = Input(settings.hidden_dims)
        dense = Dense(settings.repr_dims[0] * settings.repr_dims[1])
        reshape = Reshape((settings.repr_dims[0], settings.repr_dims[1]))
        softmax = Softmax()
        sample = tfp.layers.OneHotCategorical(settings.repr_dims[1])
        stop_grad = Lambda(lambda x: tf.stop_gradient(x))

        logits = reshape(dense(model_in))
        samples = sample(logits)
        probs = softmax(logits)
        model_out = samples + probs - stop_grad(probs) # straight-through gradients
        return Model(inputs=model_in, outputs=model_out)

    @staticmethod
    def create_representation_model(settings: DreamerSettings) -> Model:
        # fuses encoded state and hidden state
        # stochastic model that outputs a distribution
        enc_in = Input(settings.enc_dims)
        h_in = Input(settings.hidden_dims)
        concat = Concatenate()
        dense = Dense(settings.repr_dims[0] * settings.repr_dims[1])
        reshape = Reshape((settings.repr_dims[0], settings.repr_dims[1]))
        softmax = Softmax()
        sample = tfp.layers.OneHotCategorical(settings.repr_dims[1])
        stop_grad = Lambda(lambda x: tf.stop_gradient(x))

        logits = reshape(dense(concat([enc_in, h_in])))
        samples = sample(logits)
        probs = softmax(logits)
        model_out = samples + probs - stop_grad(probs) # straight-through gradients
        return Model(inputs=[enc_in, h_in], outputs=model_out)

    @staticmethod
    def create_repr_output_model(settings: DreamerSettings) -> Model:
        # fuses representation and hidden state
        # input of decoder and reward prediction
        repr_in = Input((settings.repr_dims[0], settings.repr_dims[1]))
        h_in = Input(settings.hidden_dims)
        concat = Concatenate()
        flatten = Flatten()
        model_out = concat([flatten(repr_in), h_in])
        return Model(inputs=[repr_in, h_in], outputs=model_out)

    @staticmethod
    def create_state_encoder_model(settings: DreamerSettings) -> Model:
        model_in = Input(settings.obs_dims)
        prep_cnn_1 = Conv2D(16, (5, 5), strides=(2, 2), padding="same", activation="relu")
        prep_cnn_2 = Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
        prep_cnn_3 = Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
        prep_cnn_4 = Conv2D(8, (3, 3), padding="same", activation="relu")
        prep_drop_1 = Dropout(rate=0.2)
        prep_drop_2 = Dropout(rate=0.2)
        prep_drop_3 = Dropout(rate=0.2)
        prep_drop_4 = Dropout(rate=0.2)
        prep_flatten = Flatten()
        prep_out = Dense(settings.enc_dims, activation="linear")

        prep_model_convs = \
            prep_drop_4(prep_cnn_4(prep_drop_3(prep_cnn_3(
                prep_drop_2(prep_cnn_2(prep_drop_1(prep_cnn_1(model_in))))))))
        model_out = prep_out(prep_flatten(prep_model_convs))

        return Model(inputs=model_in, outputs=model_out)

    @staticmethod
    def create_state_decoder_model(settings: DreamerSettings) -> Model:
        image_channels = settings.obs_dims[-1]
        input_shape = settings.repr_dims[0] * settings.repr_dims[1] + settings.hidden_dims
        model_in = Input((input_shape))

        input_dims = (settings.obs_dims[0] // 8 * settings.obs_dims[1] // 8) * 8
        dense_in = Dense(input_dims, activation="linear")
        reshape_in = Reshape((settings.obs_dims[0] // 8, settings.obs_dims[1] // 8, -1))

        prep_cnn_1 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same", activation="relu")
        prep_cnn_2 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
        prep_cnn_3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
        prep_cnn_4 = Conv2D(image_channels, (5, 5), padding="same", activation="relu")

        prep_drop_1 = Dropout(rate=0.2)
        prep_drop_2 = Dropout(rate=0.2)
        prep_drop_3 = Dropout(rate=0.2)
        prep_drop_4 = Dropout(rate=0.2)

        prep_in = reshape_in(dense_in(model_in))
        model_out = \
            prep_drop_4(prep_cnn_4(prep_drop_3(prep_cnn_3(
                prep_drop_2(prep_cnn_2(prep_drop_1(prep_cnn_1(prep_in))))))))

        return Model(inputs=model_in, outputs=model_out)

    @staticmethod
    def create_reward_model(settings: DreamerSettings) -> Model:
        # concat(representation, hidden state) -> (reward, done)
        input_shape = settings.repr_dims[0] * settings.repr_dims[1] + settings.hidden_dims
        model_in = Input((input_shape))

        dense_1 = Dense(64, "relu")
        dense_2 = Dense(64, "relu")
        reward = Dense(1, activation="linear")
        terminal = Dense(2)
        softmax = Softmax()
        sample = tfp.layers.OneHotCategorical(2)
        argmax = Lambda(lambda x: tf.cast(tf.argmax(tf.cast(x, dtype=tf.float32), axis=1), dtype=tf.float32))
        stop_grad = Lambda(lambda x: tf.stop_gradient(x))

        prep = dense_2(dense_1(model_in))
        reward_out = reward(prep)
        logits = terminal(prep)
        samples = argmax(sample(logits))
        probs = softmax(logits)
        terminal_out = samples + probs - stop_grad(probs) # straight-through gradients
        return Model(inputs=model_in, outputs=[reward_out, terminal_out])


@dataclass
class DreamerEnvWrapper(gym.Env):
    orig_env: gym.Env

    @property
    def observation_space(self):
        return self.orig_env.observation_space

    @property
    def action_space(self):
        return self.orig_env.action_space

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode):
        pass

    def seed(self, seed):
        pass


if __name__ == "__main__":
    settings = DreamerSettings([10], [32, 32, 3], [32, 32], 512, 64)
    model = DreamerModel(settings)
