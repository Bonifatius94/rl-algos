from typing import List, Union
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
from keras.losses import MSE, kullback_leibler_divergence, sparse_categorical_crossentropy


@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    def __post_init__(self):
        if self.actions.shape[0] != self.rewards.shape[0]:
            raise ValueError(("Invalid trajectory! "
                              "The amount of actions and rewards needs to be the same!"))
        if self.actions.shape[0] != self.states.shape[0] + 1:
            raise ValueError(("Invalid trajectory! "
                              "The amount of states needs to be one more than actions / rewards!"))

    @property
    def timesteps(self) -> int:
        return self.actions.shape[0]


@dataclass
class DreamerSettings:
    action_dims: List[int]
    obs_dims: List[int]
    repr_dims: List[int]
    hidden_dims: List[int]
    enc_dims: List[int]
    dropout_rate: float = 0.2

    @property
    def repr_dims_flat(self) -> int:
        return self.repr_dims[0] * self.repr_dims[1]

    @property
    def repr_out_dims_flat(self) -> int:
        return self.repr_dims[0] * self.repr_dims[1] + self.hidden_dims[0]


class STGradsOneHotCategorical:
    def __init__(self, dims: Union[int, List[int]]):
        self.softmax = Softmax()
        self.sample = tfp.layers.OneHotCategorical(dims)
        self.stop_grad = Lambda(lambda x: tf.stop_gradient(x))

    def __call__(self, logits):
        samples = self.sample(logits)
        probs = self.softmax(logits)
        return samples + probs - self.stop_grad(probs)


class ArgmaxLayer:
    def __init__(self):
        self.argmax = Lambda(lambda x: tf.cast(tf.argmax(
            tf.cast(x, dtype=tf.float32), axis=-1), dtype=tf.float32))

    def __call__(self, onehot_categoricals):
        return self.argmax(onehot_categoricals)


class DreamerModel:
    def __init__(self, settings: DreamerSettings):
        self.settings = settings
        self.optimizer = Adam()
        self.env_model, self.dream_model = DreamerModel.create_models(settings)
        self.env_model.summary()
        self.dream_model.summary()

    @tf.function
    def train(self, traj: Trajectory):
        BETA = 0.1

        # bootstrap first timestep
        a_in = tf.zeros(self.settings.action_dims)
        h_in = tf.zeros(self.settings.hidden_dims)
        z_in = tf.zeros(self.settings.repr_dims)
        _, __, ___, h0, z0 = self.env_model((traj.states[0], a_in, h_in, z_in))

        loss = 0.0
        with tf.GradientTape() as tape:

            # TODO: define model as recurrent cell to unroll
            #       multiple steps at once with standard api

            # unroll trajectory
            for t in range(traj.timesteps):
                term = 1.0 if t == traj.timesteps - 1 else 0.0
                a0, r1, s1 = traj.actions[t], traj.rewards[t], traj.states[t+1]
                z1_hat, s1_hat, (r1_hat, term_hat), h1, z1 = \
                    self.env_model((s1, a0, tf.stop_gradient(z0), tf.stop_gradient(h0)))

                reward_loss = MSE(r1, r1_hat)
                obs_loss = MSE(s1, s1_hat)
                term_loss = sparse_categorical_crossentropy(term, term_hat)
                repr_loss = BETA * tf.reduce_mean(kullback_leibler_divergence(z1, z1_hat))
                loss += reward_loss + obs_loss + term_loss + repr_loss

                h0, z0 = h1, z1

        grads = tape.gradient(loss, self.env_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.env_model.trainable_variables))

    @staticmethod
    def create_models(settings: DreamerSettings):
        history_model = DreamerModel.create_history_model(settings)
        trans_model = DreamerModel.create_transition_model(settings)
        repr_model = DreamerModel.create_representation_model(settings)
        repr_out_model = DreamerModel.create_repr_output_model(settings)
        encoder_model = DreamerModel.create_state_encoder_model(settings)
        decoder_model = DreamerModel.create_state_decoder_model(settings)
        reward_model = DreamerModel.create_reward_model(settings)

        s1 = Input(settings.obs_dims, name="s1")
        a0 = Input(settings.action_dims, name="a0")
        h0 = Input(settings.hidden_dims, name="h0")
        z0 = Input(settings.repr_dims, name="z0")

        s1_enc = encoder_model(s1)
        h1 = history_model((a0, z0, h0))
        z1 = repr_model((s1_enc, h1))
        z1_hat = trans_model(h1)
        # TODO: for performance reasons, leave z1_hat prediction out during simulation
        out = repr_out_model((z1, h1))
        r1_hat = reward_model(out)
        s1_hat = decoder_model(out)
        env_model = Model(
            inputs=[s1, a0, h0, z0],
            outputs=[z1_hat, s1_hat, r1_hat, h1, z1],
            name="env_model")

        h1 = history_model((a0, z0, h0))
        z1_hat = trans_model(h1)
        out = repr_out_model((z1_hat, h1))
        r1_hat = reward_model(out)
        s1_hat = decoder_model(out)
        dream_model = Model(
            inputs=[a0, h0, z0],
            outputs=[z1_hat, s1_hat, r1_hat, h1],
            name="dream_model")
        return env_model, dream_model

    @staticmethod
    def create_history_model(settings: DreamerSettings) -> Model:
        # (hidden state t, action t, representation t) -> hidden state t+1

        h0_in = Input(settings.hidden_dims, name="h0")
        a0_in = Input(settings.action_dims, name="a0")
        z0_in = Input(settings.repr_dims, name="z0")

        concat_in = Concatenate()
        flatten_repr = Flatten()
        expand_timeseries = Lambda(lambda x: tf.expand_dims(x, axis=1))
        dense_in = Dense(settings.hidden_dims[0], activation="linear")
        rnn = GRU(settings.hidden_dims[0], return_state=True)
        # TODO: think about stacking multiple GRU cells

        gru_in = dense_in(concat_in([flatten_repr(z0_in), a0_in]))
        _, h1_out = rnn(expand_timeseries(gru_in), initial_state=h0_in)
        return Model(inputs=[a0_in, z0_in, h0_in], outputs=h1_out, name="history_model")

    @staticmethod
    def create_transition_model(settings: DreamerSettings) -> Model:
        # hidden state t+1 -> representation t+1
        model_in = Input(settings.hidden_dims, name="h0")
        dense = Dense(settings.repr_dims_flat)
        reshape = Reshape(settings.repr_dims)
        st_categorical = STGradsOneHotCategorical(settings.repr_dims[1])
        model_out = st_categorical(reshape(dense(model_in)))
        return Model(inputs=model_in, outputs=model_out, name="transition_model")

    @staticmethod
    def create_representation_model(settings: DreamerSettings) -> Model:
        # fuses encoded state and hidden state
        # stochastic model that outputs a distribution
        enc_in = Input(settings.enc_dims, name="enc_obs")
        h_in = Input(settings.hidden_dims, name="h1")
        concat = Concatenate()
        dense = Dense(settings.repr_dims_flat)
        reshape = Reshape(settings.repr_dims)
        st_categorical = STGradsOneHotCategorical(settings.repr_dims)
        model_out = st_categorical(reshape(dense(concat([enc_in, h_in]))))
        return Model(inputs=[enc_in, h_in], outputs=model_out, name="repr_model")

    @staticmethod
    def create_repr_output_model(settings: DreamerSettings) -> Model:
        # fuses representation and hidden state
        # input of decoder and reward prediction
        repr_in = Input(settings.repr_dims, name="z1")
        h_in = Input(settings.hidden_dims, name="h1")
        concat = Concatenate()
        flatten = Flatten()
        model_out = concat([flatten(repr_in), h_in])
        return Model(inputs=[repr_in, h_in], outputs=model_out, name="repr_output_model")

    @staticmethod
    def create_state_encoder_model(settings: DreamerSettings) -> Model:
        # observation t -> encoded state t
        model_in = Input(settings.obs_dims, name="enc_out")
        cnn_1 = Conv2D(16, (5, 5), strides=(2, 2), padding="same", activation="relu")
        cnn_2 = Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
        cnn_3 = Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
        cnn_4 = Conv2D(8, (3, 3), padding="same", activation="relu")
        drop_1 = Dropout(rate=settings.dropout_rate)
        drop_2 = Dropout(rate=settings.dropout_rate)
        drop_3 = Dropout(rate=settings.dropout_rate)
        drop_4 = Dropout(rate=settings.dropout_rate)
        flatten = Flatten()
        dense_out = Dense(settings.enc_dims[0], activation="linear")

        prep_model_convs = drop_4(cnn_4(drop_3(cnn_3(drop_2(cnn_2(drop_1(cnn_1(model_in))))))))
        model_out = dense_out(flatten(prep_model_convs))
        return Model(inputs=model_in, outputs=model_out, name="encoder_model")

    @staticmethod
    def create_state_decoder_model(settings: DreamerSettings) -> Model:
        # concat(representation t+1, hidden state t+1) -> (obs t+1)
        image_channels = settings.obs_dims[-1]
        input_shape = settings.repr_out_dims_flat
        upscale_source_dims = (settings.obs_dims[0] // 8 * settings.obs_dims[1] // 8) * 8

        model_in = Input(settings.repr_out_dims_flat, name="repr_out")
        dense_in = Dense(upscale_source_dims, activation="linear")
        reshape_in = Reshape((settings.obs_dims[0] // 8, settings.obs_dims[1] // 8, -1))
        cnn_1 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same", activation="relu")
        cnn_2 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
        cnn_3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
        cnn_4 = Conv2D(image_channels, (5, 5), padding="same", activation="relu")
        drop_1 = Dropout(rate=settings.dropout_rate)
        drop_2 = Dropout(rate=settings.dropout_rate)
        drop_3 = Dropout(rate=settings.dropout_rate)

        prep_in = reshape_in(dense_in(model_in))
        model_out = cnn_4(drop_3(cnn_3(drop_2(cnn_2(drop_1(cnn_1(prep_in)))))))
        return Model(inputs=model_in, outputs=model_out, name="decoder_model")

    @staticmethod
    def create_reward_model(settings: DreamerSettings) -> Model:
        # concat(representation, hidden state) -> (reward, done)
        model_in = Input(settings.repr_out_dims_flat, name="repr_out")
        dense_1 = Dense(64, "relu")
        dense_2 = Dense(64, "relu")
        reward = Dense(1, activation="linear")
        terminal = Dense(2)
        st_categorical = STGradsOneHotCategorical(2)
        argmax = ArgmaxLayer()

        prep = dense_2(dense_1(model_in))
        reward_out = reward(prep)
        terminal_out = argmax(st_categorical(terminal(prep)))
        return Model(inputs=model_in, outputs=[reward_out, terminal_out], name="reward_model")


@dataclass
class DreamerEnvWrapper(gym.Env): # TODO: think about extending this to AsyncVectorEnv
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
    pass
