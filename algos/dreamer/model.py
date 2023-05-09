from typing import List, Union, Callable

# info: disable verbose tensorflow logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import Model, Input
from keras.layers import \
    Dense, Conv2D, Conv2DTranspose, Dropout, Flatten, \
    Reshape, Softmax, Lambda, Concatenate, GRU
from keras.optimizers import Adam
from keras.losses import MSE, kullback_leibler_divergence as KLDiv

from algos.dreamer.config import DreamerSettings


class STGradsOneHotCategorical:
    def __init__(self, dims: Union[int, List[int]]):
        self.softmax = Softmax()
        # self.softmax = Lambda(lambda x: tf.nn.log_softmax(x + 1e-8))
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


def _create_history_model(settings: DreamerSettings) -> Model:
    # (hidden state t, action t, representation t) -> hidden state t+1

    h0_in = Input(settings.hidden_dims, name="h0")
    a0_in = Input(settings.action_dims, name="a0")
    z0_in = Input(settings.repr_dims, name="z0")

    concat_in = Concatenate()
    flatten_repr = Flatten()
    expand_timeseries = Lambda(lambda x: tf.expand_dims(x, axis=1))
    dense_in = Dense(settings.hidden_dims[0], activation="linear", name="gru_in")
    rnn = GRU(settings.hidden_dims[0], return_state=True)
    # TODO: think about stacking multiple GRU cells

    gru_in = dense_in(concat_in([flatten_repr(z0_in), a0_in]))
    _, h1_out = rnn(expand_timeseries(gru_in), initial_state=h0_in)
    return Model(inputs=[a0_in, h0_in, z0_in], outputs=h1_out, name="history_model")


def _create_transition_model(settings: DreamerSettings) -> Model:
    # hidden state t+1 -> representation t+1
    model_in = Input(settings.hidden_dims, name="h0")
    dense1 = Dense(400, activation="relu")
    dense2 = Dense(400, activation="relu")
    dense3 = Dense(settings.repr_dims_flat, name="trans_in")
    reshape = Reshape(settings.repr_dims)
    st_categorical = STGradsOneHotCategorical(settings.repr_dims[1])
    model_out = st_categorical(reshape(dense3(dense2(dense1(model_in)))))
    return Model(inputs=model_in, outputs=model_out, name="transition_model")


def _create_representation_model(settings: DreamerSettings) -> Model:
    # fuses encoded state and hidden state
    # stochastic model that outputs a distribution
    enc_in = Input(settings.enc_dims, name="enc_obs")
    h_in = Input(settings.hidden_dims, name="h1")
    concat = Concatenate()
    dense1 = Dense(400, activation="relu")
    dense2 = Dense(400, activation="relu")
    dense3 = Dense(settings.repr_dims_flat, name="rep_concat")
    reshape = Reshape(settings.repr_dims)
    st_categorical = STGradsOneHotCategorical(settings.repr_dims)
    model_out = st_categorical(reshape(dense3(dense2(dense1(concat([enc_in, h_in]))))))
    return Model(inputs=[enc_in, h_in], outputs=model_out, name="repr_model")


def _create_repr_output_model(settings: DreamerSettings) -> Model:
    # fuses representation and hidden state
    # input of decoder and reward prediction
    z_in = Input(settings.repr_dims, name="z1")
    h_in = Input(settings.hidden_dims, name="h1")
    concat = Concatenate()
    flatten = Flatten()
    model_out = concat([flatten(z_in), h_in])
    return Model(inputs=[h_in, z_in], outputs=model_out, name="repr_output_model")


def _create_state_encoder_model(settings: DreamerSettings) -> Model:
    # observation t -> encoded state t
    model_in = Input(settings.obs_dims, name="enc_out")
    norm_img = Lambda(lambda x: x / 127.5 - 1.0)
    cnn_1 = Conv2D(16, (5, 5), strides=(2, 2), padding="same", activation="relu")
    cnn_2 = Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
    cnn_3 = Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
    cnn_4 = Conv2D(8, (3, 3), padding="same", activation="relu")
    drop_1 = Dropout(rate=settings.dropout_rate)
    drop_2 = Dropout(rate=settings.dropout_rate)
    drop_3 = Dropout(rate=settings.dropout_rate)
    drop_4 = Dropout(rate=settings.dropout_rate)
    flatten = Flatten()
    dense_out = Dense(settings.enc_dims[0], activation="linear", name="enc_dense")

    img_in = norm_img(model_in)
    prep_model_convs = drop_4(cnn_4(drop_3(cnn_3(drop_2(cnn_2(drop_1(cnn_1(img_in))))))))
    model_out = dense_out(flatten(prep_model_convs))
    return Model(inputs=model_in, outputs=model_out, name="encoder_model")


def _create_state_decoder_model(settings: DreamerSettings) -> Model:
    # concat(representation t+1, hidden state t+1) -> (obs t+1)
    image_channels = settings.obs_dims[-1]
    upscale_source_dims = (settings.obs_dims[0] // 8 * settings.obs_dims[1] // 8) * 8

    model_in = Input(settings.repr_out_dims_flat, name="repr_out")
    dense_in = Dense(upscale_source_dims, activation="linear", name="dec_in")
    reshape_in = Reshape((settings.obs_dims[0] // 8, settings.obs_dims[1] // 8, -1))
    cnn_1 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same", activation="relu")
    cnn_2 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
    cnn_3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
    cnn_4 = Conv2D(16, (5, 5), padding="same", activation="relu")
    cnn_5 = Conv2D(image_channels, (1, 1), padding="same", activation="sigmoid")
    drop_1 = Dropout(rate=settings.dropout_rate)
    drop_2 = Dropout(rate=settings.dropout_rate)
    drop_3 = Dropout(rate=settings.dropout_rate)
    scale_255 = Lambda(lambda x: x * 255)

    prep_in = reshape_in(dense_in(model_in))
    model_out = scale_255(cnn_5(cnn_4(drop_3(cnn_3(drop_2(cnn_2(drop_1(cnn_1(prep_in)))))))))
    return Model(inputs=model_in, outputs=model_out, name="decoder_model")


def _create_reward_model(settings: DreamerSettings) -> Model:
    # concat(representation, hidden state) -> (reward, done)
    model_in = Input(settings.repr_out_dims_flat, name="repr_out")
    dense_1 = Dense(64, "relu", name="rew1")
    dense_2 = Dense(64, "relu", name="rew2")
    reward = Dense(1, activation="linear", name="rew3")
    terminal = Dense(1, activation="sigmoid", name="rew4")
    stop_grads = Lambda(lambda x: tf.stop_gradient(x))

    prep = dense_2(dense_1(stop_grads(model_in)))
    reward_out = reward(prep)
    terminal_out = terminal(prep)
    return Model(inputs=model_in, outputs=[reward_out, terminal_out], name="reward_model")


class DreamerModelComponents:
    def __init__(self, settings: DreamerSettings):
        self.settings = settings
        self.history_model = _create_history_model(settings)
        self.encoder_model = _create_state_encoder_model(settings)
        self.repr_model = _create_representation_model(settings)
        self.trans_model = _create_transition_model(settings)
        self.repr_out_model = _create_repr_output_model(settings)
        self.decoder_model = _create_state_decoder_model(settings)
        self.reward_model = _create_reward_model(settings)

    def save_weights(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.history_model.save_weights(os.path.join(directory, f"history_model.h5"))
        self.trans_model.save_weights(os.path.join(directory, f"trans_model.h5"))
        self.repr_model.save_weights(os.path.join(directory, f"repr_model.h5"))
        self.repr_out_model.save_weights(os.path.join(directory, f"repr_out_model.h5"))
        self.encoder_model.save_weights(os.path.join(directory, f"encoder_model.h5"))
        self.decoder_model.save_weights(os.path.join(directory, f"decoder_model.h5"))
        self.reward_model.save_weights(os.path.join(directory, f"reward_model.h5"))

    def load_weights(self, directory: str):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The specified directory '{directory}' does not exist!")
        self.history_model.load_weights(os.path.join(directory, f"history_model.h5"))
        self.trans_model.load_weights(os.path.join(directory, f"trans_model.h5"))
        self.repr_model.load_weights(os.path.join(directory, f"repr_model.h5"))
        self.repr_out_model.load_weights(os.path.join(directory, f"repr_out_model.h5"))
        self.encoder_model.load_weights(os.path.join(directory, f"encoder_model.h5"))
        self.decoder_model.load_weights(os.path.join(directory, f"decoder_model.h5"))
        self.reward_model.load_weights(os.path.join(directory, f"reward_model.h5"))

    def compose_models(self):
        s1 = Input(self.settings.obs_dims, name="s1")
        a0 = Input(self.settings.action_dims, name="a0")
        h0 = Input(self.settings.hidden_dims, name="h0")
        z0 = Input(self.settings.repr_dims, name="z0")

        history_model = self.history_model
        trans_model = self.trans_model
        repr_model = self.repr_model
        repr_out_model = self.repr_out_model
        encoder_model = self.encoder_model
        decoder_model = self.decoder_model
        reward_model = self.reward_model

        s1_enc = encoder_model(s1)
        h1 = history_model((a0, h0, z0))
        z1 = repr_model((s1_enc, h1))
        z1_hat = trans_model(h1)
        out = repr_out_model((h1, z1))
        r1_hat = reward_model(out)
        s1_hat = decoder_model(out)
        env_model = Model(
            inputs=[s1, a0, h0, z0],
            outputs=[z1_hat, s1_hat, r1_hat, h1, z1],
            name="env_model")

        h1 = history_model((a0, h0, z0))
        z1_hat = trans_model(h1)
        out = repr_out_model((h1, z1_hat))
        r1_hat = reward_model(out)
        dream_model = Model(
            inputs=[a0, h0, z0],
            outputs=[r1_hat, h1, z1_hat],
            name="dream_model")

        s1_enc = encoder_model(s1)
        h1 = history_model((a0, h0, z0))
        z1 = repr_model((s1_enc, h1))
        step_model = Model(
            inputs=[s1, a0, h0, z0],
            outputs=[h1, z1],
            name="step_model")

        out = repr_out_model((h0, z0))
        s1_hat = decoder_model(out)
        render_model = Model(
            inputs=[h0, z0],
            outputs=[s1_hat],
            name="render_model")

        return env_model, dream_model, step_model, render_model


class DreamerModel:
    def __init__(
            self, settings: DreamerSettings,
            model_comps: DreamerModelComponents=None,
            loss_logger: Callable[[float, float, float, float], None]=lambda l1, l2, l3, l4: None):
        self.settings = settings
        self.model_comps = model_comps if model_comps else DreamerModelComponents(settings)
        self.loss_logger = loss_logger
        self.optimizer = Adam(learning_rate=1e-4, epsilon=1e-5)
        self.env_model, self.dream_model, self.step_model, self.render_model = \
            self.model_comps.compose_models()

    def seed(self, seed: int):
        tf.random.set_seed(seed)

    def save(self, directory: str):
        self.model_comps.save_weights(directory)

    def load(self, directory: str):
        self.model_comps.load_weights(directory)

    def summary(self):
        self.env_model.summary()
        self.dream_model.summary()
        self.step_model.summary()
        self.render_model.summary()

    def bootstrap(self, initial_state):
        batch_size = initial_state.shape[0]
        a_in = tf.zeros([batch_size] + self.settings.action_dims)
        h_in = tf.zeros([batch_size] + self.settings.hidden_dims)
        z_in = tf.zeros([batch_size] + self.settings.repr_dims)
        return self.step_model((initial_state, a_in, h_in, z_in))

    def step(self, s1, a0, h0, z0):
        return self.step_model((s1, a0, h0, z0))

    def dream_bootstrap(self, s0_init, a0_init):
        assert s0_init.shape[1] == a0_init.shape[1] + 1
        steps = a0_init.shape[1]
        h_in, z_in = self.bootstrap(s0_init[:, 0])
        for t in range(0, steps):
            s0, a0 = s0_init[:, t+1], a0_init[:, t]
            h_in, z_in = self.step(s0, a0, h_in, z_in)
        return h_in, z_in

    def dream_step(self, a, h, z):
        return self.dream_model((a, h, z))

    def render(self, h, z):
        return self.render_model((h, z))

    @tf.function
    def train(self, s0_init, a0_init, s1, a0, r1, t1):
        ALPHA = 0.8

        assert s0_init.shape[1] == a0_init.shape[1] + 1
        bootstrap_steps = a0_init.shape[1]
        batch_size = a0_init.shape[0]
        a_zero = tf.zeros([batch_size] + self.settings.action_dims)
        h0 = tf.zeros([batch_size] + self.settings.hidden_dims)
        z0 = tf.zeros([batch_size] + self.settings.repr_dims)

        with tf.GradientTape() as tape:
            h0, z0 = self.step_model((s0_init[:, 0], a_zero, h0, z0))
            for t in range(0, bootstrap_steps):
                h0, z0 = self.step_model((s0_init[:, t+1], a0_init[:, t], h0, z0))
            z1_hat, s1_hat, (r1_hat, term_hat), _, z1 = self.env_model((s1, a0, h0, z0))

            # z1 = tf.reshape(z1, (-1, z1.shape[-1]))
            # z1_hat = tf.reshape(z1_hat, (-1, z1_hat.shape[-1]))

            obs_loss = tf.reduce_mean(MSE(tf.cast(s1, dtype=tf.float32) / 255.0, s1_hat / 255.0))
            repr_loss = ALPHA * tf.reduce_mean(KLDiv(tf.stop_gradient(z1), z1_hat)) \
                + (1 - ALPHA) * tf.reduce_mean(KLDiv(z1, tf.stop_gradient(z1_hat)))
            reward_loss = tf.reduce_mean(MSE(r1, r1_hat))
            term_loss = tf.reduce_mean(MSE(t1, term_hat))
            loss = obs_loss + repr_loss + reward_loss + term_loss

            grads = tape.gradient(loss, self.env_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.env_model.trainable_variables))
            self.loss_logger(obs_loss, repr_loss, reward_loss, term_loss)
