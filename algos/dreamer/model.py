from typing import Callable

# info: disable verbose tensorflow logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from keras import Model, Input
from keras.layers import \
    Dense, Conv2D, Conv2DTranspose, Dropout, Flatten, \
    Reshape, Lambda, Concatenate, GRU
from keras.optimizers import Adam
from keras.losses import MSE, kullback_leibler_divergence as KLDiv

from algos.dreamer.config import DreamerSettings
from algos.dreamer.layers import \
    STOneHotCategorical, VQCodebook, VQCombined


def create_encoder(settings: DreamerSettings) -> Model:
    model_in = Input(settings.obs_dims, name="obs")
    norm_img = Lambda(lambda x: x / 127.5 - 1.0)
    cnn_1 = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="elu")
    cnn_2 = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="elu")
    cnn_3 = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="elu")
    cnn_4 = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="elu")
    drop_1 = Dropout(rate=settings.dropout_rate)
    drop_2 = Dropout(rate=settings.dropout_rate)
    drop_3 = Dropout(rate=settings.dropout_rate)

    img_in = norm_img(model_in)
    z_enc = cnn_4(drop_3(cnn_3(drop_2(cnn_2(drop_1(cnn_1(img_in)))))))
    return Model(inputs=model_in, outputs=z_enc, name="encoder_model")


def create_decoder(settings: DreamerSettings) -> Model:
    model_in = Input(settings.obs_enc_dims, name="obs_enc")
    cnn_1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="elu")
    cnn_2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="elu")
    cnn_3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="elu")
    cnn_4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="elu")
    cnn_5 = Conv2D(settings.obs_dims[-1], (1, 1), padding="same", activation="linear")
    drop_1 = Dropout(rate=settings.dropout_rate)
    drop_2 = Dropout(rate=settings.dropout_rate)
    drop_3 = Dropout(rate=settings.dropout_rate)
    scale_255 = Lambda(lambda x: x * 255)

    model_out = scale_255(cnn_5(cnn_4(drop_3(cnn_3(drop_2(cnn_2(drop_1(cnn_1(model_in)))))))))
    return Model(inputs=model_in, outputs=model_out, name="decoder_model")


def create_vqvae(settings: DreamerSettings):
    encoder = create_encoder(settings)
    decoder = create_decoder(settings)
    vq = VQCombined(settings.repr_dims[0], settings.repr_dims[1], name="vq_z")

    obs_in = Input(settings.obs_dims, name="obs")
    z_enc = encoder(obs_in)
    z_quant, z_cat = vq(z_enc)
    obs_reconst = decoder(z_enc + tf.stop_gradient(z_quant - z_enc))

    z_cat_in = Input(settings.repr_dims)
    z_hat_quant = vq.vq_codebook(z_cat_in)
    obs_dream = decoder(z_hat_quant)

    vqvae = Model(inputs=obs_in, outputs=[obs_reconst, z_enc, z_quant])
    obs_encoder = Model(inputs=obs_in, outputs=[z_quant, z_cat])
    dream_render = Model(inputs=z_cat_in, outputs=obs_dream)
    return vqvae, obs_encoder, dream_render, vq.vq_codebook


def create_vqrnn(settings: DreamerSettings):
    rnn = GRU(settings.hidden_dims[0], return_state=True)
    vq = VQCombined(settings.repr_dims[0], settings.repr_dims[1], name="vq_h")

    h0_quant_in = Input(settings.hidden_dims)
    x_in = Input([1] + settings.hidden_dims)

    _, h1_enc = rnn(x_in, initial_state=h0_quant_in)
    h1_quant, h1_cat = vq(h1_enc)
    h1_st_out = h1_enc + tf.stop_gradient(h1_quant - h1_enc)

    vqrnn = Model(inputs=[x_in, h0_quant_in], outputs=[h1_quant, h1_cat])
    vqrnn_train = Model(inputs=[x_in, h0_quant_in], outputs=[h1_enc, h1_quant, h1_cat, h1_st_out])
    return vqrnn, vqrnn_train, vq.vq_codebook


def create_vqrnn_embedding(
        settings: DreamerSettings, z_codebook: VQCodebook, h_codebook: VQCodebook):
    concat = Concatenate()
    flatten = Flatten()
    expand_timeseries = Reshape([-1] + settings.hidden_dims)
    fuse_z_a = Dense(settings.hidden_dims[-1], activation="linear")

    action_in = Input(settings.action_dims)
    h_cat_in = Input(settings.repr_dims)
    z_cat_in = Input(settings.repr_dims)

    z_quant = z_codebook(z_cat_in)
    h_quant = h_codebook(h_cat_in)
    x = expand_timeseries(fuse_z_a(concat((flatten(z_quant), action_in))))
    embed_ahz_cat = Model(inputs=[action_in, h_cat_in, z_cat_in], outputs=[x, h_quant])

    z_quant_in = Input(settings.obs_enc_dims)
    x = expand_timeseries(fuse_z_a(concat((flatten(z_quant_in), action_in))))
    fuse_az_rnn_input = Model(inputs=[action_in, z_quant_in], outputs=[x])
    return embed_ahz_cat, fuse_az_rnn_input


def create_dream_trans(settings: DreamerSettings, z_codebook: VQCodebook) -> Model:
    model_in = Input(settings.hidden_dims, name="h1_quant")
    dense1 = Dense(400, activation="relu")
    dense2 = Dense(400, activation="relu")
    dense3 = Dense(settings.repr_dims_flat)
    reshape = Reshape(settings.repr_dims)
    st_cat = STOneHotCategorical(settings.repr_dims[1])

    z1_cat = st_cat(reshape(dense3(dense2(dense1(model_in)))))
    z1_quant = z_codebook(z1_cat)
    return Model(inputs=model_in, outputs=[z1_cat, z1_quant], name="transition_model")


def create_reward_pred(settings: DreamerSettings) -> Model:
    flatten = Flatten()
    concat = Concatenate()
    dense_1 = Dense(64, "elu")
    dense_2 = Dense(64, "elu")
    reward = Dense(1, activation="linear")
    terminal = Dense(1, activation="sigmoid")
    stop_grads = Lambda(lambda x: tf.stop_gradient(x))

    h_quant_in = Input(settings.hidden_dims, name="h_quant_in")
    z_quant_in = Input(settings.obs_enc_dims, name="z_quant_in")

    prep = stop_grads(concat((flatten(z_quant_in), h_quant_in)))
    out = dense_2(dense_1(prep))
    reward_out = reward(out)
    terminal_out = terminal(out)
    return Model(inputs=[h_quant_in, z_quant_in], outputs=[reward_out, terminal_out], name="reward_model")


class DreamerModelComponents:
    def __init__(self, settings: DreamerSettings):
        self.settings = settings
        self.vqvae, self.obs_encoder, self.dream_render, self.z_codebook = create_vqvae(settings)
        self.vqrnn, self.vqrnn_train, self.h_codebook = create_vqrnn(settings)
        self.embed_ahz_cat, self.fuse_az_rnn_input = create_vqrnn_embedding(
            settings, self.z_codebook, self.h_codebook)
        self.dream_trans = create_dream_trans(settings, self.z_codebook)
        self.reward_pred = create_reward_pred(settings)

    def save_weights(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.vqvae.save_weights(os.path.join(directory, f"vqvae.h5"))
        self.obs_encoder.save_weights(os.path.join(directory, f"obs_encoder.h5"))
        self.dream_render.save_weights(os.path.join(directory, f"dream_render.h5"))
        self.vqrnn.save_weights(os.path.join(directory, f"vqrnn.h5"))
        self.embed_ahz_cat.save_weights(os.path.join(directory, f"embed_ahz_cat.h5"))
        self.fuse_az_rnn_input.save_weights(os.path.join(directory, f"fuse_az_rnn_input.h5"))
        self.dream_trans.save_weights(os.path.join(directory, f"dream_trans.h5"))
        self.reward_pred.save_weights(os.path.join(directory, f"reward_pred.h5"))

    def load_weights(self, directory: str):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The specified directory '{directory}' does not exist!")
        self.vqvae.load_weights(os.path.join(directory, f"vqvae.h5"))
        self.obs_encoder.load_weights(os.path.join(directory, f"obs_encoder.h5"))
        self.dream_render.load_weights(os.path.join(directory, f"dream_render.h5"))
        self.vqrnn.load_weights(os.path.join(directory, f"vqrnn.h5"))
        self.embed_ahz_cat.load_weights(os.path.join(directory, f"embed_ahz_cat.h5"))
        self.fuse_az_rnn_input.load_weights(os.path.join(directory, f"fuse_az_rnn_input.h5"))
        self.dream_trans.load_weights(os.path.join(directory, f"dream_trans.h5"))
        self.reward_pred.load_weights(os.path.join(directory, f"reward_pred.h5"))

    def compose_models(self):
        s1 = Input(self.settings.obs_dims, name="s1")
        a0 = Input(self.settings.action_dims, name="a0")
        h0_cat = Input(self.settings.repr_dims, name="h0_cat")
        z0_cat = Input(self.settings.repr_dims, name="z0_cat")
        h0_quant = Input(self.settings.hidden_dims, name="h0_quant")
        z0_quant = Input(self.settings.obs_enc_dims, name="z0_quant")

        s1_hat, z1_enc, z1_quant = self.vqvae(s1)
        z1_quant, z1_cat = self.obs_encoder(s1)
        x0 = self.fuse_az_rnn_input((a0, tf.stop_gradient(z0_quant)))
        h1_enc, h1_quant, _, h1_st = self.vqrnn_train((x0, h0_quant))
        z1_cat_hat, _ = self.dream_trans(h1_st)
        r1_hat = self.reward_pred((h1_st, z1_quant))
        train_model = Model(
            inputs=[s1, a0, h0_quant, z0_quant],
            outputs=[z1_enc, z1_quant, h1_enc, h1_quant, s1_hat, r1_hat, z1_cat, z1_cat_hat],
            name="train_model")

        x0, h0_quant = self.embed_ahz_cat((a0, h0_cat, z0_cat))
        h1_quant, h1_cat = self.vqrnn((x0, h0_quant))
        z1_cat, z1_quant = self.dream_trans(h1_quant)
        r1_hat = self.reward_pred((h1_quant, z1_quant))
        dream_model = Model(
            inputs=[a0, h0_cat, z0_cat],
            outputs=[r1_hat, h1_cat, z1_cat],
            name="dream_model")

        z1_quant, z1_cat = self.obs_encoder(s1)
        x0, h0_quant = self.embed_ahz_cat((a0, h0_cat, z0_cat))
        h1_quant, h1_cat = self.vqrnn((x0, h0_quant))
        step_model = Model(
            inputs=[s1, a0, h0_cat, z0_cat],
            outputs=[h1_cat, z1_cat],
            name="step_model")

        z1_quant, _ = self.obs_encoder(s1)
        x0 = self.fuse_az_rnn_input((a0, tf.stop_gradient(z0_quant)))
        h1_quant, _ = self.vqrnn((x0, h0_quant))
        train_step_model = Model(
            inputs=[s1, a0, h0_quant, z0_quant],
            outputs=[h1_quant, z1_quant],
            name="step_model_diff")

        return train_model, dream_model, step_model, train_step_model, self.dream_render


class DreamerModel:
    def __init__(
            self, settings: DreamerSettings,
            model_comps: DreamerModelComponents=None,
            loss_logger: Callable[[float, float, float, float], None]=lambda l1, l2, l3, l4: None):
        self.settings = settings
        self.model_comps = model_comps if model_comps else DreamerModelComponents(settings)
        self.loss_logger = loss_logger
        self.optimizer = Adam(learning_rate=1e-4, epsilon=1e-5)
        self.train_model, self.dream_model, self.step_model, self.train_step_model, self.render_model = \
            self.model_comps.compose_models()

    def seed(self, seed: int):
        tf.random.set_seed(seed)

    def save(self, directory: str):
        self.model_comps.save_weights(directory)

    def load(self, directory: str):
        self.model_comps.load_weights(directory)

    def summary(self):
        self.train_model.summary()
        self.dream_model.summary()
        self.step_model.summary()
        self.render_model.summary()

    def bootstrap(self, initial_state):
        batch_size = initial_state.shape[0]
        a_in = tf.zeros([batch_size] + self.settings.action_dims)
        h_in = tf.zeros([batch_size] + self.settings.repr_dims)
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

    def render(self, z):
        return self.render_model(z)

    @tf.function
    def train(self, s0_init, a0_init, s1, a0, r1, t1):
        ALPHA = 0.2

        assert s0_init.shape[1] == a0_init.shape[1] + 1
        bootstrap_steps = a0_init.shape[1]
        batch_size = a0_init.shape[0]
        a_zero = tf.zeros([batch_size] + self.settings.action_dims)
        h0_quant = tf.zeros([batch_size] + self.settings.hidden_dims)
        z0_quant = tf.zeros([batch_size] + self.settings.obs_enc_dims)

        with tf.GradientTape() as tape:
            h0_quant, z0_quant = self.train_step_model((s0_init[:, 0], a_zero, h0_quant, z0_quant))
            for t in range(0, bootstrap_steps):
                h0_quant, z0_quant = self.train_step_model((s0_init[:, t+1], a0_init[:, t], h0_quant, z0_quant))
            z1_enc, z1_quant, h1_enc, h1_quant, s1_hat, (r1_hat, term_hat), z1_cat, z1_cat_hat \
                = self.train_model((s1, a0, h0_quant, z0_quant))

            committment_loss = tf.reduce_mean((tf.stop_gradient(z1_quant) - z1_enc) ** 2)
            codebook_loss = tf.reduce_mean((z1_quant - tf.stop_gradient(z1_enc)) ** 2)
            vqvae_loss = self.settings.committment_cost * committment_loss + codebook_loss

            committment_loss = tf.reduce_mean((tf.stop_gradient(h1_quant) - h1_enc) ** 2)
            codebook_loss = tf.reduce_mean((h1_quant - tf.stop_gradient(h1_enc)) ** 2)
            vqrnn_loss = self.settings.committment_cost * committment_loss + codebook_loss

            reconst_loss = tf.reduce_mean(MSE(tf.cast(s1, dtype=tf.float32) / 255.0, s1_hat / 255.0))
            repr_loss = tf.reduce_mean(KLDiv(tf.stop_gradient(z1_cat), z1_cat_hat))
                #+ (1 - ALPHA) * tf.reduce_mean(KLDiv(z1_cat, tf.stop_gradient(z1_cat_hat)))
            reward_loss = tf.reduce_mean(MSE(r1, r1_hat))
            term_loss = tf.reduce_mean(MSE(t1, term_hat))
            loss = vqvae_loss + vqrnn_loss + reconst_loss + reward_loss + term_loss + repr_loss

            grads = tape.gradient(loss, self.train_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.train_model.trainable_variables))
            self.loss_logger(reconst_loss, repr_loss, reward_loss, term_loss)
