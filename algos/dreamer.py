from typing import List, Union, Callable
from dataclasses import dataclass, field
from threading import Thread
from time import sleep
from signal import signal, SIGINT

import gym
import numpy as np

import pygame
from PIL import Image

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
from keras.losses import MSE, kullback_leibler_divergence, sparse_categorical_crossentropy


@dataclass
class Trajectory:
    states: np.ndarray
    zs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    def __post_init__(self):
        if self.actions.shape[0] != self.rewards.shape[0]:
            raise ValueError(
                "Invalid trajectory! The amount of actions and rewards needs to be the same!")
        if self.actions.shape[0] + 1 != self.states.shape[0] \
                or self.states.shape[0] != self.zs.shape[0]:
            raise ValueError(
                f"Invalid trajectory! Expected {self.rewards.shape[0] + 1} states!")

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
        self.env_model, self.dream_model, self.step_model, self.render_model = \
            DreamerModel._create_models(settings)

    def seed(self, seed: int):
        tf.random.set_seed(seed)

    def summary(self):
        self.env_model.summary()
        self.dream_model.summary()
        self.step_model.summary()

    def bootstrap(self, initial_state: np.ndarray):
        batch_size = initial_state.shape[0]
        a_in = tf.zeros([batch_size] + self.settings.action_dims)
        h_in = tf.zeros([batch_size] + self.settings.hidden_dims)
        z_in = tf.zeros([batch_size] + self.settings.repr_dims)
        _, __, ___, h0, z0 = self.env_model((initial_state, a_in, h_in, z_in))
        return h0, z0

    @tf.function
    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        BETA = 0.1
        timesteps = actions.shape[0]
        loss = 0.0
        with tf.GradientTape() as tape:

            h0, z0 = self.bootstrap(tf.expand_dims(states[0], axis=0))

            # TODO: define model as recurrent cell to unroll
            #       multiple steps at once with standard api

            # unroll trajectory
            for t in range(timesteps):
                a0, r1, s1 = actions[t], rewards[t], states[t+1]
                a0 = tf.expand_dims(a0, axis=0)
                r1 = tf.expand_dims(r1, axis=0)
                s1 = tf.expand_dims(s1, axis=0)
                term = tf.constant(1.0 if t == timesteps - 1 else 0.0, shape=(1, 1))

                z1_hat, s1_hat, (r1_hat, term_hat), h1, z1 = \
                    self.env_model((s1, a0, tf.stop_gradient(h0), tf.stop_gradient(z0)))

                reward_loss = MSE(r1, r1_hat)
                obs_loss = MSE(s1, s1_hat)
                term_loss = MSE(term, term_hat)
                repr_loss = BETA * tf.reduce_mean(kullback_leibler_divergence(z1, z1_hat))
                loss += reward_loss + obs_loss + term_loss + repr_loss

                h0, z0 = h1, z1

        grads = tape.gradient(loss, self.env_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.env_model.trainable_variables))

    @staticmethod
    def _create_models(settings: DreamerSettings):
        history_model = DreamerModel._create_history_model(settings)
        trans_model = DreamerModel._create_transition_model(settings)
        repr_model = DreamerModel._create_representation_model(settings)
        repr_out_model = DreamerModel._create_repr_output_model(settings)
        encoder_model = DreamerModel._create_state_encoder_model(settings)
        decoder_model = DreamerModel._create_state_decoder_model(settings)
        reward_model = DreamerModel._create_reward_model(settings)

        s1 = Input(settings.obs_dims, name="s1")
        a0 = Input(settings.action_dims, name="a0")
        h0 = Input(settings.hidden_dims, name="h0")
        z0 = Input(settings.repr_dims, name="z0")

        s1_enc = encoder_model(s1)
        h1 = history_model((a0, z0, h0))
        z1 = repr_model((s1_enc, h1))
        z1_hat = trans_model(h1)
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

        s1_enc = encoder_model(s1)
        h1 = history_model((a0, z0, h0))
        z1 = repr_model((s1_enc, h1))
        step_model = Model(
            inputs=[s1, a0, h0, z0],
            outputs=[h1, z1],
            name="step_model")

        out = repr_out_model((z0, h0))
        s1_hat = decoder_model(out)
        render_model = Model(
            inputs=[z0, h0],
            outputs=[s1_hat],
            name="render_model")

        return env_model, dream_model, step_model, render_model

    @staticmethod
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
        return Model(inputs=[a0_in, z0_in, h0_in], outputs=h1_out, name="history_model")

    @staticmethod
    def _create_transition_model(settings: DreamerSettings) -> Model:
        # hidden state t+1 -> representation t+1
        model_in = Input(settings.hidden_dims, name="h0")
        dense = Dense(settings.repr_dims_flat, name="trans_in")
        reshape = Reshape(settings.repr_dims)
        st_categorical = STGradsOneHotCategorical(settings.repr_dims[1])
        model_out = st_categorical(reshape(dense(model_in)))
        return Model(inputs=model_in, outputs=model_out, name="transition_model")

    @staticmethod
    def _create_representation_model(settings: DreamerSettings) -> Model:
        # fuses encoded state and hidden state
        # stochastic model that outputs a distribution
        enc_in = Input(settings.enc_dims, name="enc_obs")
        h_in = Input(settings.hidden_dims, name="h1")
        concat = Concatenate()
        dense = Dense(settings.repr_dims_flat, name="rep_concat")
        reshape = Reshape(settings.repr_dims)
        st_categorical = STGradsOneHotCategorical(settings.repr_dims)
        model_out = st_categorical(reshape(dense(concat([enc_in, h_in]))))
        return Model(inputs=[enc_in, h_in], outputs=model_out, name="repr_model")

    @staticmethod
    def _create_repr_output_model(settings: DreamerSettings) -> Model:
        # fuses representation and hidden state
        # input of decoder and reward prediction
        repr_in = Input(settings.repr_dims, name="z1")
        h_in = Input(settings.hidden_dims, name="h1")
        concat = Concatenate()
        flatten = Flatten()
        model_out = concat([flatten(repr_in), h_in])
        return Model(inputs=[repr_in, h_in], outputs=model_out, name="repr_output_model")

    @staticmethod
    def _create_state_encoder_model(settings: DreamerSettings) -> Model:
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
        dense_out = Dense(settings.enc_dims[0], activation="linear", name="enc_dense")

        prep_model_convs = drop_4(cnn_4(drop_3(cnn_3(drop_2(cnn_2(drop_1(cnn_1(model_in))))))))
        model_out = dense_out(flatten(prep_model_convs))
        return Model(inputs=model_in, outputs=model_out, name="encoder_model")

    @staticmethod
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

    @staticmethod
    def _create_reward_model(settings: DreamerSettings) -> Model:
        # concat(representation, hidden state) -> (reward, done)
        model_in = Input(settings.repr_out_dims_flat, name="repr_out")
        dense_1 = Dense(64, "relu", name="rew1")
        dense_2 = Dense(64, "relu", name="rew2")
        reward = Dense(1, activation="linear", name="rew3")
        terminal = Dense(2, name="rew4")
        st_categorical = STGradsOneHotCategorical(2)
        argmax = ArgmaxLayer()

        prep = dense_2(dense_1(model_in))
        reward_out = reward(prep)
        terminal_out = argmax(st_categorical(terminal(prep)))
        return Model(inputs=model_in, outputs=[reward_out, terminal_out], name="reward_model")


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class DreamerDebugDisplay:
    def __init__(self, img_width: int, img_height: int, scaling: float=1):
        self.scaling = scaling
        self.img_width, self.img_height = img_width * scaling, img_height * scaling
        pygame.init()
        pygame.font.init()

        window_shape = (2 * self.img_width + 30, self.img_height + 50)
        self.screen = pygame.display.set_mode(window_shape, pygame.RESIZABLE)
        self.std_font = pygame.font.SysFont('Consolas', 14, bold=True)

        display_shape = (self.img_width, self.img_height)
        label_shape = (self.img_width, 30)
        self.display_orig = pygame.surface.Surface(display_shape)
        self.display_hall = pygame.surface.Surface(display_shape)
        self.label_orig = pygame.surface.Surface(label_shape)
        self.label_hall = pygame.surface.Surface(label_shape)

        def render_text(text: str) -> pygame.surface.Surface:
            return self.std_font.render(text, True, BLACK, WHITE)

        self.label_orig_text = render_text("original")
        self.label_hall_text = render_text("hallucinated")

        self.label_orig_offset = (10, 10)
        self.label_hall_offset = (self.img_width + 20, 10)
        self.display_orig_offset = (10, 40)
        self.display_hall_offset = (self.img_width + 20, 40)

        self.is_exit_requested = False

        def process_event_queue():
            while not self.is_exit_requested:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        self.is_exit_requested = True
                sleep(0.01)

        self.ui_events_thread = Thread(target=process_event_queue)
        self.ui_events_thread.start()

        def handle_sigint(signum, frame):
            self.is_exit_requested = True

        signal(SIGINT, handle_sigint)

    def clear(self):
        self.screen.fill(WHITE)
        pygame.display.update()

    def next_frame(self, frame_orig: np.ndarray, frame_hall: np.ndarray):
        if self.is_exit_requested:
            self.ui_events_thread.join()
            pygame.quit()
            exit()

        self.screen.fill(WHITE)

        def show_text_centered(
                surface: pygame.surface.Surface,
                text: pygame.surface.Surface):
            total_width, total_height = surface.get_size()
            text_width, text_height = text.get_size()
            x_offset = total_width / 2 - text_width / 2
            y_offset = total_height / 2 - text_height / 2
            surface.fill(WHITE)
            surface.blit(text, (x_offset, y_offset))

        def prepare_frame(image: np.ndarray) -> np.ndarray:
            return np.rot90(np.fliplr(np.clip(image, 0, 255)))

        show_text_centered(self.label_orig, self.label_orig_text)
        show_text_centered(self.label_hall, self.label_hall_text)

        orig_surface = pygame.surfarray.make_surface(prepare_frame(frame_orig))
        hall_surface = pygame.surfarray.make_surface(prepare_frame(frame_hall))
        orig_surface = pygame.transform.scale(orig_surface, (self.img_width, self.img_height))
        hall_surface = pygame.transform.scale(hall_surface,  (self.img_width, self.img_height))
        self.display_orig.blit(orig_surface, (0, 0))
        self.display_hall.blit(hall_surface, (0, 0))

        self.screen.blit(self.display_orig, self.display_orig_offset)
        self.screen.blit(self.display_hall, self.display_hall_offset)
        self.screen.blit(self.label_orig, self.label_orig_offset)
        self.screen.blit(self.label_hall, self.label_hall_offset)

        pygame.display.update()


@dataclass
class DreamerEnvWrapper(gym.Env): # TODO: think about extending this to AsyncVectorEnv
    orig_env: gym.Env
    model: DreamerModel
    debug: bool = False
    display_factory: Callable[[], DreamerDebugDisplay] = field(default=lambda: None)
    h0: np.ndarray = field(init=False)
    z0: np.ndarray = field(init=False)
    orig_state: np.ndarray = field(init=False)
    display: DreamerDebugDisplay = field(init=False, default=None)

    def __post_init__(self):
        if self.debug and not self.display:
            self.display = self.display_factory()

    @property
    def observation_space(self):
        return self.orig_env.observation_space

    @property
    def action_space(self):
        return self.orig_env.action_space

    def reset(self):
        state = self.orig_env.reset()
        state = self._resize_image(state)
        self.h0, self.z0 = self.model.bootstrap(self._batch(state))
        self.orig_state = state
        return state, self._unbatch(self.z0)

    def step(self, action):
        state, reward, done, meta = self.orig_env.step(action)
        state = self._resize_image(state)
        inputs = (self._batch(state), self._batch(action), self.h0, self.z0)
        self.h0, self.z0 = self.model.step_model(inputs)
        self.orig_state = state
        return (state, self._unbatch(self.z0)), reward, done, meta

    def render(self, mode="human"):
        if not self.debug:
            raise RuntimeError(
                "Cannot render outside of debug mode! Consider setting debug=True!")
        if not self.display:
            raise RuntimeError(
                "Cannot render without display! Consider initializing the display factory!")

        hallucinated_state = self.model.render_model((self.z0, self.h0))
        self.display.next_frame(self.orig_state, self._unbatch(hallucinated_state))

    def seed(self, seed: int):
        self.orig_env.seed(seed) # seed environment
        self.model.seed(seed)

    def _batch(self, arr: np.ndarray) -> np.ndarray:
        return np.expand_dims(arr, axis=0)

    def _unbatch(self, arr: np.ndarray) -> np.ndarray:
        return np.squeeze(arr, axis=0)

    def _resize_image(self, orig_image: np.ndarray) -> np.ndarray:
        width, height, _ = self.model.settings.obs_dims
        img = Image.fromarray(orig_image)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(img)
