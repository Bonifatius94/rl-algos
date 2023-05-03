from typing import Any, Callable, Tuple, List, Union, Iterator
from dataclasses import dataclass, field

import gym
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from keras.optimizers import Optimizer, Adam
from keras.metrics import Metric, Mean


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


@dataclass
class PPOTrainingSettings:
    obs_shape: List[int]
    num_actions: int
    total_steps: int = 10_000_000
    learn_rate: float = 3e-4
    reward_discount: float = 0.99
    gae_discount: float = 0.95
    prob_clip: float = 0.2
    value_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    norm_advantages: bool = True
    clip_values: bool = True
    batch_size: int = 64
    n_envs: int = 32
    steps_per_update: int = 512
    update_epochs: int = 4
    num_model_snapshots: int = 20

    def __post_init__(self):
        if self.n_envs * self.steps_per_update % self.batch_size != 0:
            print((f"WARNING: training examples will be cut because "
                f"{self.n_envs} environments * {self.steps_per_update} steps per update interval "
                f"isn't divisible by a mini-batch size of {self.batch_size}!"))

    @property
    def train_steps(self) -> int:
        return self.total_steps // self.n_envs

    @property
    def model_snapshot_interval(self) -> int:
        return self.train_steps // self.num_model_snapshots


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

    def save(self, step: Union[int, None] = None):
        model_file = f"model/ppo_model_{step}.h5" if step else f"model/ppo_model_final.h5"
        self.model.save_weights(model_file, overwrite=True)


@dataclass
class PPOVecEnvRolloutBuffer:
    config: PPOTrainingSettings
    consume_batch: Callable[[List[TrainingBatch]], None]
    prob_dist_cache: np.ndarray = field(init=False)
    baseline_cache: np.ndarray = field(init=False)
    s0_cache: np.ndarray = field(init=False)
    actinon_cache: np.ndarray = field(init=False)
    reward_cache: np.ndarray = field(init=False)
    done_cache: np.ndarray = field(init=False)
    step: int = 0

    def __post_init__(self):
        steps, n_envs = self.config.steps_per_update + 1, self.config.n_envs
        obs_shape = [steps, n_envs] + [d for d in self.config.obs_shape]
        self.prob_dist_cache = np.zeros((steps, n_envs, self.config.num_actions), dtype=np.float32)
        self.baseline_cache = np.zeros((steps, n_envs), dtype=np.float32)
        self.s0_cache = np.zeros(obs_shape, dtype=np.float32)
        self.actinon_cache = np.zeros((steps, n_envs), dtype=np.int64)
        self.reward_cache = np.zeros((steps, n_envs), dtype=np.float32)
        self.done_cache = np.zeros((steps, n_envs), dtype=np.bool8)

    def append_step(self, preds: BatchedPredictions, exps: BatchedSarsExps):
        prob_dists, baselines = preds
        states_before, actions, rewards, _, dones = exps
        t = self.step

        self.prob_dist_cache[t] = prob_dists
        self.baseline_cache[t] = baselines
        self.s0_cache[t] = states_before
        self.actinon_cache[t] = actions
        self.reward_cache[t] = rewards
        self.done_cache[t] = dones
        self.step += 1

        if self.step == self.config.steps_per_update + 1:
            batch = self._sample_batch()
            mini_batches = self._partition_minibatches(batch)
            self.consume_batch(mini_batches)

            # info: sampling requires the baseline of last s1
            #       -> cache 1 more step, assign it to beginning of the next batch
            self.prob_dist_cache[0] = self.prob_dist_cache[t]
            self.baseline_cache[0] = self.baseline_cache[t]
            self.s0_cache[0] = self.s0_cache[t]
            self.actinon_cache[0] = self.actinon_cache[t]
            self.reward_cache[0] = self.reward_cache[t]
            self.done_cache[0] = self.done_cache[t]
            self.step = 1

    def _sample_batch(self) -> TrainingBatch:
        batch_steps = self.config.steps_per_update
        no_t_dim = lambda old_shape: [-1] + [d for d in old_shape[2:]]
        states = self.s0_cache[:batch_steps].reshape(no_t_dim(self.s0_cache.shape))
        actions = self.actinon_cache[:batch_steps].reshape(-1)
        advantages = self._compute_advantages()[:batch_steps].reshape(-1)
        returns = self.baseline_cache[:batch_steps].reshape(-1) + advantages
        prob_dists_old = self.prob_dist_cache[:batch_steps].reshape(no_t_dim(self.prob_dist_cache.shape))
        baselines_old = self.baseline_cache[:batch_steps].reshape(-1)
        return states, actions, returns, advantages, prob_dists_old, baselines_old

    def _compute_advantages(self) -> BatchedAdvantages:
        baselines, rewards = self.baseline_cache, self.reward_cache
        reward_discount, gae_discount = self.config.reward_discount, self.config.gae_discount
        advantages, last_gae_lam = np.zeros_like(rewards), 0
        for t in reversed(range(self.config.steps_per_update)):
            next_nonterminal = 1.0 - self.done_cache[t+1]
            delta = rewards[t] + baselines[t+1] * next_nonterminal * reward_discount - baselines[t]
            last_gae_lam = delta + next_nonterminal * reward_discount * gae_discount * last_gae_lam
            advantages[t] = last_gae_lam
        return advantages

    def _partition_minibatches(self, big_batch: TrainingBatch) -> List[TrainingBatch]:
        num_examples = big_batch[0].shape[0]
        batch_size = self.config.batch_size
        states, actions, returns, advantages, prob_dists_old, baselines_old = big_batch
        ranges = [(i * batch_size, (i+1) * batch_size) for i in range(num_examples // batch_size)]
        mini_batches = [(states[i:j], actions[i:j], returns[i:j], advantages[i:j],
                         prob_dists_old[i:j], baselines_old[i:j])
                        for i, j in ranges]
        return mini_batches


@dataclass
class VecEnvTrainingSession:
    config: PPOTrainingSettings
    vec_env: gym.vector.vector_env.VectorEnv
    model: Callable[[BatchedObservations], BatchedPredictions]
    encode_obs: Callable[[Any], BatchedObservations]
    sample_actions: Callable[[BatchedPredictions], BatchedActions]
    exps_consumer: Callable[[BatchedPredictions, BatchedSarsExps], None]
    log_timestep: Callable[[int, BatchedRewards, BatchedDones], None]
    snapshot_model: Callable[[int], None]

    def training(self, steps: int):
        obs = self.encode_obs(self.vec_env.reset())

        for step in tqdm(range(steps)):
            predictions = self.model(obs)
            actions = self.sample_actions(predictions)
            next_obs, rewards, dones, _ = self.vec_env.step(actions)
            next_obs = self.encode_obs(next_obs)
            sars_exps = (obs, actions, rewards, next_obs, dones)
            self.exps_consumer(predictions, sars_exps)
            self.log_timestep(step, rewards, dones)
            obs = next_obs
            if (step + 1) % self.config.model_snapshot_interval == 0:
                self.snapshot_model(step)


@dataclass
class TensorboardLogging:
    step: int = 0
    policy_loss: Metric = Mean()
    value_loss: Metric = Mean()
    entropy_loss: Metric = Mean()

    def __post_init__(self):
        train_log_dir = 'logs/train'
        # TODO: add some timestamp to avoid overwriting previous training logs
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


@dataclass
class VecEnvEpisodeStats:
    n_envs: int
    log_episode: Callable[[int, float, float], None]
    ep_steps: np.ndarray = field(init=False)
    ep_rewards: np.ndarray = field(init=False)

    def __post_init__(self):
        self.ep_steps = np.zeros((self.n_envs), dtype=np.int64)
        self.ep_rewards = np.zeros((self.n_envs), dtype=np.float32)

    def log_step(self, step: int, rewards: BatchedRewards, dones: BatchedDones):
        self.ep_steps += np.ones((self.n_envs), dtype=np.int64)
        self.ep_rewards += rewards
        envs_in_final_state = np.where(dones)[0]
        num_dones = envs_in_final_state.shape[0]

        if num_dones > 0:
            avg_steps = sum(self.ep_steps[envs_in_final_state]) / num_dones
            avg_rewards = sum(self.ep_rewards[envs_in_final_state]) / num_dones
            self.log_episode(step, avg_rewards, avg_steps)
            self.ep_steps[envs_in_final_state] = 0
            self.ep_rewards[envs_in_final_state] = 0.0


class PPOAgent:
    def __init__(self, config: PPOTrainingSettings):
        self.config = config

        tb_logger = TensorboardLogging()
        model = PPOModel(config, tb_logger.log_training_loss, tb_logger.flush_losses)
        exp_buffer = PPOVecEnvRolloutBuffer(config, model.train)
        episode_logger = VecEnvEpisodeStats(config.n_envs, tb_logger.log_episode)

        encode_obs = lambda x: x
        sample_actions = lambda pred: \
            [int(np.random.choice(config.num_actions, 1, p=prob_dist)) for prob_dist in pred[0]]

        self.predict_action = lambda obs: model.predict(np.expand_dims(obs, axis=0))
        self.train_session_factory = lambda env: VecEnvTrainingSession(
            config, env, model.predict, encode_obs, sample_actions,
            exp_buffer.append_step, episode_logger.log_step, model.save)

    def act(self, obs: Any) -> Any:
        prob_dist, _ = self.predict_action(obs)
        prob_dist = np.squeeze(prob_dist)
        return int(np.random.choice(self.config.num_actions, 1, p=prob_dist))

    def train(self, env: gym.vector.VectorEnv, steps: int):
        session = self.train_session_factory(env)
        session.training(steps)


def train_pong():
    # obs space: (210, 160, 3), uint8)
    # action space: Discrete(6)
    config = PPOTrainingSettings([210, 160, 3], 6)
    make_env = lambda: gym.make("ALE/Pong-v5")
    vec_env = gym.vector.AsyncVectorEnv([make_env for _ in range(config.n_envs)])

    tb_logger = TensorboardLogging()
    model = PPOModel(config, tb_logger.log_training_loss, tb_logger.flush_losses)
    exp_buffer = PPOVecEnvRolloutBuffer(config, model.train)
    episode_logger = VecEnvEpisodeStats(config.n_envs, tb_logger.log_episode)

    sample_actions = lambda pred: \
        [int(np.random.choice(config.num_actions, 1, p=prob_dist)) for prob_dist in pred[0]]
    encode_obs = lambda obs: obs / 127.5 - 1.0
    session = VecEnvTrainingSession(
        config, vec_env, model.predict, encode_obs, sample_actions,
        exp_buffer.append_step, episode_logger.log_step, model.save)

    session.training(config.train_steps)
    model.save()


if __name__ == '__main__':
    train_pong()
