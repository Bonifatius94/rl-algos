from typing import Any, Callable, Tuple, List
from dataclasses import dataclass, field
from random import shuffle

import gym
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Dense, Input
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
    train_steps: int = 200_000
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
    n_envs: int = 16
    steps_per_update: int = 2048
    update_epochs: int = 4

    def __post_init__(self):
        if self.n_envs * self.steps_per_update % self.batch_size != 0:
            print((f"WARNING: training examples will be cut because "
                f"{self.n_envs} environments * {self.steps_per_update} steps per update interval "
                f"isn't divisible by a mini-batch size of {self.batch_size}!"))


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
        print(self.model.summary())

    @staticmethod
    def create_model(obs_shape: List[int], num_actions: int) -> Model:
        model_in = Input(obs_shape)
        actor_fc_1 = Dense(64, activation='relu')
        actor_fc_2 = Dense(64, activation='relu')
        critic_fc_1 = Dense(64, activation='relu')
        critic_fc_2 = Dense(64, activation='relu')
        actor_out = Dense(num_actions, activation='softmax')
        critic_out = Dense(1, activation='linear')
        actor = actor_out(actor_fc_2(actor_fc_1(model_in)))
        critic = critic_out(critic_fc_2(critic_fc_1(model_in)))
        return Model(inputs=model_in, outputs=[actor, critic])

    def predict(self, states: BatchedObservations) -> BatchedPredictions:
        prob_dists, baselines = self.model(states)
        prob_dists, baselines = prob_dists.numpy(), baselines.numpy()
        return prob_dists, np.squeeze(baselines)

    def train(self, batches: List[TrainingBatch]):
        for _ in range(self.config.update_epochs):
            shuffle(batches)
            for batch in batches:
                shuffled_batch = PPOModel.shuffle_batch(batch)
                self.train_step(shuffled_batch)
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

            # info: p_new / p_old = exp(log(p_new) - log(p_old)) -> no float errors for small p
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

    def save(self):
        self.model.save_weights("model/ppo_model.h5", overwrite=True)


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
        steps = self.config.n_envs * (self.config.steps_per_update + 1)
        obs_shape = [steps] + [d for d in self.config.obs_shape]
        self.prob_dist_cache = np.zeros((steps, self.config.num_actions), dtype=np.float32)
        self.baseline_cache = np.zeros((steps), dtype=np.float32)
        self.s0_cache = np.zeros(obs_shape, dtype=np.float32)
        self.actinon_cache = np.zeros((steps), dtype=np.int64)
        self.reward_cache = np.zeros((steps), dtype=np.float32)
        self.done_cache = np.zeros((steps), dtype=np.bool8)

    def append_step(self, preds: BatchedPredictions, exps: BatchedSarsExps):
        prob_dists, baselines = preds
        states_before, actions, rewards, _, dones = exps
        i, j = self.step * self.config.n_envs, (self.step + 1) * self.config.n_envs
        self.step += 1

        self.prob_dist_cache[i:j] = prob_dists
        self.baseline_cache[i:j] = baselines
        self.s0_cache[i:j] = states_before
        self.actinon_cache[i:j] = actions
        self.reward_cache[i:j] = rewards
        self.done_cache[i:j] = dones

        if self.step == self.config.steps_per_update + 1:
            batch = self._sample_batch()
            mini_batches = self._partition_minibatches(batch)
            self.consume_batch(mini_batches)

            # info: sampling requires the baseline of last s1
            #       -> cache 1 more step, assign it to beginning of the next batch
            self.prob_dist_cache[0:self.config.n_envs] = self.prob_dist_cache[i:j]
            self.baseline_cache[0:self.config.n_envs] = self.baseline_cache[i:j]
            self.s0_cache[0:self.config.n_envs] = self.s0_cache[i:j]
            self.actinon_cache[0:self.config.n_envs] = self.actinon_cache[i:j]
            self.reward_cache[0:self.config.n_envs] = self.reward_cache[i:j]
            self.done_cache[0:self.config.n_envs] = self.done_cache[i:j]
            self.step = 1

    def _sample_batch(self) -> TrainingBatch:
        batch_steps = self.config.n_envs * self.config.steps_per_update
        states = self.s0_cache[:batch_steps]
        actions = self.actinon_cache[:batch_steps]
        advantages = self._compute_advantages()
        returns = self.baseline_cache[:batch_steps] + advantages
        prob_dists_old = self.prob_dist_cache[:batch_steps]
        baselines_old = self.baseline_cache[:batch_steps]
        return states, actions, returns, advantages, prob_dists_old, baselines_old

    def _compute_advantages(self) -> BatchedAdvantages:
        batch_size = self.config.steps_per_update * self.config.n_envs
        baselines = self.baseline_cache[:batch_size]
        rewards = self.reward_cache[:batch_size]
        reward_discount = self.config.reward_discount
        gae_discount = self.config.gae_discount

        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(self.config.steps_per_update)):
            # info: slice [i:j] belongs to step t, slice [j:k] belongs to step t+1
            i, j, k = t * self.config.n_envs, (t + 1) * self.config.n_envs, (t + 2) * self.config.n_envs
            nextnonterminal = 1.0 - self.done_cache[j:k]
            nextvalues = self.baseline_cache[j:k]
            delta = rewards[i:j] + nextvalues * nextnonterminal * reward_discount - baselines[i:j]
            lastgaelam = delta + nextnonterminal * reward_discount * gae_discount * lastgaelam
            advantages[i:j] = lastgaelam
        return advantages

    def _partition_minibatches(self, big_batch: TrainingBatch) -> List[TrainingBatch]:
        num_examples = big_batch[0].shape[0]
        batch_size = self.config.batch_size
        states, actions, returns, advantages, prob_dists_old, baselines_old = big_batch

        mini_batches = []
        for b in range(num_examples // batch_size):
            i, j = batch_size * b, batch_size * (b+1)
            mini_batch = (states[i:j], actions[i:j], returns[i:j],
                advantages[i:j], prob_dists_old[i:j], baselines_old[i:j])
            mini_batches.append(mini_batch)
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

    def training(self):
        obs = self.encode_obs(self.vec_env.reset())

        for sim_step in tqdm(range(self.config.train_steps)):
            predictions = self.model(obs)
            actions = self.sample_actions(predictions)
            next_obs, rewards, dones, _ = self.vec_env.step(actions)
            next_obs = self.encode_obs(next_obs)
            sars_exps = (obs, actions, rewards, next_obs, dones)
            self.exps_consumer(predictions, sars_exps)
            self.log_timestep(sim_step, rewards, dones)
            obs = next_obs


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
            self.log_episode(step, avg_steps, avg_rewards)
            self.ep_steps[envs_in_final_state] = 0
            self.ep_rewards[envs_in_final_state] = 0.0


def train_cartpole():
    config = PPOTrainingSettings([4], 2)
    make_env = lambda: gym.make("CartPole-v1")
    vec_env = gym.vector.SyncVectorEnv([make_env for _ in range(config.n_envs)])

    tb_logger = TensorboardLogging()
    model = PPOModel(config, tb_logger.log_training_loss, tb_logger.flush_losses)
    exp_buffer = PPOVecEnvRolloutBuffer(config, model.train)
    episode_logger = VecEnvEpisodeStats(config.n_envs, tb_logger.log_episode)

    sample_actions = lambda pred: \
        [int(np.random.choice(config.num_actions, 1, p=prob_dist)) for prob_dist in pred[0]]
    encode_obs = lambda obs: obs / np.array([[4.8, 10.0, 0.42, 10.0]])
    session = VecEnvTrainingSession(
        config, vec_env, model.predict, encode_obs, sample_actions,
        exp_buffer.append_step, episode_logger.log_step)

    session.training()
    model.save()


if __name__ == '__main__':
    train_cartpole()
