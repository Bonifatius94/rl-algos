import gc
from typing import Protocol, Any, Callable

import gym
from tqdm import tqdm

from algos.dreamer.config import DreamerTrainSettings
from algos.dreamer.env import DreamerEnvWrapper, DreamVecEnv
from algos.dreamer.dataset import sample_trajectory, concat_datasets


class TrainableAgent(Protocol):
    def act(self, obs: Any) -> Any:
        raise NotImplementedError()

    def train(self, env: gym.vector.VectorEnv, steps: int):
        raise NotImplementedError()


def train(
        config: DreamerTrainSettings,
        env: DreamerEnvWrapper,
        agent: TrainableAgent,
        on_end_of_episode: Callable[[int], None] = lambda ep: None):
    rand_actor = lambda x: env.action_space.sample()

    for ep in range(config.epochs):
        print(f"start of episode {ep+1}")

        world_actor = rand_actor if ep == 0 else agent.act
        world_actor = agent.act

        print("collect trajectories")
        trajs = []
        for _ in tqdm(range(config.num_world_trajs)):
            trajs.append(sample_trajectory(env, world_actor))
        dataset = concat_datasets(trajs)
        shuffled_world_data = dataset.shuffle(100).batch(config.batch_size)

        # TODO: add loss logging to world model training
        print("update world model")
        for _ in tqdm(range(config.world_epochs)):
            for s1, z0, h0, a0, r1, t1 in iter(shuffled_world_data):
                env.model.train(s1, z0, h0, a0, r1, t1)

        initial_states = shuffled_world_data.unbatch().repeat().map(lambda s1, z0, h0, a0, r1, t1: (s1))
        initial_states_iter = iter(initial_states)

        agent_env = DreamVecEnv(
            config.n_envs, env.observation_space,
            env.action_space, env.model, initial_states_iter.next)

        print("update agent")
        agent.train(agent_env, config.agent_timesteps)

        on_end_of_episode(ep)
        gc.collect()
