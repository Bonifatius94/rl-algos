import gc
from typing import Protocol, Any, Callable

import gym
from tqdm import tqdm

from algos.dreamer.config import DreamerTrainSettings
from algos.dreamer.env import DreamerEnvWrapper, DreamVecEnv
from algos.dreamer.dataset import sample_trajectory, concat_datasets
from algos.dreamer.logging import DreamerTensorboardLogger


class TrainableAgent(Protocol):
    def act(self, obs: Any) -> Any:
        raise NotImplementedError()

    def train(self, env: gym.vector.VectorEnv, steps: int):
        raise NotImplementedError()

    def save(self, directory: str):
        raise NotImplementedError()


def train(
        config: DreamerTrainSettings,
        env: DreamerEnvWrapper,
        agent: TrainableAgent,
        tb_logger: DreamerTensorboardLogger,
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

        print("update world model")
        for i in tqdm(range(config.world_epochs)):
            for s0_init, a0_init, s1, a0, r1, t1 in iter(shuffled_world_data):
                env.model.train(s0_init, a0_init, s1, a0, r1, t1)
            log_step = ep * config.world_epochs + i
            tb_logger.flush_losses(log_step)

        initial_states = shuffled_world_data.unbatch().repeat()\
            .map(lambda s0_init, a0_init, s1, a0, r1, t1: (s0_init, a0_init))
        initial_states_iter = iter(initial_states)

        agent_env = DreamVecEnv(
            config.n_envs, env.observation_space,
            env.action_space, env.model, initial_states_iter.next)

        print("update agent")
        agent.train(agent_env, config.agent_timesteps)

        if (ep + 1) % config.save_model_interval == 0:
            print("snapshot models")
            agent.save(f"model/agent_{ep}")
            env.model.save(f"model/dreamer_{ep}")

        on_end_of_episode(ep)
        gc.collect()

    agent.save(f"model/agent_final")
    env.model.save("model/dreamer_final")
