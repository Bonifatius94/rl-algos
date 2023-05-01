import gym

from algos.dreamer.config import DreamerSettings
from algos.dreamer.env import DreamerEnvWrapper, play_episode
from algos.dreamer.dataset import sample_trajectory, generate_dataset
from algos.dreamer.display import DreamerDebugDisplay
from algos.dreamer.model import DreamerModel


def train_interactive(num_epochs: int, num_trajs: int, train_steps_per_epoch: int):
    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [32, 32, 3], [32, 32], [512], [64])
    model = DreamerModel(settings)
    env = DreamerEnvWrapper(orig_env, settings, model=model)

    ui_env = DreamerEnvWrapper(orig_env, settings, model=model, debug=True, debug_scaling=8)
    display = DreamerDebugDisplay(settings.obs_dims[1], settings.obs_dims[0], 8)
    ui_env.render_output = display.next_frame

    actor = lambda x: env.action_space.sample()
    trajs = [sample_trajectory(env, actor) for _ in range(num_trajs)]
    dataset = generate_dataset(trajs)
    dataset = dataset.batch(32).repeat().shuffle(100)
    batch_iter = iter(dataset)

    for ep in range(num_epochs):
        print(f"starting epoch {ep+1}")

        for _ in range(train_steps_per_epoch):
            s1, z0, h0, a0, r1, t1 = batch_iter.next()
            env.model.train(s1, z0, h0, a0, r1, t1)

        play_episode(ui_env, render=True, max_steps=100)


if __name__ == "__main__":
    train_interactive(1000, 10, 50)
