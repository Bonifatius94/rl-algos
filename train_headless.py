import os
import gym

from algos.dreamer.config import DreamerSettings
from algos.dreamer.env import DreamerEnvWrapper
from algos.dreamer.dataset import sample_trajectory, generate_dataset
from algos.dreamer.logging import record_episode


def train_headless(num_epochs: int, num_trajs: int, train_steps_per_epoch: int):
    VIDEOS_ROOTDIR = "./videos"
    if os.path.exists(VIDEOS_ROOTDIR):
        os.system(f"rm -rf {VIDEOS_ROOTDIR}")
    os.mkdir(VIDEOS_ROOTDIR)

    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [32, 32, 3], [32, 32], [512], [64])
    env = DreamerEnvWrapper(orig_env, settings)

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

        video_file = os.path.join(VIDEOS_ROOTDIR, f"ep_{ep+1}.avi")
        record_episode(env, video_file)


if __name__ == "__main__":
    train_headless(1000000, 10, 50)
