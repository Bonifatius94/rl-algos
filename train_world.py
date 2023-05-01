import os
import gym

from algos.dreamer.config import DreamerSettings
from algos.dreamer.env import DreamerEnvWrapper
from algos.dreamer.dataset import sample_trajectory
from algos.dreamer.logging import record_episode


def train(num_epochs: int, num_trajectories: int, debug_interval: int):
    VIDEOS_ROOTDIR = "./videos"
    if os.path.exists(VIDEOS_ROOTDIR):
        os.system(f"rm -rf {VIDEOS_ROOTDIR}")
    os.mkdir(VIDEOS_ROOTDIR)

    orig_env = gym.make("ALE/Pong-v5")
    settings = DreamerSettings([1], [32, 32, 3], [32, 32], [512], [64])
    env = DreamerEnvWrapper(orig_env, settings)

    actor = lambda x: env.action_space.sample()
    trajs = [sample_trajectory(env, actor) for _ in range(num_trajectories)]

    for ep in range(num_epochs):
        print(f"starting epoch {ep+1}")

        for traj in trajs:
            env.model.train(
                traj.states,traj.zs, traj.hs,
                traj.actions, traj.rewards, traj.terms)

        if (ep+1) % debug_interval == 0:
            video_file = os.path.join(VIDEOS_ROOTDIR, f"ep_{ep+1}.avi")
            # TODO: figure out why the recorded files are empty
            record_episode(env, video_file)

if __name__ == "__main__":
    train(1000000, 4, 1)
