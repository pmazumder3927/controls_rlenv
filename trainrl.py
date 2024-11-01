from multiprocessing import Lock, Manager
import os
import gymnasium as gym
from sbx import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from trajectoryenv import TrajectoryEnv
from wandb.integration.sb3 import WandbCallback
import wandb
import random
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th
from gymnasium import spaces

# Custom callback for logging data at the end of an episode


class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self):
        # Log only if done (end of episode)
        if self.locals['dones'].any():
            # Log lataccel history
            lataccel_image_path = "lataccel_history.png"
            if os.path.exists(lataccel_image_path):
                wandb.log(
                    {"lataccel_history": wandb.Image(lataccel_image_path)})

        return True

# Function to create environments for SubprocVecEnv


def make_env(rank, total_processes, file_queue, lock, eval_env=False):
    def _init():
        # # Lock to synchronize file access among processes
        # if eval_env:
        #     return TrajectoryEnv(file_number=26)
        # with lock:
        #     if not file_queue.empty():
        #         file_idx = file_queue.get()
        #     else:
        #         # Refill or end if no more files
        #         file_idx = random.choice(file_indices)  # Optional fallback
        env = TrajectoryEnv()
        return env

    return _init


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume input shape is (batch_size, 50, 5) and reshape it as (batch_size, 1, 50, 5)
        n_input_channels = 1  # Since we're treating it as a single channel
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(8, 2), stride=(
                4, 1), padding=0),  # Adjust kernel size and stride
            nn.ReLU(),
            # Further adjust kernel size and stride
            nn.Conv2d(32, 64, kernel_size=(4, 2), stride=(2, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[
                                        None, None]).float()  # Add channels dimension
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Reshape the input tensor to add a channels dimension (1 in this case)
        observations = observations.unsqueeze(1)  # Adds the channel dimension
        return self.linear(self.cnn(observations))


policy_kwargs = dict(features_extractor_class=CustomCNN,
                     features_extractor_kwargs=dict(features_dim=32), net_arch=[32, 32])


if __name__ == "__main__":
    # Initialize wandb bbin the main thread
    wandb.require("core")
    wandb.init(project="rl", sync_tensorboard=True, monitor_gym=True)
    # Global manager and lock for shared state
    manager = Manager()
    file_queue = manager.Queue()
    lock = Lock()

    # Initialize the queue with all file indices
    # Assuming 20001 files, from 00000.csv to 20000.csv
    file_indices = list(range(20000))
    random.shuffle(file_indices)  # Shuffle to randomize the order
    for idx in file_indices:
        file_queue.put(idx)

    n_envs = 32
    env = SubprocVecEnv([make_env(i, n_envs, file_queue, lock)
                        for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # env = TrajectoryEnv(file_number=0)
    # Params:
    # batch_size: 512
    # n_steps: 1024
    # gamma: 0.999
    # learning_rate: 3.689755741177914e-05jj
    # ent_coef: 6.112296542666746e-07
    # clip_range: 0.3
    # n_epochs: 1
    # gae_lambda: 0.92
    # max_grad_norm: 0.5
    # vf_coef: 0.1573564255774144
    # net_arch: small
    # activation_fn: tanh

    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log="./ppo_tensorboard/", gae_lambda=0.92, batch_size=512, n_steps=1024, policy_kwargs=policy_kwargs)

    # {'gamma': 0.999, 'learning_rate': 0.19480554242169765, 'batch_size': 1024, 'buffer_size': 100000, 'learning_starts': 1000, 'train_freq': 4, 'tau': 0.01, 'log_std_init': -1.6124611345225233, 'net_arch': 'medium'}
    # model = DDPG("MlpPolicy", env, verbose=1)
    # Set up evaluation and checkpoints
    eval_env = SubprocVecEnv(
        [make_env(i, 1, file_queue, lock, eval_env=True) for i in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True,
                            norm_reward=True, clip_obs=10)
    # eval_env = TrajectoryEnv(file_number=2)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=5000,
                                 deterministic=True, render=True)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',
                                             name_prefix='ppo_model')

    # Include the custom callback for wandb logging
    wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path='./models/',
                                   verbose=2)

    # Start training
    model.learn(total_timesteps=10000000, callback=[
                eval_callback, checkpoint_callback, wandb_callback])

    env.close()
    wandb.finish()  # Finish the wandb run
