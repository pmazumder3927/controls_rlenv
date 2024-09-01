from multiprocessing import Lock, Manager
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from trajectoryenv import TrajectoryEnv
from wandb.integration.sb3 import WandbCallback
import wandb
import random
from torch import nn
from sbx import DroQ
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
        # Lock to synchronize file access among processes
        if eval_env:
            return TrajectoryEnv(file_number=26)
        with lock:
            if not file_queue.empty():
                file_idx = file_queue.get()
            else:
                # Refill or end if no more files
                file_idx = random.choice(file_indices)  # Optional fallback

        env = TrajectoryEnv(file_number=file_idx)
        return env

    return _init


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
    # env = SubprocVecEnv([make_env(i, n_envs, file_queue, lock)
    #                     for i in range(n_envs)])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    env = TrajectoryEnv(file_number=0)
    # Params:
    # batch_size: 512
    # n_steps: 1024
    # gamma: 0.999
    # learning_rate: 3.689755741177914e-05
    # ent_coef: 6.112296542666746e-07
    # clip_range: 0.3
    # n_epochs: 1
    # gae_lambda: 0.92
    # max_grad_norm: 0.5
    # vf_coef: 0.1573564255774144
    # net_arch: small
    # activation_fn: tanh

    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log="./ppo_tensorboard/", policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.ReLU), learning_rate=3.689755741177914e-05, ent_coef=6.112296542666746e-07, clip_range=0.3, n_epochs=1, gae_lambda=0.92, max_grad_norm=0.5, vf_coef=0.1573564255774144, batch_size=512, n_steps=1024, gamma=0.999, use_sde=True)

    # Set up evaluation and checkpoints
    # eval_env = SubprocVecEnv(
    #     [make_env(i, 1, file_queue, lock, eval_env=True) for i in range(1)])
    # eval_env = VecNormalize(eval_env, norm_obs=True,
    #                         norm_reward=True, clip_obs=10.)
    eval_env = TrajectoryEnv(file_number=2)
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
