# train.py

import os
import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
from multiprocessing import Lock, Manager
import random

from sbx import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

from trajectoryenv import TrajectoryEnv

import optuna
from stable_baselines3.common.evaluation import evaluate_policy

# For logging
from wandb.integration.sb3 import WandbCallback
import wandb

# Custom feature extractor for policy network


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN to process the observation space.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = 1  # Since observations are (50, 5), treated as single-channel image
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(8, 2), stride=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 2), stride=(2, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_input = th.as_tensor(
                observation_space.sample()[None, None]
            ).float()  # Add batch and channel dimensions
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Add channel dimension
        observations = observations.unsqueeze(1)
        return self.linear(self.cnn(observations))


# Custom callback for logging data at the end of an episode


class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)

    def _on_step(self):
        # Log only if done (end of episode)
        if self.locals.get("dones", [False])[0]:
            # Log lataccel history
            lataccel_image_path = "lataccel_history.png"
            if os.path.exists(lataccel_image_path):
                wandb.log({"lataccel_history": wandb.Image(lataccel_image_path)})

        return True


# Function to create environments for SubprocVecEnv
def make_env(rank, seed=0, eval_env=False):
    def _init():
        env = TrajectoryEnv()
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


# Hyperparameter optimization function
def optimize_agent(trial):
    """Defines the hyperparameter search space and returns the evaluation score."""
    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    gamma = trial.suggest_uniform("gamma", 0.95, 0.9999)
    tau = trial.suggest_uniform("tau", 0.005, 0.02)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-4, 1e-1)

    # Network architecture
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    net_arch_options = {
        "small": [256, 256],
        "medium": [400, 300],
    }
    policy_kwargs = dict(
        net_arch=net_arch_options[net_arch],
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # Create environment
    n_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
    )

    # Evaluate the model using cross-validation
    eval_env = SubprocVecEnv([make_env(i + n_envs, eval_env=True) for i in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        n_eval_episodes=5,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # Train the model
    model.learn(total_timesteps=50000, callback=eval_callback)

    # Evaluate the trained agent
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)

    env.close()
    eval_env.close()

    return mean_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Whether to perform hyperparameter optimization.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of hyperparameter trials."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="Total timesteps for training.",
    )
    args = parser.parse_args()

    # Initialize wandb in the main thread
    wandb.require("service")
    wandb.init(project="rl", sync_tensorboard=True, monitor_gym=True)

    if args.optimize:
        # Hyperparameter optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_agent, n_trials=args.n_trials, n_jobs=4)

        print("Best hyperparameters:", study.best_params)
        best_params = study.best_params

        # Save the best hyperparameters
        with open("best_hyperparameters.txt", "w") as f:
            f.write(str(best_params))
    else:
        # Load best hyperparameters if available
        if os.path.exists("best_hyperparameters.txt"):
            with open("best_hyperparameters.txt", "r") as f:
                best_params = eval(f.read())
        else:
            # Default hyperparameters
            best_params = {
                "learning_rate": 3e-4,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.005,
                "ent_coef": "auto",
            }

        # Network architecture
        net_arch_options = {
            "small": [256, 256],
            "medium": [400, 300],
        }
        net_arch = best_params.get("net_arch", "small")
        policy_kwargs = dict(
            net_arch=net_arch_options[net_arch],
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),
        )

        # Create environment
        n_envs = 8
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Create the model
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=best_params["learning_rate"],
            batch_size=best_params["batch_size"],
            gamma=best_params["gamma"],
            tau=best_params["tau"],
            ent_coef=best_params["ent_coef"],
            policy_kwargs=policy_kwargs,
            tensorboard_log="./sac_tensorboard/",
        )

        # Set up evaluation and checkpoints
        eval_env = SubprocVecEnv([make_env(i + n_envs, eval_env=True) for i in range(1)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./logs/best_model",
            log_path="./logs/",
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000, save_path="./models/", name_prefix="sac_model"
        )

        # Include the custom callback for wandb logging
        wandb_callback = WandbCallback(
            gradient_save_freq=100, model_save_path="./models/", verbose=2
        )

        # Start training
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback, wandb_callback],
        )

        env.close()
        wandb.finish()  # Finish the wandb run