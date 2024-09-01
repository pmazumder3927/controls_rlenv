import pickle
import gymnasium
from gymnasium import spaces
import matplotlib
import numpy as np
from ghettophysics import generate_rollout
from ghettophysics import State, FuturePlan
from controllers.pid import Controller
import matplotlib.pyplot as plt
import torch
import time


def find_scaling_factor(min_cost, max_cost):
    scaling_factor = (max_cost - min_cost) / 10
    return scaling_factor


class TrajectoryEnv(gymnasium.Env):
    def __init__(self, future_trajectory_length=49, file_number=0):
        super(TrajectoryEnv, self).__init__()

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        )
        self.action_space_min = np.array([0, 0, -10, 0])
        self.action_space_max = np.array([10, 1, 10, 1])

        self.feature_scales = [0.02337214, 0.05447264,
                               0.31470417, 0.11180328, 0.11180328]
        self.feature_mins = [-9.62861519e-50,  7.00576470e-01,
                             5.00282936e-01,  5.59016393e-01, 5.59016393e-01]

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(50, 5),
            dtype=np.float32
        )

        self.controller = Controller()

        self.min_cost = 1000
        self.max_cost = 10000
        self.scaling_factor = find_scaling_factor(self.min_cost, self.max_cost)
        self.output_history = []

        self.state = None
        self.future_trajectory_length = future_trajectory_length
        self.sim = generate_rollout(
            f'./data/{file_number:05d}.csv', './models/tinyphysics.onnx', debug=False)
        self.params_history = []
        self.lataccel_history = []
        self.target_lataccel_history = []
        matplotlib.use('Agg')

    def convert_sim_state_to_obs(self, target_lataccel, current_lataccel, state, futureplan):
        observation_object = np.zeros((50, 5), dtype=np.float32)
        observation_object[0, :] = np.array(
            [state.v_ego, state.a_ego, state.roll_lataccel, target_lataccel, current_lataccel], dtype=np.float32)
        futureplan_padded = np.zeros((self.future_trajectory_length, 5))
        futureplan_array = np.array(futureplan)
        futureplan_array = futureplan_array[[2, 3, 1, 0], :]
        futureplan_padded[:futureplan_array.shape[1], :4] = futureplan_array.T
        if futureplan_array.shape[0] < self.future_trajectory_length:
            futureplan_padded[futureplan_array.shape[1]:, :] = -1
        observation_object[1:, :] = np.array(futureplan_padded)
        observation_object = (observation_object *
                              self.feature_scales) + self.feature_mins
        return observation_object.astype(np.float32)

    def convert_obs_to_sim_state(self, observation_object):
        observation_object = (observation_object -
                              self.feature_mins) / self.feature_scales
        state = State(
            v_ego=observation_object[0, 0], a_ego=observation_object[0, 1], roll_lataccel=observation_object[0, 2])
        futureplan = FuturePlan(v_ego=observation_object[1:, 0], a_ego=observation_object[1:, 1],
                                roll_lataccel=observation_object[1:, 2], lataccel=observation_object[1:, 3])
        target_lataccel = observation_object[0, 3]
        current_lataccel = observation_object[0, 4]
        return target_lataccel, current_lataccel, state, futureplan

    def reset(self, seed=None, options=None):
        self.sim.reset()
        self.sim_state = self.sim.pre_step()
        target_lataccel, current_lataccel, state, futureplan = self.sim_state
        self.state = self.convert_sim_state_to_obs(
            target_lataccel, current_lataccel, state, futureplan)
        self.lataccel_history.clear()
        self.target_lataccel_history.clear()
        self.params_history.clear()
        self.output_history.clear()
        self.min_cost = 1000
        self.max_cost = 10000
        self.scaling_factor = find_scaling_factor(self.min_cost, self.max_cost)
        return self.state, {}

    def step(self, action):
        # v_x0, alpha_0, K_alpha, K_p, K_d = action
        target_lataccel, current_lataccel, state, futureplan = self.sim_state
        # bring action to [0, 1]
        action = (action + 1) / 2
        action = action * (self.action_space_max -
                           self.action_space_min) + self.action_space_min
        self.controller.update_params(action)
        controller_output = self.controller.update(
            target_lataccel, current_lataccel, state, futureplan)
        cost = self.sim.step(controller_output)
        self.output_history.append(controller_output)

        reward = self._compute_reward(
            cost, controller_output, target_lataccel, current_lataccel)

        done = self.sim.is_done()
        truncated = False
        # self.params_history.append(controller_output)
        self.params_history.append(action)
        self.lataccel_history.append(current_lataccel)
        self.target_lataccel_history.append(target_lataccel)

        if done:
            cost = self.sim.compute_cost()

        if not done:
            self.sim_state = self.sim.pre_step()
            target_lataccel, current_lataccel, state, futureplan = self.sim_state
            self.state = self.convert_sim_state_to_obs(
                target_lataccel, current_lataccel, state, futureplan)
        return self.state, reward, done, truncated, {'action_history': self.output_history, 'params_history': self.params_history, 'lataccel_history': self.lataccel_history, 'target_lataccel_history': self.target_lataccel_history}

    def plot_and_log_lataccel_history(self):
        # This function is used to log data only when needed, and is triggered externally, returns rgb array of the plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.target_lataccel_history, label='Target Lataccel')
        plt.plot(self.lataccel_history, label='Current Lataccel')
        plt.legend()
        plt.title('Lataccel History')
        plt.xlabel('Time Steps')
        plt.ylabel('Lateral Acceleration')
        plt.savefig("lataccel_history.png")
        plt.close()
        return plt.gcf()

    def render(self):
        return self.plot_and_log_lataccel_history()

    def _compute_reward(self, cost, control_input, target_lataccel, current_lataccel):
        self.scaling_factor = find_scaling_factor(self.min_cost, self.max_cost)
        total_cost = cost[0]['total_cost']
        scaled_cost = -2 / (1 + np.exp(-total_cost / self.scaling_factor)) + 1

        # error = np.abs(target_lataccel - current_lataccel)
        # control_penalty = np.sum(np.square(control_input))
# reward = -scaled_cost - 0.01 * control_penalty - 0.1 * error
        reward = scaled_cost
        return reward

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]
