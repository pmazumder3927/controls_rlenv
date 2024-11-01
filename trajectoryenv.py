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


class TrajectoryEnv(gymnasium.Env):
    def __init__(self, future_trajectory_length=49, file_number=0):
        super(TrajectoryEnv, self).__init__()

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        )
        self.action_space_min = np.array(
            [0.3 - 10, 0.05 - 1, -0.1 - 10, 0 - 1])
        self.action_space_max = np.array(
            [0.3 + 10, 0.05 + 1, -0.1 + 10, 0 + 1])

        self.feature_scales = [0.02337214, 0.05447264,
                               0.31470417, 0.11180328, 0.11180328]
        self.feature_mins = [0,  7.00576470e-01,
                             5.00282936e-01,  5.59016393e-01, 5.59016393e-01]

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(50, 5),
            dtype=np.float32
        )

        self.controller = Controller()

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
        file_number = np.random.randint(0, 19999)
        self.sim = generate_rollout(
            f'./data/{file_number:05d}.csv', './models/tinyphysics.onnx', debug=False)
        self.sim_state = self.sim.pre_step()
        target_lataccel, current_lataccel, state, futureplan = self.sim_state
        self.state = self.convert_sim_state_to_obs(
            target_lataccel, current_lataccel, state, futureplan)
        self.lataccel_history.clear()
        self.target_lataccel_history.clear()
        self.params_history.clear()
        self.output_history.clear()

        # Initialize the controller with default action (e.g., zeros)
        default_action = np.zeros(self.action_space.shape)
        self.controller.update_params(default_action)

        # Simulate first 100 steps internally
        for _ in range(100):
            # Update the controller and simulate one step
            controller_output = self.controller.update(
                target_lataccel, current_lataccel, state, futureplan)
            cost = self.sim.step(controller_output)

            # Advance to the next state
            self.sim_state = self.sim.pre_step()
            target_lataccel, current_lataccel, state, futureplan = self.sim_state

            # Optionally, you can collect data here if needed for internal purposes
            # but do not expose it to the agent

        # After simulating 50 steps, update the state to be returned to the agent
        self.state = self.convert_sim_state_to_obs(
            target_lataccel, current_lataccel, state, futureplan)

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

    def render(self, mode='rgb_array'):
        return self.plot_and_log_lataccel_history()

    def _compute_reward(self, cost, control_input, target_lataccel, current_lataccel):
        # Instantaneous error
        error = current_lataccel - target_lataccel

        # Penalize squared error
        error_penalty = error ** 2

        # Compute derivative of error
        if hasattr(self, 'previous_error'):
            error_derivative = error - self.previous_error
        else:
            error_derivative = 0
        self.previous_error = error

        # Penalize squared derivative of error
        error_derivative_penalty = error_derivative ** 2
        # Penalize magnitude of control input (to minimize control effort)
        control_effort_penalty = np.sum(control_input ** 2)

        # Penalize rate of change in control input
        if hasattr(self, 'previous_control_input'):
            delta_u = control_input - self.previous_control_input
        else:
            delta_u = np.zeros_like(control_input)
        self.previous_control_input = control_input

        control_smoothness_penalty = np.sum(delta_u ** 2)
        # Penalize lag (positive error derivative when setpoint is increasing)
        setpoint_derivative = target_lataccel - self.previous_target_lataccel if hasattr(self, 'previous_target_lataccel') else 0
        self.previous_target_lataccel = target_lataccel

        lag_penalty = 0
        if setpoint_derivative != 0:
            # If setpoint is increasing and error is positive (lagging), penalize
            if setpoint_derivative > 0 and error > 0:
                lag_penalty = error ** 2
            # If setpoint is decreasing and error is negative (lagging), penalize
            elif setpoint_derivative < 0 and error < 0:
                lag_penalty = error ** 2
            # Weighting factors for each penalty
        w_error = 1.0
        w_error_derivative = 0.5
        w_control_effort = 0.01
        w_control_smoothness = 0.1
        w_lag = 0.5

        # Total penalty
        total_penalty = (
            w_error * error_penalty +
            w_error_derivative * error_derivative_penalty +
            w_control_effort * control_effort_penalty +
            w_control_smoothness * control_smoothness_penalty +
            w_lag * lag_penalty
        )

        # Reward is negative total penalty
        reward = -total_penalty
        return reward

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]
