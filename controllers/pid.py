from . import BaseController
import numpy as np
import torch
import torch.nn as nn
import os
import time


class Controller(BaseController):
    """
    A simple PID controller
    """

    def __init__(self, params=None):
        if params is None:
            params = [0.3, 0.05, -0.1, 0]
        self.p = params[0]
        self.i = params[1]
        self.d = params[2]
        # does nothing for now
        self.bias = params[3]
        self.error_integral = 0
        self.prev_error = 0
        # self.mlp = MLPFeedForward(
        #     model_path='./mlp_model.pth', scaler_path='./scaler.pkl')
        self.output = np.linspace(-2, 2, 600)
        self.step = 0
        self.last_action = 0
        self.last_state = None
        self.last_target_lataccel = None
        # create a unique file name
        # self.data_file_name = f'./fake_data/data_{time.time()}.csv'
        # if os.path.exists(self.data_file_name):
        #     self.data_file = open(self.data_file_name, 'a')
        # else:
        #     self.data_file = open(self.data_file_name, 'w')
        #     self.data_file.write(
        #         't,vEgo,aEgo,roll,targetLateralAcceleration,steerCommand\n')

    def update_params(self, params):
        self.p = params[0]
        self.i = params[1]
        self.d = params[2]
        self.bias = params[3]

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # breakpoint
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        # simple pid
        pid = self.p * error + self.i * self.error_integral + \
            self.d * error_diff
        return pid

        # # log last state and current lat accel
        # if self.last_state is not None:
        #     self.data_file.write(
        #         f'{self.step-1},{self.last_state.v_ego},{self.last_state.a_ego},{self.last_state.roll_lataccel},{current_lataccel},{self.output[self.step-1]}\n')
        # self.last_state = state
        # self.last_target_lataccel = target_lataccel
        # self.step += 1
        # return self.output[self.step-1]
