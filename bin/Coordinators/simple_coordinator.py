from warnings import catch_warnings
from warnings import simplefilter

import numpy as np
import skopt.acquisition as acq
from sklearn.metrics import mean_squared_error as mse
from skopt.learning.gaussian_process import gpr, kernels


class Coordinator(object):
    def __init__(self, agents, map_data, main_sensor='None'):
        self.agents = agents
        self.map_data = map_data
        self.acquisition = 'gaussian_lcb'
        self.gp = gpr.GaussianProcessRegressor(normalize_y=True, kernel=1000 * kernels.RBF(2))

        self.data = [np.array([[], []]), np.array([])]

        # self.vector_pos = np.mgrid[0:self.map_data.shape[1]:1, 0:self.map_data.shape[0]:1].reshape(2, -1).T
        self.vector_pos = np.asarray(np.where(self.map_data == 0)).reshape(2, -1).T

        self.main_sensor = main_sensor

    def initialize_data_gpr(self, data):
        self.data = [np.array([data[0]]).reshape(-1, 2), np.array([data[1]])]
        self.fit_data()

    def fit_data(self):
        self.gp.fit(self.data[0], self.data[1])

    def surrogate(self, _x=None, return_std=False, return_sensor=False):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            if _x is None:
                _x = self.vector_pos
                if not return_sensor:
                    return self.gp.predict(_x, return_std)
                else:
                    mu, std = self.gp.predict(_x, return_std)
                    return mu, std, self.main_sensor

    def add_data(self, new_data):
        self.data[0] = np.append(self.data[0], new_data[0]).reshape(-1, 2)
        self.data[1] = np.append(self.data[1], new_data[1])

    def generate_new_goal(self, method=None):
        # ?
        if method is None:
            method = self.acquisition
        new_pos = None
        all_acq = getattr(acq, method)(self.vector_pos, self.gp, return_grad=False)
        while new_pos is None:
            if "lcb" in method:
                new_pos = self.vector_pos[np.where(all_acq == np.min(all_acq))][0]
            else:
                new_pos = self.vector_pos[np.where(all_acq == np.max(all_acq))][0]
        new_pos = np.append(new_pos, 0)
        return new_pos

    def get_mse(self, y_true):
        return mse(y_true, self.surrogate(self.vector_pos))
