from warnings import catch_warnings
from warnings import simplefilter

import numpy as np
from sklearn.metrics import mean_squared_error as mse
from skopt.learning.gaussian_process import gpr, kernels
from skopt.acquisition import gaussian_lcb as ge

try:
    from Utils.acquisition_functions import gaussian_ei
except ModuleNotFoundError:
    from bin.Utils.acquisition_functions import gaussian_ei
from scipy.optimize import minimize


class Coordinator(object):
    def __init__(self, agents, map_data, main_sensor='None'):
        self.agents = agents
        self.map_data = map_data
        self.acquisition = 'gaussian_ei'
        self.gp = gpr.GaussianProcessRegressor(normalize_y=True, kernel=20 * kernels.RBF(150), alpha=1e-7)

        self.data = [np.array([[], []]), np.array([])]

        self.all_vector_pos = np.mgrid[0:self.map_data.shape[1]:1, 0:self.map_data.shape[0]:1].reshape(2, -1).T

        self.vector_pos = np.fliplr(np.asarray(np.where(self.map_data == 0)).reshape(2, -1).T)

        self.main_sensor = main_sensor

    def initialize_data_gpr(self, data):
        self.data = [np.array(data[0][0]).reshape(-1, 2), np.array([data[0][1]])]
        if len(data) > 1:
            for nd in data[1:]:
                self.add_data(nd)
        self.fit_data()

    def fit_data(self):
        self.gp.fit(self.data[0], self.data[1])

    def surrogate(self, _x=None, return_std=False, return_sensor=False):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            if _x is None:
                _x = self.all_vector_pos
            if not return_sensor:
                return self.gp.predict(_x, return_std)
            else:
                if return_std:
                    mu, std = self.gp.predict(_x, return_std)
                    return mu, std, self.main_sensor
                else:
                    return self.gp.predict(_x, False), self.main_sensor

    def add_data(self, new_data):
        self.data[0] = np.append(self.data[0], new_data[0]).reshape(-1, 2)
        self.data[1] = np.append(self.data[1], new_data[1])

    def propose_location(self, n_restarts=25):
        dim = self.data[0].shape[1]
        min_val = 1
        min_x = None

        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -gaussian_ei(X.reshape(-1, dim), self.gp, np.max(self.data[1]), xi=0.01)

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform([0, 0], [999, 1499], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=[[0, 999], [0, 1499]], method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x
                print(min_x)
                print(min_val)
        return np.round(min_x.reshape(-1, 1).T).astype(np.int)[0]

    def generate_new_goal(self, method=None):
        all_acq = gaussian_ei(self.vector_pos, self.gp, np.max(self.data[1]))
        new_pos = self.vector_pos[np.where(all_acq == np.nanmax(all_acq))][0]
        new_pos = np.append(new_pos, 0)

        return new_pos

    def get_mse(self, y_true):
        nan = np.isnan(y_true)
        return mse(y_true[~nan], self.surrogate()[~nan])

    def get_acq(self):
        return gaussian_ei(self.all_vector_pos, self.gp, np.max(self.data[1])), self.main_sensor
        # return ge(self.all_vector_pos, self.gp, 3), self.main_sensor
