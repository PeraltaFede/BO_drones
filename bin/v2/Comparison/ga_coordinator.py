import json
from warnings import catch_warnings
from warnings import simplefilter

import numpy as np
from sklearn.metrics import mean_squared_error as mse
from skopt.acquisition import gaussian_pi
from skopt.learning.gaussian_process import gpr, kernels

try:
    from Utils.acquisition_functions import gaussian_sei, maxvalue_entropy_search
except ModuleNotFoundError:
    from bin.Utils.acquisition_functions import gaussian_sei, maxvalue_entropy_search


class Coordinator(object):
    def __init__(self, agents, map_data, main_sensor='None'):
        self.agents = agents
        self.map_data = map_data
        self.acquisition = 'gaussian_sei'
        self.gp = gpr.GaussianProcessRegressor(kernel=kernels.RBF(100), alpha=1e-7)
        # self.gp = gpr.GaussianProcessRegressor(kernel=kernels.Matern(150, nu=3.5), alpha=1e-7)
        # self.gp = gpr.GaussianProcessRegressor(kernel=kernels.RationalQuadratic(150, 0.1), alpha=1e-7)
        #         self.gp = gpr.GaussianProcessRegressor(normalize_y=True, kernel=20 * kernels.RBF(150), alpha=1e-7)
        self.data = [np.array([[], []]), np.array([])]

        self.all_vector_pos = np.mgrid[0:self.map_data.shape[1]:1, 0:self.map_data.shape[0]:1].reshape(2, -1).T

        self.vector_pos = np.fliplr(np.asarray(np.where(self.map_data == 0)).reshape(2, -1).T)

        self.main_sensor = main_sensor
        self.fpx, self.fpy = self.obtain_points()
        self.fpx = list(self.fpx)
        self.fpy = list(self.fpy)

        print(len(self.fpx))
        # dists = []
        # for i in range(len(self.fpx) - 1):
        #     dists.append(np.linalg.norm(
        #         np.subtract(np.array([self.fpx[i], self.fpy[i]]), np.array([self.fpx[i + 1], self.fpy[i + 1]]))))
        #
        # import matplotlib.pyplot as plt
        # plt.Figure()
        # plt.hist(dists)
        # print(np.mean(dists))
        # print(np.std(dists))
        # plt.plot(self.fpx, self.fpy, '-*')

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

    def generate_new_goal(self, pose=np.zeros((1, 3))):

        new_pos = np.array([self.fpx.pop(0), self.fpy.pop(0)])
        new_pos = np.append(new_pos, 0)
        print('future pos is: ', new_pos)

        return new_pos

    def get_mse(self, y_true):
        nan = np.isnan(y_true)
        return mse(y_true[~nan], self.surrogate()[~nan])

    def get_acq(self, pose=np.zeros((1, 2)), acq_func="gaussian_sei"):
        if acq_func == "gaussian_sei":
            return gaussian_sei(self.all_vector_pos, self.gp, np.max(self.data[1]),
                                c_point=pose[0][:2]), self.main_sensor
        elif acq_func == "maxvalue_entropy_search":
            return maxvalue_entropy_search(self.all_vector_pos, self.gp, np.max(self.data[1]),
                                           c_point=pose[0][:2]), self.main_sensor
        elif acq_func == "gaussian_pi":
            return gaussian_pi(self.all_vector_pos, self.gp, np.max(self.data[1])), self.main_sensor
        # return ge(self.all_vector_pos, self.gp, 3), self.main_sensor

    @staticmethod
    def obtain_points():
        with open("E:/ETSI/Proyecto/data/Map/Ypacarai/beacons_normalized.json") as f:
            scale = 1407.6
            data = json.load(f)
            xpos = data['x1 ']['xRel'] * scale
            ypos = data['x1 ']['yRel'] * scale
            beacons = np.array([xpos, ypos])
            beacons = np.expand_dims(beacons, 0)
            for i in range(len(data) - 1):
                name = 'x' + str(i + 2) + ' '
                xpos = data[name]['xRel'] * scale
                ypos = data[name]['yRel'] * scale
                beacons = np.concatenate([beacons, [np.array([xpos, ypos])]])
        perm_matrix = [25, 59, 26, 7, 44, 2, 42, 55, 21, 30, 53, 22,
                       52, 11, 51, 20, 50, 18, 31, 8, 33, 10, 45, 54, 43, 48, 39,
                       4, 46, 37, 38, 1, 32, 3, 19, 49, 14, 58, 9, 41, 5, 24, 36,
                       12, 47, 6, 35, 17, 57, 23, 16, 29, 13, 56, 40, 0, 28, 27, 15, 34]

        perm_matrix = np.roll(perm_matrix, np.random.randint(0, 60))

        beacons = beacons[perm_matrix]
        beacons_splitted = [np.array(beacons[0, :])]

        for i in range(len(beacons) - 1):
            vect_dist = np.subtract(beacons[i+1, :], beacons[i, :])
            ang = np.arctan2(vect_dist[1], vect_dist[0])
            lx = np.arange(beacons[i, 0], beacons[i + 1, 0], 290 * np.cos(ang))[1:]
            ly = np.arange(beacons[i, 1], beacons[i + 1, 1], 290 * np.sin(ang))[1:]
            [beacons_splitted.append(np.array([dx, dy])) for dx, dy in zip(lx, ly)]
            beacons_splitted.append(np.array(beacons[i+1, :]))
        beacons_splitted = np.round(np.array(beacons_splitted)).astype(np.int)

        return beacons_splitted[:, 0], beacons_splitted[:, 1]
