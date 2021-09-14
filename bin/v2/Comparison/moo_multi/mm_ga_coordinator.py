import json
from sys import path

import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from skopt.learning.gaussian_process import kernels

from src.gpr import gprnew


class Coordinator(object):
    def __init__(self, map_data, sensors: set, k_names=None, d=1.0, no_drones=2):
        if k_names is None:
            k_names = ["RBF"] * len(sensors)
        self.map_data = map_data
        self.k_names = k_names  # "RBF" Matern" "RQ"
        self.sensors = sensors
        self.gps = dict()
        self.train_inputs = [np.array([[], []])]
        self.train_targets = dict()
        self.proportion = d

        self.mus = dict()
        self.stds = dict()
        self.has_calculated = dict()

        for sensor, kernel in zip(sensors, k_names):
            if kernel == "RBF":  # "RBF" Matern" "RQ"
                self.gps[sensor] = gprnew.GaussianProcessRegressor(kernel=kernels.RBF(100), alpha=1e-7)
                self.train_targets[sensor] = np.array([])
                self.mus[sensor] = np.array([])
                self.stds[sensor] = np.array([])
                self.has_calculated[sensor] = False

        self.all_vector_pos = np.mgrid[0:self.map_data.shape[1]:1, 0:self.map_data.shape[0]:1].reshape(2, -1).T
        self.vector_pos = np.fliplr(np.asarray(np.where(self.map_data == 0)).reshape(2, -1).T)

        self.splitted_goals = []
        self.nans = None
        self.fpx_, self.fpy_ = self.obtain_points()
        self.fpx_ = list(self.fpx_)
        self.fpy_ = list(self.fpy_)
        self.fpx = [self.fpx_[int(i * (60 / no_drones)):int(i * (60 / no_drones) + 60 / no_drones)] for i in
                    range(no_drones)]
        self.fpy = [self.fpy_[int(i * (60 / no_drones)):int(i * (60 / no_drones) + 60 / no_drones)] for i in
                    range(no_drones)]

        # fpx = [0 1 2 3]
        # 2 drones: 30 30, 3: 20 20 20, 4: 15 15 15 15
        self.current_goals = [[] for _ in range(no_drones)]
        for i in range(no_drones):
            self.current_goals[i] = np.array([self.fpx[i].pop(0), self.fpy[i].pop(0)])
        # print(self.current_goals)

    def initialize_data_gpr(self, data: list):
        self.train_inputs = np.array(data[0]["pos"]).reshape(-1, 2)
        for key in data[0].keys():
            if key != "pos":
                self.train_targets[key] = np.array([data[0][key]])
        if len(data) > 1:
            for nd in data[1:]:
                self.add_data(nd)
        self.fit_data()

    def add_data(self, new_data):
        self.train_inputs = np.append(self.train_inputs, new_data["pos"]).reshape(-1, 2)
        for key in new_data.keys():
            if key != "pos":
                self.train_targets[key] = np.append(self.train_targets[key], new_data[key])

    def fit_data(self):
        for key in self.sensors:
            self.gps[key].fit(self.train_inputs, self.train_targets[key])
            self.has_calculated[key] = False

    def surrogate(self, _x=None, return_std=False, keys=None):
        if keys is None:
            keys = self.sensors
        if _x is None:
            _x = self.vector_pos

        for key in keys:
            if not self.has_calculated[key]:
                mu, std = self.gps[key].predict(_x, True)
                self.mus[key] = mu
                self.stds[key] = std
                self.has_calculated[key] = True
        if return_std:
            return [(self.mus[key], self.stds[key]) for key in keys]
        else:
            return [self.mus[key] for key in keys]

    def generate_new_goal(self, pose=np.zeros((1, 3)), agt_id=0):
        if len(self.current_goals[agt_id]) == 0:
            self.current_goals[agt_id] = np.array([self.fpx[agt_id].pop(0), self.fpy[agt_id].pop(0)]).T
        beacons_splitted = []
        vect_dist = np.subtract(self.current_goals[agt_id], pose[:2])
        ang = np.arctan2(vect_dist[1], vect_dist[0])
        d = np.exp(np.min([self.gps[key].kernel_.theta[0] for key in list(self.sensors)])) * self.proportion
        for di in np.arange(0, np.linalg.norm(vect_dist), d)[1:]:
            mini_goal = np.array([di * np.cos(ang) + pose[0], di * np.sin(ang) + pose[1]]).astype(np.int)
            if self.map_data[mini_goal[1], mini_goal[0]] == 0:
                beacons_splitted.append(mini_goal)
        beacons_splitted.append(self.current_goals[agt_id])
        new_pos = np.array(beacons_splitted[0])
        new_pos = np.append(new_pos, 0)
        if len(beacons_splitted) == 1:
            self.current_goals[agt_id] = []
        return new_pos

    def get_mse(self, y_true, key=None):
        if key is None:
            key = list(self.sensors)[0]
        if self.nans is None:
            self.nans = np.isnan(y_true)
        return mse(y_true[~self.nans], self.surrogate(keys=[key])[0])

    def get_score(self, y_true, key=None):
        if key is None:
            key = list(self.sensors)[0]
        if self.nans is None:
            self.nans = np.isnan(y_true)
        return r2_score(y_true[~self.nans], self.surrogate(keys=[key])[0])

    def obtain_points(self):
        with open(path[-1] + "/data/Map/Ypacarai/beacons_normalized.json") as f:
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
        beacons_splitted = np.round(np.array(beacons[perm_matrix])).astype(np.int)
        return beacons_splitted[:, 0], beacons_splitted[:, 1]
