import numpy as np
from shapely.geometry import Polygon
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from skopt.learning.gaussian_process import kernels

from bin.v2.Comparison.grid_based_sweep_coverage_path_planner import planning, SweepSearcher
from src.gpr import gprnew


class Coordinator(object):
    def __init__(self, map_data, sensors: set, k_names=None, d=1.0, no_drones=2, acq=0):
        if k_names is None:
            k_names = ["RBF"] * len(sensors)
        self.map_data = map_data
        self.k_names = k_names  # "RBF" Matern" "RQ"
        self.sensors = sensors
        self.gps = dict()
        self.train_inputs = [np.array([[], []])]
        self.train_targets = dict()
        self.proportion = d
        self.nans = None

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
        self.acq_method = acq % 4

        _x, _y = self.obtain_shapely_polygon().exterior.coords.xy
        self.fpx_, self.fpy_ = self.obtain_points(_x, _y)
        no_wps = len(self.fpx_)
        self.fpx_ = list(self.fpx_)
        self.fpy_ = list(self.fpy_)

        self.fpx = [self.fpx_[int(i * (no_wps / no_drones)):int(i * (no_wps / no_drones) + no_wps / no_drones)] for i in
                    range(no_drones)]
        self.fpy = [self.fpy_[int(i * (no_wps / no_drones)):int(i * (no_wps / no_drones) + no_wps / no_drones)] for i in
                    range(no_drones)]
        self.current_goals = [[] for _ in range(no_drones)]
        for i in range(no_drones):
            self.current_goals[i] = np.array([self.fpx[i].pop(0), self.fpy[i].pop(0)])

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
        self.current_goals[agt_id] = np.array([self.fpx[agt_id].pop(0), self.fpy[agt_id].pop(0)]).T
        new_pos = self.current_goals[agt_id]
        new_pos = np.append(new_pos, 0)
        print('future pos is: ', new_pos)

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

    def obtain_shapely_polygon(self):
        reg = Polygon(
            [(0, 0), (0, np.size(self.map_data, 0)), (np.size(self.map_data, 1), np.size(self.map_data, 0)),
             (np.size(self.map_data, 1), 0)])
        return reg

    def obtain_points(self, x, y):
        if self.acq_method == 0:
            bx, by = planning(x.tolist(), y.tolist(), 100, moving_direction=SweepSearcher.MovingDirection.LEFT,
                              sweeping_direction=SweepSearcher.SweepDirection.DOWN)
            print('1')
        elif self.acq_method == 1:
            bx, by = planning(x.tolist(), y.tolist(), 100, moving_direction=SweepSearcher.MovingDirection.LEFT,
                              sweeping_direction=SweepSearcher.SweepDirection.UP)
            print('2')
        elif self.acq_method == 2:
            bx, by = planning(x.tolist(), y.tolist(), 100, moving_direction=SweepSearcher.MovingDirection.RIGHT,
                              sweeping_direction=SweepSearcher.SweepDirection.DOWN)
            print('3')
        else:
            bx, by = planning(x.tolist(), y.tolist(), 100, moving_direction=SweepSearcher.MovingDirection.RIGHT,
                              sweeping_direction=SweepSearcher.SweepDirection.UP)
        px = []
        py = []
        for ipx, ipy in zip(bx, by):
            iipx = np.round(ipx).astype(np.int)
            iipy = np.round(ipy).astype(np.int)

            if self.map_data[iipy, iipx] == 0:
                px.append(iipx)
                py.append(iipy)
        return px, py
