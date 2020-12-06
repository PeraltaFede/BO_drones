from warnings import catch_warnings
from warnings import simplefilter

import numpy as np
from shapely.geometry import Polygon
from sklearn.metrics import mean_squared_error as mse
from skopt.acquisition import gaussian_pi
from skopt.learning.gaussian_process import gpr, kernels

from bin.v2.Comparison.grid_based_sweep_coverage_path_planner import planning, SweepSearcher

try:
    from Utils.acquisition_functions import gaussian_sei, maxvalue_entropy_search
except ModuleNotFoundError:
    from bin.Utils.acquisition_functions import gaussian_sei, maxvalue_entropy_search


class Coordinator(object):
    def __init__(self, agents, map_data, main_sensor='None', acq="RU"):
        self.agents = agents
        self.map_data = map_data
        self.acquisition = 'gaussian_sei'
        self.gp = gpr.GaussianProcessRegressor(kernel=kernels.RBF(100), alpha=1e-7)
        # self.gp = gpr.GaussianProcessRegressor(kernel=kernels.Matern(150, nu=3.5), alpha=1e-7)
        # self.gp = gpr.GaussianProcessRegressor(kernel=kernels.RationalQuadratic(150, 0.1), alpha=1e-7)
        #         self.gp = gpr.GaussianProcessRegressor(normalize_y=True, kernel=20 * kernels.RBF(150), alpha=1e-7)
        self.data = [np.array([[], []]), np.array([])]

        self.acq_method = acq

        self.all_vector_pos = np.mgrid[0:self.map_data.shape[1]:1, 0:self.map_data.shape[0]:1].reshape(2, -1).T

        self.vector_pos = np.fliplr(np.asarray(np.where(self.map_data == 0)).reshape(2, -1).T)

        self.main_sensor = main_sensor
        _x, _y = self.obtain_shapely_polygon().exterior.coords.xy
        self.fpx, self.fpy = self.obtain_points(_x, _y)

        # dists = []
        # for i in range(len(self.fpx) - 1):
        #     dists.append(np.linalg.norm(
        #         np.subtract(np.array([self.fpx[i], self.fpy[i]]), np.array([self.fpx[i + 1], self.fpy[i + 1]]))))
        #
        import matplotlib.pyplot as plt
        # plt.Figure()
        # plt.hist(dists)
        # print(np.mean(dists))
        # print(np.std(dists))
        plt.plot(self.fpx, self.fpy, '-*')
        # plt.show(block=True)

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

    def obtain_shapely_polygon(self):
        reg = Polygon(
            [(0, 0), (0, np.size(self.map_data, 0)), (np.size(self.map_data, 1), np.size(self.map_data, 0)),
             (np.size(self.map_data, 1), 0)])
        return reg

    def obtain_points(self, x, y):
        if self.acq_method == "LD":
            bx, by = planning(x.tolist(), y.tolist(), 250 / 3, moving_direction=SweepSearcher.MovingDirection.LEFT,
                              sweeping_direction=SweepSearcher.SweepDirection.DOWN)
        if self.acq_method == "LU":
            bx, by = planning(x.tolist(), y.tolist(), 250 / 3, moving_direction=SweepSearcher.MovingDirection.LEFT,
                              sweeping_direction=SweepSearcher.SweepDirection.UP)
        if self.acq_method == "RD":
            bx, by = planning(x.tolist(), y.tolist(), 250 / 3, moving_direction=SweepSearcher.MovingDirection.RIGHT,
                              sweeping_direction=SweepSearcher.SweepDirection.DOWN)
        if self.acq_method == "RU":
            bx, by = planning(x.tolist(), y.tolist(), 250 / 3, moving_direction=SweepSearcher.MovingDirection.RIGHT,
                              sweeping_direction=SweepSearcher.SweepDirection.UP)
        px = []
        py = []
        for ipx, ipy in zip(bx, by):
            iipx = np.round(ipx).astype(np.int)
            iipy = np.round(ipy).astype(np.int)

            if self.map_data[iipy, iipx] == 0:
                px.append(iipx)
                py.append(iipy)
        # px = [px[i] for i in range(len(px)) if i % 3 == 0]
        # py = [py[i] for i in range(len(py)) if i % 3 == 0]
        return px, py
