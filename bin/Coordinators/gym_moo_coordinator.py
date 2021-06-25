import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from shapely.geometry import Polygon, Point
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from skopt.learning.gaussian_process import gpr, kernels

from bin.Utils.acquisition_functions import gaussian_sei, maxvalue_entropy_search, gaussian_pi, gaussian_ei, max_std
from bin.Utils.voronoi_regions import calc_voronoi


class MyProblem(Problem):

    def __init__(self, gps, train_targets, map_data):
        super().__init__(n_var=2,
                         n_obj=len(gps),
                         n_constr=1,
                         xl=np.array([0, 0]),
                         xu=np.array([999, 1499]))
        self.sensors = gps.keys()
        self.gps = gps
        self.map_data = map_data
        self.train_targets = train_targets

    def _evaluate(self, x, out, *args, **kwargs):
        # X = [x, y] vectores de posicion
        # f1 = gaussian_ei(X, self.gps["s2"], np.min(self.train_targets["s2"]))
        # f = {"s1": x[:, 0] ** 2 + x[:, 1] ** 2, "s2": (x[:, 0] - 1000) ** 2 + (x[:, 1]) ** 2}
        # out["F"] = np.column_stack([f[key] for key in self.sensors])

        out["F"] = np.column_stack([-gaussian_ei(x, self.gps[key],
                                                 np.min(self.train_targets[key])) for key in self.sensors])

        constraints = np.ones(np.shape(x)[0])

        for iid, point in enumerate(x):
            point = np.round(point).astype(np.int)
            if self.map_data[point[1], point[0]] == 0 and reg_poly.contains(Point(point)):
                constraints[iid] = -1
        out["G"] = np.column_stack([constraints])
        # out["G"] = np.column_stack([constraints for key in self.sensors])


class Coordinator(object):
    def __init__(self, map_data, sensors: set, k_names=None, acq="gaussian_ei", acq_mod="masked",
                 acq_fusion="decoupled", d=0.375):
        if k_names is None:
            k_names = ["RBF"] * len(sensors)
        self.map_data = map_data
        self.acquisition = acq  # 'gaussian_sei' 'gaussian_ei' 'maxvalue_entropy_search''gaussian_pi'
        self.acq_mod = acq_mod  # 'masked' 'split_path' 'truncated', 'normal',
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
                self.gps[sensor] = gpr.GaussianProcessRegressor(kernel=kernels.RBF(100), alpha=1e-7)
                # self.gps[sensor] = gprnew.GaussianProcessRegressor(kernel=kernels.RBF(100), alpha=1e-7, noise=0.01)
                self.train_targets[sensor] = np.array([])
                self.mus[sensor] = np.array([])
                self.stds[sensor] = np.array([])
                self.has_calculated[sensor] = False

        self.all_vector_pos = np.mgrid[0:self.map_data.shape[1]:1, 0:self.map_data.shape[0]:1].reshape(2, -1).T
        self.vector_pos = np.fliplr(np.asarray(np.where(self.map_data == 0)).reshape(2, -1).T)

        self.problem = MyProblem(gps=self.gps, train_targets=self.train_targets, map_data=map_data)
        self.algorithm = NSGA2(pop_size=50)

        self.splitted_goals = []
        self.nans = None

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
            # for key in keys:
            #     print(np.sum(np.subtract(self.gps[key].predict(_x)[~self.nans], self.mus[key][~self.nans])))
            #     print(np.sum(np.subtract(self.gps[key].predict(self.vector_pos), self.mus[key][~self.nans])))
            # return [(self.mus[key][~self.nans], self.stds[key][~self.nans]) for key in keys]
            return [(self.mus[key], self.stds[key]) for key in keys]
        else:
            return [self.mus[key] for key in keys]

    def generate_new_goal(self, pose=np.zeros((1, 3)), other_poses=np.zeros((1, 3))):
        global reg_poly

        if self.acq_mod == "split_path":
            if len(self.splitted_goals) > 0:
                new_pos = self.splitted_goals[0, :]
                self.splitted_goals = self.splitted_goals[1:, :]
                return np.append(new_pos, 0)

        _, reg = calc_voronoi(pose, other_poses, self.map_data)
        reg_poly = Polygon(reg)

        res = minimize(self.problem,
                       self.algorithm,
                       ("n_gen", 150),
                       verbose=False)
        prev_min_dist = 10000
        new_pos = pose[:2]
        for point in res.X:
            curr_dist = np.linalg.norm(np.subtract(point, pose[:2]))
            if curr_dist < prev_min_dist:
                new_pos = np.round(point).astype(np.int)
                prev_min_dist = curr_dist

        if True and len(self.train_inputs) > 1:
            # from pymoo.visualization.scatter import Scatter
            # plot = Scatter()
            # plot.add(res.F, color="red")
            # plot.show()

            plt.style.use("seaborn")
            # results = res.F
            # img_gen = [
            #     [0, 1],
            #     [0, 2]
            # ]
            # for s in img_gen:
            #     f1 = results[:, s[0]]
            #     f4 = results[:, s[1]]
            #     plt.figure()
            #     plt.title("Objective Space", fontsize=20)
            #     plt.plot(f1, f4, 'ob', label=f"Pareto Set: $\\alpha_{s[0] + 1}(x) \\;,\\; \\alpha_{s[1] + 1}(x)$")
            #     plt.xlabel(f"$\\alpha_{s[0] + 1}(x)$", fontsize=20)
            #     plt.ylabel(f"$\\alpha_{s[1] + 1}(x)$", fontsize=20)
            #     plt.xticks(fontsize=20)
            #     plt.yticks(fontsize=20)
            #     plt.legend(loc='lower left', prop={'size': 17}, fancybox=True, shadow=True, frameon=True)

            xticks = np.arange(0, 1000, 200)
            yticks = np.arange(0, 1500, 200)
            xnticks = [str(format(num * 10, ',')) for num in xticks]
            ynticks = [str(format(num * 10, ',')) for num in yticks]
            surr = self.surrogate(keys=["s1"])[0]
            # mapz = np.full_like(self.map_data, np.nan)
            # for pos in enumerate(self.vector_pos):
            #     mapz[pos[1][1], pos[1][0]] = surr[pos[0]]

            # mapz = np.power(np.subtract(mapz, lol), 2)
            plt.imshow(self.map_data, cmap="BuGn_r",
                       origin='lower', zorder=8)
            # CS = plt.contour(mapz, colors='k',
            #                  alpha=0.6, linewidths=1.0, zorder=9)
            plt.grid(True, zorder=0, color="white")
            plt.gca().set_facecolor('#eaeaf2')
            # plt.clabel(CS, inline=1, fontsize=10)
            plt.xlabel("x (m)", fontsize=20)
            plt.ylabel("y (m)", fontsize=20)
            plt.xticks(xticks, labels=xnticks, fontsize=20)
            plt.yticks(yticks, labels=ynticks, fontsize=20)

            k = True
            # for point in self.train_inputs:
            #     if k:
            #         plt.plot(point[0], point[1], 'yo', zorder=10, label='Previous Locations')
            #         k = False
            #     else:
            #         plt.plot(point[0], point[1], 'yo', zorder=10)
            plt.plot(pose[0], pose[1], 'ro', zorder=10, label='Current Location')
            # k = True
            # for pareto in res.X:
            #     pareto = np.round(pareto).astype(np.int)
            #     if self.map_data[pareto[1], pareto[0]] == 0:
            #         if k:
            #             plt.plot(pareto[0], pareto[1], 'Xb', zorder=10, label="Pareto Points")
            #             k = False
            #         else:
            #             plt.plot(pareto[0], pareto[1], 'Xb', zorder=10)
            # plt.plot(new_pos[0], new_pos[1], 'Xg', zorder=10, label='Closest Pareto Point')
            plt.plot(np.append(reg[:, 0], reg[0, 0]), np.append(reg[:, 1], reg[0, 1]), '-b', zorder=10)
            # plt.show(block=True)

        if self.acq_mod == "split_path" or self.acq_mod == "truncated":
            beacons_splitted = []
            vect_dist = np.subtract(new_pos, pose[:2])
            ang = np.arctan2(vect_dist[1], vect_dist[0])
            d = np.exp(np.min([self.gps[key].kernel_.theta[0] for key in list(self.sensors)])) * self.proportion
            for di in np.arange(0, np.linalg.norm(vect_dist), d)[1:]:
                mini_goal = np.array([di * np.cos(ang) + pose[0], di * np.sin(ang) + pose[1]]).astype(np.int)
                if self.map_data[mini_goal[1], mini_goal[0]] == 0:
                    beacons_splitted.append(mini_goal)
            beacons_splitted.append(np.array(new_pos))
            self.splitted_goals = np.array(beacons_splitted)
            new_pos = self.splitted_goals[0, :]

            self.splitted_goals = self.splitted_goals[1:, :]

        if len(self.train_inputs) > 1:
            # plt.plot(new_pos[0], new_pos[1], 'Xk', zorder=10, label='Next Meas. Location')
            plt.legend(loc='lower left', prop={'size': 17}, fancybox=True, shadow=True,
                       frameon=True)
            plt.show(block=True)
        new_pos = np.append(new_pos, 0)

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

    def get_acq(self, key, pose=np.zeros((1, 2)), acq_func="gaussian_ei", acq_mod="normal"):
        if acq_func == "gaussian_sei":
            return gaussian_sei(self.all_vector_pos, self.gps[key], np.min(self.train_targets[key]),
                                c_point=pose[0][:2], masked=acq_mod == "masked")
        elif acq_func == "maxvalue_entropy_search":
            return maxvalue_entropy_search(self.all_vector_pos, self.gps[key], np.min(self.train_targets[key]),
                                           c_point=pose[0][:2], masked=acq_mod == "masked")
        elif acq_func == "gaussian_pi":
            return gaussian_pi(self.all_vector_pos, self.gps[key], np.min(self.train_targets[key]),
                               masked=acq_mod == "masked")
        elif acq_func == "gaussian_ei":
            return gaussian_ei(self.all_vector_pos, self.gps[key], np.min(self.train_targets[key]),
                               c_point=pose[:2],
                               masked=False, xi=1.0)
        elif acq_func == "max_std":
            return max_std(self.all_vector_pos, self.gps[key], np.min(self.train_targets[key]),
                           masked=acq_mod == "masked")
