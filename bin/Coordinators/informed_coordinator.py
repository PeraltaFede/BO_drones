from warnings import catch_warnings
from warnings import simplefilter

import numpy as np
from sklearn.metrics import mean_squared_error as mse
# from skopt.acquisition import gaussian_ei
from skopt.learning.gaussian_process import gpr, kernels

try:
    from Utils.acquisition_functions import gaussian_sei, maxvalue_entropy_search, gaussian_pi, gaussian_ei
except ModuleNotFoundError:
    from bin.Utils.acquisition_functions import gaussian_sei, maxvalue_entropy_search, gaussian_pi, gaussian_ei


# from skopt.acquisition import gaussian_ei


# import matplotlib.pyplot as plt


class Coordinator(object):

    def __init__(self, map_data, main_sensor='None', k_name="RBF", acq="gaussian_sei", acq_mod="masked"):
        # self.agents = agents
        self.map_data = map_data
        self.acquisition = acq  # 'gaussian_sei' 'gaussian_ei' 'maxvalue_entropy_search''gaussian_pi'
        self.acq_mod = acq_mod  # 'masked' 'split_path' 'truncated', 'normal'
        self.k_name = k_name  # "RBF" Matern" "RQ"
        if k_name == "RBF":
            self.gp = gpr.GaussianProcessRegressor(kernel=kernels.RBF(100), alpha=1e-7)
        elif k_name == "Matern":
            self.gp = gpr.GaussianProcessRegressor(kernel=kernels.Matern(100, nu=3.5), alpha=1e-7)
        elif k_name == "RQ":
            self.gp = gpr.GaussianProcessRegressor(kernel=kernels.RationalQuadratic(100, 0.1), alpha=1e-7)

        self.data = [np.array([[], []]), np.array([])]

        self.all_vector_pos = np.mgrid[0:self.map_data.shape[1]:1, 0:self.map_data.shape[0]:1].reshape(2, -1).T
        self.vector_pos = np.fliplr(np.asarray(np.where(self.map_data == 0)).reshape(2, -1).T)

        self.main_sensor = main_sensor
        self.splitted_goals = []

    def initialize_data_gpr(self, data):
        self.data = [np.array(data[0][0]).reshape(-1, 2), np.array([data[0][1]])]
        if len(data) > 1:
            for nd in data[1:]:
                self.add_data(nd)
        self.fit_data()

    def fit_data(self):
        self.gp.fit(self.data[0], self.data[1])
        # datos = self.gp.kernel(self.all_vector_pos[25500:27000])
        # datos2 = self.gp.kernel_(self.all_vector_pos[25500:27000])
        # print(self.gp.kernel.get_params())
        # print(self.gp.kernel_.get_params())
        # mini = np.min(datos)
        # maxi = np.max(datos)
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(datos, vmin=mini, vmax=maxi)
        # plt.subplot(122)
        # plt.imshow(datos2, vmin=mini, vmax=maxi)
        # plt.show(block=True)

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

    def generate_new_goal(self, pose=np.zeros((1, 3)), idx=-1):
        if self.acq_mod == "split_path":
            if len(self.splitted_goals) > 0:
                new_pos = self.splitted_goals[0, :]
                self.splitted_goals = self.splitted_goals[1:, :]
                print('new pos is', new_pos)

                return np.append(new_pos, 0)
        xi = 1.0
        if self.acquisition == "gaussian_sei":
            all_acq = gaussian_sei(self.vector_pos, self.gp, np.min(self.data[1]), c_point=pose[:2], xi=xi,
                                   masked=self.acq_mod == "masked")
        elif self.acquisition == "maxvalue_entropy_search":
            all_acq = maxvalue_entropy_search(self.vector_pos, self.gp, np.min(self.data[1]), c_point=pose[:2], xi=xi,
                                              masked=self.acq_mod == "masked")
        elif self.acquisition == "gaussian_pi":
            all_acq = gaussian_pi(self.vector_pos, self.gp, np.min(self.data[1]), c_point=pose[:2], xi=xi,
                                  masked=self.acq_mod == "masked")
        elif self.acquisition == "gaussian_ei":
            all_acq = gaussian_ei(self.vector_pos, self.gp, np.min(self.data[1]), c_point=pose[:2], xi=xi,
                                  masked=self.acq_mod == "masked")
            # all_acq = gaussian_ei(self.vector_pos, self.gp, np.min(self.data[1]), xi=xi, return_grad=False)
        new_pos = self.vector_pos[np.where(all_acq == np.nanmax(all_acq))][0]

        if self.acq_mod == "split_path" or self.acq_mod == "truncated":
            beacons_splitted = []
            vect_dist = np.subtract(new_pos, pose[:2])
            ang = np.arctan2(vect_dist[1], vect_dist[0])
            lx = np.round(np.arange(pose[0], new_pos[0], 220 * np.cos(ang))[1:]).astype(np.int)
            ly = np.round(np.arange(pose[1], new_pos[1], 220 * np.sin(ang))[1:]).astype(np.int)
            for dx, dy in zip(lx, ly):
                if self.map_data[dy, dx] == 0:
                    beacons_splitted.append(np.array([dx, dy]))
            beacons_splitted.append(np.array(new_pos))
            self.splitted_goals = np.array(beacons_splitted)
            new_pos = self.splitted_goals[0, :]
            # if idx == 5:
            # plt.plot(np.append(pose[0], self.splitted_goals[:, 0]), np.append(pose[1], self.splitted_goals[:, 1]),
            #          '-*b')
            # plt.plot([pose[0], new_pos[0]], [pose[1], new_pos[1]], '-*b')
            self.splitted_goals = self.splitted_goals[1:, :]
        # if idx == 0:
        #     all_acq, all_grad = gaussian_ei(self.all_vector_pos, self.gp, np.min(self.data[1]), xi=xi,
        #     return_grad=True)

        #     all_acq = gaussian_sei(self.vector_pos, self.gp, np.max(self.data[1]), c_point=pose[:2],
        #                            masked=self.acq_mod == "masked")
        #     best_acqs = self.vector_pos[np.where(all_acq == np.nanmax(all_acq))]
        #     new_pos = best_acqs[np.random.choice(np.arange(0, len(best_acqs)))]
        #
        #     self.acq_mod = "masked"
        #
        #     all_acq = gaussian_sei(self.vector_pos, self.gp, np.max(self.data[1]), c_point=pose[:2],
        #                            masked=self.acq_mod == "masked")
        #     best_acqs = self.vector_pos[np.where(all_acq == np.nanmax(all_acq))]
        #     aux_pos = best_acqs[np.random.choice(np.arange(0, len(best_acqs)))]
        #     plt.plot(pose[0], pose[1], '^r', markersize=12, label="Current Position")
        #
        #     # beacons_splitted = []
        #     # vect_dist = np.subtract(new_pos, pose[:2])
        #     # ang = np.arctan2(vect_dist[1], vect_dist[0])
        #     # lx = np.round(np.arange(pose[0], new_pos[0], 150 * np.cos(ang))[1:]).astype(np.int)
        #     # ly = np.round(np.arange(pose[1], new_pos[1], 150 * np.sin(ang))[1:]).astype(np.int)
        #     # for dx, dy in zip(lx, ly):
        #     #     if self.map_data[dy, dx] == 0:
        #     #         beacons_splitted.append(np.array([dx, dy]))
        #     # beacons_splitted.append(np.array(new_pos))
        #     # asdf = np.array(beacons_splitted)
        #     # plt.plot(asdf[:, 0], asdf[:, 1], 'Xb', markersize=12, label="Next positions")
        #     # plt.plot(aux_pos[0], aux_pos[1], 'Xb', markersize=12, label="Next Position")
        #     # plt.plot(new_pos[0], new_pos[1], 'Xk', markersize=12, alpha=0.2, label="Optimal Position")
        #
        # extent = [0, 999, 0, 1499]
        #     img, _ = self.get_acq([pose], "gaussian_ei")
        #     img = img.reshape((1000, 1000)).T
        #
        # plt.imshow(all_grad.reshape((1000, 1500)), origin='lower', cmap='YlGn_r', extent=extent)
        # plt.show(block=True)
        #     from scipy.spatial.distance import cdist
        #     # cmap = plt.cm.gray
        #     # my_cmap = cmap(np.arange(cmap.N))
        #     # my_cmap[:, -1] = np.linspace(0.8, 0, cmap.N)
        #     # my_cmap = ListedColormap(my_cmap)
        #     img = np.exp(-cdist([pose[:2]], self.all_vector_pos) / 250).reshape((1000, 1000)).T
        #     # for nnan in nans:
        #     #     img[nnan[0], nnan[1]] = -1
        #     plt.imshow(img, origin='lower', cmap="gray")
        #
        #     plt.legend(loc="upper right", prop={"size": 20})
        #     cbar = plt.colorbar(orientation='vertical')
        #     cbar.ax.tick_params(labelsize=20)
        #     plt.show(block=True)
        # plt.plot(new_pos[0], new_pos[1], '^y', markersize=12)
        new_pos = np.append(new_pos, 0)

        # print('new pos is', new_pos)
        return new_pos

    def get_mse(self, y_true):
        nan = np.isnan(y_true)
        return mse(y_true[~nan], self.surrogate()[~nan])

    def get_acq(self, pose=np.zeros((1, 2)), acq_func="gaussian_sei"):
        if acq_func == "gaussian_sei":
            return gaussian_sei(self.all_vector_pos, self.gp, np.min(self.data[1]),
                                c_point=pose[:2], masked=self.acq_mod == "masked", xi=1.0), self.main_sensor
        elif acq_func == "maxvalue_entropy_search":
            return maxvalue_entropy_search(self.all_vector_pos, self.gp, np.min(self.data[1]),
                                           c_point=pose[:2], masked=self.acq_mod == "masked"), self.main_sensor
        elif acq_func == "gaussian_pi":
            return gaussian_pi(self.all_vector_pos, self.gp, np.min(self.data[1]),
                               masked=self.acq_mod == "masked"), self.main_sensor
        elif acq_func == "gaussian_ei":
            return gaussian_ei(self.all_vector_pos, self.gp, np.min(self.data[1]),
                               c_point=pose[:2],
                               masked=False, xi=1.0), self.main_sensor
        # return ge(self.all_vector_pos, self.gp, 3), self.main_sensor
