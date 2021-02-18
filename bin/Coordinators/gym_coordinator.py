from warnings import catch_warnings
from warnings import simplefilter

# import gpytorch
import numpy as np
# import torch as to
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from skopt.learning.gaussian_process import gpr, kernels

from bin.Utils.acquisition_functions import gaussian_sei, maxvalue_entropy_search, gaussian_pi, gaussian_ei, max_std, \
    predictive_entropy_search
from bin.Utils.voronoi_regions import calc_voronoi, find_vect_pos4region


class Coordinator(object):
    def __init__(self, map_data, sensors: set, k_names=None, acq="gaussian_ei", acq_mod="masked",
                 acq_fusion="decoupled"):
        if k_names is None:
            k_names = ["RBF"] * len(sensors)
        self.map_data = map_data
        self.acquisition = acq  # 'gaussian_sei' 'gaussian_ei' 'maxvalue_entropy_search''gaussian_pi'
        self.acq_mod = acq_mod  # 'masked' 'split_path' 'truncated', 'normal'
        self.k_names = k_names  # "RBF" Matern" "RQ"
        self.sensors = sensors
        self.gps = dict()
        self.train_inputs = [np.array([[], []])]
        self.train_targets = dict()

        for sensor, kernel in zip(sensors, k_names):
            if kernel == "RBF":  # "RBF" Matern" "RQ"
                self.gps[sensor] = gpr.GaussianProcessRegressor(kernel=kernels.RBF(100), alpha=1e-7)
                self.train_targets[sensor] = np.array([])

        self.all_vector_pos = np.mgrid[0:self.map_data.shape[1]:1, 0:self.map_data.shape[0]:1].reshape(2, -1).T
        self.vector_pos = np.fliplr(np.asarray(np.where(self.map_data == 0)).reshape(2, -1).T)

        self.acq_fusion = acq_fusion
        # simple_max: maximum value found
        # max_sum: sum of acq on max for each maximum

        self.splitted_goals = []

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

    def surrogate(self, _x=None, return_std=False, keys=None):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            if keys is None:
                keys = self.sensors
            simplefilter("ignore")
            if _x is None:
                _x = self.all_vector_pos
            return [self.gps[key].predict(_x, return_std) for key in keys]

    def generate_new_goal(self, pose=np.zeros((1, 3)), other_poses=np.zeros((1, 3))):
        # nans = np.load(open('E:/ETSI/Proyecto/data/Databases/numpy_files/nans.npy', 'rb'))
        # smapz = np.zeros((1500, 1000))
        # max_mapz = None
        # c_max = 0.0

        if self.acq_mod == "split_path":
            if len(self.splitted_goals) > 0:
                new_pos = self.splitted_goals[0, :]
                self.splitted_goals = self.splitted_goals[1:, :]
                return np.append(new_pos, 0)
        xi = 1.0

        _, reg = calc_voronoi(pose, other_poses, self.map_data)
        all_acq = []
        c_max = 0.0
        new_pos = None
        sum_all_acq = None

        if self.acquisition == "predictive_entropy_search":
            gps = self.surrogate(self.vector_pos, return_std=True)
            sum_sigmas = None
            for _, sigma in gps:
                sum_sigmas = sigma if sum_sigmas is None else sigma + sum_sigmas
            x_star = self.vector_pos[np.where(sum_sigmas == np.max(sum_sigmas))[0][0]]
            for i in range(len(self.sensors)):
                mu, sigma = gps[i]
                all_acq = predictive_entropy_search(self.vector_pos, mu, sigma, model=self.gps[list(self.sensors)[i]],
                                                    x_star=x_star)
                if self.acq_fusion == "decoupled":
                    arr1inds = all_acq.argsort()
                    sorted_arr1 = self.vector_pos[arr1inds[::-1]]
                    best_pos, idx = find_vect_pos4region(sorted_arr1, reg, return_idx=True)
                    if all_acq[arr1inds[::-1][idx]] > c_max:
                        new_pos = best_pos
                        c_max = all_acq[arr1inds[::-1][idx]]
                elif self.acq_fusion == "coupled":
                    sum_all_acq = sum_all_acq + all_acq if sum_all_acq is not None else all_acq
        else:
            for key in self.sensors:
                if self.acquisition == "gaussian_sei":
                    all_acq = gaussian_sei(self.vector_pos, self.gps[key], np.min(self.train_targets[key]),
                                           c_point=pose[:2], xi=xi,
                                           masked=self.acq_mod == "masked")
                elif self.acquisition == "maxvalue_entropy_search":
                    all_acq = maxvalue_entropy_search(self.vector_pos, self.gps[key], np.min(self.train_targets[key]),
                                                      c_point=pose[:2], xi=xi,
                                                      masked=self.acq_mod == "masked")
                elif self.acquisition == "gaussian_pi":
                    all_acq = gaussian_pi(self.vector_pos, self.gps[key], np.min(self.train_targets[key]),
                                          c_point=pose[:2],
                                          xi=xi,
                                          masked=self.acq_mod == "masked")
                elif self.acquisition == "gaussian_ei":
                    all_acq = gaussian_ei(self.vector_pos, self.gps[key], np.min(self.train_targets[key]),
                                          c_point=pose[:2],
                                          xi=xi,
                                          masked=self.acq_mod == "masked")
                elif self.acquisition == "max_std":
                    all_acq = max_std(self.vector_pos, self.gps[key], np.min(self.train_targets[key]),
                                      masked=self.acq_mod == "masked")
                if self.acq_fusion == "decoupled":
                    arr1inds = all_acq.argsort()
                    sorted_arr1 = self.vector_pos[arr1inds[::-1]]
                    best_pos, idx = find_vect_pos4region(sorted_arr1, reg, return_idx=True)
                    if all_acq[arr1inds[::-1][idx]] > c_max:
                        new_pos = best_pos
                        c_max = all_acq[arr1inds[::-1][idx]]
                elif self.acq_fusion == "coupled":
                    sum_all_acq = sum_all_acq + all_acq if sum_all_acq is not None else all_acq
                # mapz = gaussian_ei(self.all_vector_pos, self.gps[key], np.min(self.train_targets[key]),
                # c_point=pose[:2],
                #                    xi=xi,
                #                    masked=self.acq_mod == "masked").reshape((1000, 1500)).T
                # smapz += mapz
                # for nnan in nans:
                #     mapz[nnan[0], nnan[1]] = -1
                # mapz = np.ma.array(mapz, mask=(mapz == -1))
                # if key == "s1":
                #     plt.subplot(231)
                # elif key == "s2":
                #     plt.subplot(232)
                # else:
                #     plt.subplot(233)
                # plt.imshow(mapz, origin='lower', cmap=cmo.cm.matter_r)
                # plt.title("$AF_{}(x)$".format(str("{" + key + "}")))
                # plt.plot(new_pos[0], new_pos[1], 'r.')
                # for pm in self.train_inputs:
                #     plt.plot(pm[0], pm[1], 'y^')
                # plt.plot(pose[0], pose[1], 'b^')
                # plt.colorbar()
                # if max_mapz is None or all_acq[arr1inds[::-1][idx]] > c_max:
                #     max_mapz = mapz
                #     c_max = all_acq[arr1inds[::-1][idx]]
        # plt.subplot(235)
        # for nnan in nans:
        #     smapz[nnan[0], nnan[1]] = -1
        # smapz = np.ma.array(smapz, mask=(smapz == -1))
        # plt.imshow(smapz, origin='lower', cmap=cmo.cm.matter_r)
        # plt.title("$\sum AF_i(x)$")
        # maxx = np.where(smapz == np.max(smapz))
        # plt.plot(maxx[1][0], maxx[0][0], 'r.')
        # plt.plot(pose[0], pose[1], 'b^', zorder=9)
        # for pm in self.train_inputs:
        #     plt.plot(pm[0], pm[1], 'y^')
        # plt.colorbar()

        # plt.legend(["best_next", "c_pose", "prev. measurements"], bbox_to_anchor=(3.5, 1.0), fancybox=True,
        # shadow=True)

        # plt.subplot(234)
        # plt.imshow(max_mapz, origin='lower', cmap=cmo.cm.matter_r)
        # for pm in self.train_inputs:
        #     plt.plot(pm[0], pm[1], 'y^')
        # plt.plot(pose[0], pose[1], 'b^')
        # plt.title("$max(AF_i(x))$")
        # maxx = np.where(max_mapz == np.max(max_mapz))
        # plt.plot(maxx[1][0], maxx[0][0], 'r.')
        # plt.colorbar()
        # plt.show(block=True)

        # if self.acq_fusion == "maxcoupled":
        #     for best_pos in new_poses:
        #         suma = 0
        #         for key in self.sensors:
        #             this_acq = gaussian_ei(best_pos[0],
        #                                    self.surrogate(best_pos[0].reshape(1, -1), return_std=True, keys=[key])[0],
        #                                    np.min(self.train_targets[key]),
        #                                    c_point=pose[:2], xi=xi, masked=self.acq_mod == "masked")
        #             suma += this_acq
        #         if suma > c_max:
        #             new_pos = best_pos[0]
        #             c_max = suma
        if self.acq_fusion == "coupled":
            arr1inds = sum_all_acq.argsort()
            sorted_arr1 = self.vector_pos[arr1inds[::-1]]
            best_pos = find_vect_pos4region(sorted_arr1, reg, return_idx=False)
            new_pos = best_pos
        if self.acq_mod == "split_path" or self.acq_mod == "truncated":
            beacons_splitted = []
            vect_dist = np.subtract(new_pos, pose[:2])
            ang = np.arctan2(vect_dist[1], vect_dist[0])
            d = np.min([self.gps[key].kernel_.theta[0] for key in list(self.sensors)]) / 2
            for di in np.arange(0, np.linalg.norm(vect_dist), d)[1:]:
                mini_goal = np.array([di * np.cos(ang) + pose[0], di * np.sin(ang) + pose[1]]).astype(np.int)
                if self.map_data[mini_goal[1], mini_goal[0]] == 0:
                    beacons_splitted.append(mini_goal)
            beacons_splitted.append(np.array(new_pos))
            self.splitted_goals = np.array(beacons_splitted)
            new_pos = self.splitted_goals[0, :]

            self.splitted_goals = self.splitted_goals[1:, :]
        new_pos = np.append(new_pos, 0)

        return new_pos

    def get_mse(self, y_true, key=None):
        if key is None:
            key = list(self.sensors)[0]
        nan = np.isnan(y_true)
        return mse(y_true[~nan], self.surrogate(keys=[key])[0][~nan])

    def get_score(self, y_true, key=None):
        if key is None:
            key = list(self.sensors)[0]
        nan = np.isnan(y_true)
        return r2_score(y_true[~nan], self.surrogate(keys=[key])[0][~nan])

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

# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_inputs, train_targets, likelihood):
#         super(ExactGPModel, self).__init__(likelihood=likelihood, train_inputs=train_inputs,
#                                            train_targets=train_targets)
#         self.mean_module = gpytorch.means.ConstantMean()
#         lengthscale_prior = gpytorch.priors.UniformPrior(100, 6.0)
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))
#
#         self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class TorchCoordinator:
#     def __init__(self, map_data, sensors: set, k_names=None, acq="gaussian_ei", acq_mod="masked"):
#         if k_names is None:
#             k_names = ["RBF"]
#         device = to.device("cpu")
#         self.map_data = map_data
#         self.acquisition = acq  # 'gaussian_sei' 'gaussian_ei' 'maxvalue_entropy_search''gaussian_pi'
#         self.acq_mod = acq_mod  # 'masked' 'split_path' 'truncated', 'normal'
#         self.k_names = k_names  # "RBF" Matern" "RQ"
#         self.likelihoods = dict()
#         self.gps = dict()
#
#         self.train_inputs = to.empty(1, 2, device=device, dtype=to.int32)
#         self.train_targets = dict()
#         self.sensors = sensors
#         for sensor, kernel in zip(sensors, k_names):
#             if kernel == "RBF":
#                 self.likelihoods[sensor] = gpytorch.likelihoods.GaussianLikelihood()
#                 self.train_targets[sensor] = to.empty(1, 1, device=device, dtype=to.float32)
#                 self.gps[sensor] = ExactGPModel(train_inputs=self.train_inputs,
#                                                 train_targets=self.train_targets[sensor],
#                                                 likelihood=self.likelihoods[sensor])
#             else:
#                 raise NotImplementedError("add more kernels pls")
#
#         self.all_vector_pos = to.tensor(
#             np.mgrid[0:self.map_data.shape[1]:1, 0:self.map_data.shape[0]:1].reshape(2, -1).T, dtype=to.int)
#         self.vector_pos = to.tensor(np.fliplr(np.asarray(np.where(self.map_data == 0)).reshape(2, -1).T).copy(),
#                                     dtype=to.int)
#
#         self.splitted_goals = []
#
#     def initialize_data_gpr(self, data: list):
#         self.train_inputs = to.tensor(np.array(data[0]['pos']).reshape(-1, 2))
#         for key in data[0].keys():
#             if key in self.sensors:
#                 self.train_targets[key] = to.tensor([data[0][key]], dtype=to.float32)
#         if len(data) > 1:
#             for nd in data[1:]:
#                 self.add_data(nd, set_train_data=False)
#
#         for key in self.sensors:
#             self.gps[key].set_train_data(self.train_inputs, self.train_targets[key], strict=False)
#         self.fit_data()
#
#     def add_data(self, new_data, set_train_data=True):
#         self.train_inputs = to.cat(
#             (self.train_inputs, to.tensor(np.array(new_data['pos']).reshape(-1, 2), dtype=to.int32)), 0)
#         for key in new_data.keys():
#             if key in self.sensors:
#                 self.train_targets[key] = to.cat(
#                     (self.train_targets[key], to.tensor([new_data[key]], dtype=to.float32)), 0)
#         if set_train_data:
#             for key in self.sensors:
#                 self.gps[key].set_train_data(self.train_inputs, self.train_targets[key], strict=False)
#
#     def fit_data(self):
#         for name in self.sensors:
#             self.gps[name].train()
#             self.likelihoods[name].train()
#             # Use the adam optimizer
#             optimizer = to.optim.Adam(self.gps[name].parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
#
#             # "Loss" for GPs - the marginal log likelihood
#             mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[name], self.gps[name])
#
#             for i in range(50):
#                 # Zero gradients from previous iteration
#                 optimizer.zero_grad()
#                 # Output from model
#                 output = self.gps[name](self.train_inputs)
#                 # Calc loss and backprop gradients
#                 loss = -mll(output, self.train_targets[name])
#                 loss.backward()
#                 optimizer.step()
#                 if loss.item() < 0:
#                     break
#                 # print('Iter %d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#                 #     i + 1, loss.item(),
#                 #     self.gps[name].covar_module.base_kernel.lengthscale.item(),
#                 #     self.gps[name].likelihood.noise.item()
#                 # ))
#
#         # self.gp.fit(self.data[0], self.data[1])
#
#     def surrogate(self, _x=None, return_std=False, key=None):
#         if key is None:
#             key = self.sensors
#         if _x is None:
#             _x = self.all_vector_pos
#         [self.gps[name].eval() for name in key]
#         [self.likelihoods[name].eval() for name in key]
#         with to.no_grad(), gpytorch.settings.fast_pred_var():
#             y_preds = [self.likelihoods[name](self.gps[name](_x)) for name in key]
#
#         return [y_pred.mean.numpy() for y_pred in y_preds], [y_pred.stddev.detach().numpy() for y_pred in
#                                                              y_preds] if return_std else [y_pred.mean.numpy() for y_pred
#                                                                                           in y_preds]
#
#     def generate_new_goal(self, pose=np.zeros((1, 3)), other_poses=np.zeros((1, 3))):
#         if self.acq_mod == "split_path":
#             if len(self.splitted_goals) > 0:
#                 new_pos = self.splitted_goals[0, :]
#                 self.splitted_goals = self.splitted_goals[1:, :]
#                 return np.append(new_pos, 0)
#         xi = 1.0
#
#         _, reg = calc_voronoi(pose, other_poses, self.map_data)
#
#         for key in self.sensors:
#             mu, std = self.surrogate(self.vector_pos, key=[key], return_std=True)
#             if self.acquisition == "gaussian_sei":
#                 all_acq = gaussian_sei(self.vector_pos, mu, std,
#                                        to.min(self.train_targets[key]),
#                                        c_point=pose[:2], xi=xi,
#                                        masked=self.acq_mod == "masked")
#             elif self.acquisition == "maxvalue_entropy_search":
#                 all_acq = maxvalue_entropy_search(self.vector_pos, mu, std,
#                                                   to.min(self.train_targets[key]),
#                                                   c_point=pose[:2], xi=xi,
#                                                   masked=self.acq_mod == "masked")
#             elif self.acquisition == "gaussian_pi":
#                 all_acq = gaussian_pi(self.vector_pos, mu, std,
#                                       to.min(self.train_targets[key]),
#                                       c_point=pose[:2], xi=xi,
#                                       masked=self.acq_mod == "masked")
#             elif self.acquisition == "gaussian_ei":
#                 all_acq = gaussian_ei(self.vector_pos, mu[0], std[0],
#                                       to.min(self.train_targets[key]),
#                                       c_point=pose[:2], xi=xi,
#                                       masked=self.acq_mod == "masked")
#             elif self.acquisition == "max_std":
#                 all_acq = max_std(self.vector_pos, mu, std,
#                                   to.min(self.train_targets[key]),
#                                   masked=self.acq_mod == "masked")
#             else:
#                 all_acq = []
#
#         arr1inds = all_acq.argsort()
#         sorted_arr1 = self.vector_pos.numpy()[arr1inds[::-1]]
#         new_pos = find_vect_pos4region(sorted_arr1, reg)
#
#         if self.acq_mod == "split_path" or self.acq_mod == "truncated":
#             beacons_splitted = []
#             vect_dist = np.subtract(new_pos, pose[:2])
#             ang = np.arctan2(vect_dist[1], vect_dist[0])
#             d = 220
#             for di in np.arange(0, np.linalg.norm(vect_dist), d)[1:]:
#                 mini_goal = np.array([di * np.cos(ang) + pose[0], di * np.sin(ang) + pose[1]]).astype(np.int)
#                 if self.map_data[mini_goal[1], mini_goal[0]] == 0:
#                     beacons_splitted.append(mini_goal)
#             beacons_splitted.append(np.array(new_pos))
#             self.splitted_goals = np.array(beacons_splitted)
#             new_pos = self.splitted_goals[0, :]
#
#             self.splitted_goals = self.splitted_goals[1:, :]
#         new_pos = np.append(new_pos, 0)
#
#         return new_pos
#
#     def get_mse(self, y_true):
#         nan = np.isnan(y_true)
#         return mse(y_true[~nan], self.surrogate()[~nan])
#
#     def get_score(self, y_true):
#         nan = np.isnan(y_true)
#         return r2_score(y_true[~nan], self.surrogate()[~nan])
#
#     def get_acq(self, pose=np.zeros((1, 2)), acq_func="gaussian_sei", acq_mod="normal"):
#         if acq_func == "gaussian_sei":
#             return gaussian_sei(self.all_vector_pos, self.gp, np.min(self.data[1]),
#                                 c_point=pose[0][:2], masked=acq_mod == "masked"), self.main_sensor
#         elif acq_func == "maxvalue_entropy_search":
#             return maxvalue_entropy_search(self.all_vector_pos, self.gp, np.min(self.data[1]),
#                                            c_point=pose[0][:2], masked=acq_mod == "masked"), self.main_sensor
#         elif acq_func == "gaussian_pi":
#             return gaussian_pi(self.all_vector_pos, self.gp, np.min(self.data[1]),
#                                masked=acq_mod == "masked"), self.main_sensor
#         elif acq_func == "gaussian_ei":
#             return gaussian_ei(self.all_vector_pos, self.gp, np.min(self.data[1]),
#                                c_point=pose[:2],
#                                masked=False, xi=1.0), self.main_sensor
#         elif acq_func == "max_std":
#             return max_std(self.all_vector_pos, self.gp, np.min(self.data[1]),
#                            masked=acq_mod == "masked"), self.main_sensor
