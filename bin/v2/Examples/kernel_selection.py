from sys import path

import matplotlib.pyplot as plt
import numpy as np

path.extend([path[0][:path[0].rindex("bin") - 1]])
from bin.Agents.pathplanning_agent import SimpleAgent
from bin.Coordinators.informed_coordinator import Coordinator
from bin.Environment.simple_env import Env

# EXPERIMENTS = 50
# SIZE = 15
# Matern has mean, std MSE: 0.08003373876971603, 0.1140966672448736
# RQ has mean, std MSE: 0.07049722685106351, 0.1077240751360609
# RBF has mean, std MSE: 0.07066263792095316, 0.1077025811611788


EXPERIMENTS = 1
SIZE = 50
seeds = np.linspace(76842153, 1123581321, 100)

sensors = ["t"]

drones = [SimpleAgent(sensors)]
env = Env(map_path2yaml="E:/ETSI/Proyecto/data/Map/Ypacarai/map.yaml")
env.add_new_map(sensors, file=53)

plt.style.use("seaborn")

plt.subplot(121)
plt.imshow(env.render_maps()["t"], origin='lower', cmap='inferno')
CS = plt.contour(env.render_maps()["t"], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                 alpha=0.6, linewidths=1.0)
plt.clabel(CS, inline=1, fontsize=10)
plt.title("Ground Truth")
plt.show(block=False)
plt.pause(0.00001)

mses = {"RBF": []}  # "Matern": [], "RQ": [],
with open('E:/ETSI/Proyecto/data/Databases/numpy_files/nans.npy', 'rb') as g:
    nans = np.load(g)
# nans = []
# for k in range(np.shape(env.grid)[0]):
#     for j in range(np.shape(env.grid)[1]):
#         if env.grid[k, j] == 1.0:
#             nans.append([k, j])
# with open('E:/ETSI/Proyecto/data/Databases/numpy_files/nans.npy', 'wb') as g:
#     np.save(g, nans)

# candidates = np.array([
#     [671, 906],
#     [625, 899],
#     [577, 900],
#     [538, 923],
#     [519, 950],
#     [504, 932],
#     [493, 892],
#     [487, 850],
#     [476, 809],
#     [501, 521],
#     [517, 481],
#     [551, 451],
#     [591, 429],
#     [630, 397],
#     [669, 365]
# ])

for k in range(EXPERIMENTS):
    print('current experiment is: ', k)
    np.random.seed(np.round(seeds[k]).astype(int))
    d = []
    x = []
    while len(d) < SIZE:
        candidate = np.round(np.random.uniform([0, 0], [999, 1499]).astype(int))
        # candidate = candidates[0]
        # candidates = candidates[1:]
        if env.grid[candidate[1], candidate[0]] == 0:
            x.append(candidate)
            d.append([candidate, env.maps["t"][candidate[1], candidate[0]]])
    coords = [
        # Coordinator(env.grid, "t", "RQ"),
        Coordinator(env.grid, "t", "RBF")
        # Coordinator(env.grid, "t", "Matern")
    ]
    mu = dict()
    sd = dict()
    for coord in coords:
        coord.initialize_data_gpr(d)
        mses[coord.k_name].append(coord.get_mse(env.maps['t'].T.flatten()))
        cmu, csd = coord.surrogate(return_std=True, return_sensor=False)
        mu[coord.k_name] = cmu.reshape((1000, 1500)).T
        # sd[coord.k_name] = csd.reshape((1000, 1500)).T

        for nnan in nans:
            mu[coord.k_name][nnan[0], nnan[1]] = -1
        mu[coord.k_name] = np.ma.array(mu[coord.k_name], mask=(mu[coord.k_name] == -1))
        # for nnan in nans:
        #     sd[coord.k_name][nnan[0], nnan[1]] = -1
        # sd[coord.k_name] = np.ma.array(sd[coord.k_name], mask=(sd[coord.k_name] == -1))

for coor in coords:
    print(np.exp(coord.gp.kernel_.theta))
# plt.style.use("seaborn")
# legends = []
# for key in mses:
#     print("{} has mean, std MSE: {}, {}".format(key, np.mean(mses[key]), np.std(mses[key])))
#     plt.hist(mses[key], alpha=0.3)
#     legends.append(key)
# plt.legend(legends)
# plt.show(block=True)

# plt.subplot(121)
# # plt.imshow(env.render_maps()["t"], origin='lower', cmap='inferno')
x = np.array(x)
# plt.plot(x[:, 0], x[:, 1], 'ob')
# CS = plt.contour(env.render_maps()["t"], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
#                  alpha=0.6, linewidths=1.0)
# plt.clabel(CS, inline=1, fontsize=10)
# plt.title("Ground Truth")
# plt.figure()
# plt.xlabel("x")
# plt.ylabel("y")
#
i = 2
for key in mu.keys():
    #     plt.subplot(240 + i)
    plt.subplot(122)
    plt.imshow(mu[key], origin='lower', cmap='inferno')
    plt.plot(x[:, 0], x[:, 1], 'ob')
    CS = plt.contour(mu[key], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                     alpha=0.6, linewidths=1.0)

    plt.clabel(CS, inline=1, fontsize=10)
    plt.title("{} has MSE: {}".format(key, coords[i - 2].get_mse(env.maps['t'].T.flatten())))
    # plt.subplot(240 + i + 4)
    # plt.imshow(sd[key], origin='lower', cmap='viridis')
    # plt.plot(x[:, 0], x[:, 1], 'ob')
    # plt.title("95% conf. std")
#     i += 1
#
plt.show(block=True)
