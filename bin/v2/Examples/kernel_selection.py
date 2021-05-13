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
SIZE = 2
seeds = np.linspace(76842153, 1123581321, 100)

sensors = ["t"]

drones = [SimpleAgent(sensors)]
env = Env(map_path2yaml="E:/ETSI/Proyecto/data/Map/Simple/map.yaml")
env.add_new_map(sensors, file=98)

plt.style.use("seaborn")

# plt.subplot(241)
plt.subplot(131)
plt.imshow(env.render_maps()["t"], origin='lower', cmap='inferno')
CS = plt.contour(env.render_maps()["t"], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                 alpha=0.6, linewidths=1.0)
plt.clabel(CS, inline=1, fontsize=10)
plt.title("Ground Truth")
# plt.show(block=False)
# plt.pause(0.01)

mses = {"RBF": [], "RBF_N": [], "RQ": []}
with open('E:/ETSI/Proyecto/data/Databases/numpy_files/nans.npy', 'rb') as g:
    nans = np.load(g)

# candidates = [[263, 923],
#               [287, 888],
#               [176, 930],
#               [301, 943],
#               [410, 795],
#               [563, 848],
#               [483, 750],
#               [433, 673],
#               [377, 596],
#               [318, 517],
#               [263, 428],
#               [316, 335],
#               [243, 257],
#               [168, 178],
#               [94, 99],
#               [201, 87],
#               [309, 75],
#               [416, 61],
#               [524, 48],
#               [634, 37],
#               [722, 103],
#               [623, 153],
#               [668, 254],
#               [713, 356],
#               [759, 459],
#               [811, 547]]
candidates = [[489, 796],
              # [508, 753],
              # [487, 799],
              # [511, 876],
              # [419, 887],
              # [304, 888],
              # [264, 771],
              # [221, 647],
              # [154, 563],
              # [160, 453],
              # [122, 346],
              # [84, 239],
              # [147, 142],
              # [63, 61],
              # [185, 53],
              # [307, 45],
              # [430, 36],
              # [552, 27],
              # [676, 19],
              # [715, 139],
              # [754, 259],
              # [794, 380],
              # [834, 502],
              # [880, 604],
              [899, 503],
              [919, 402], ]

# candidate = np.round(np.random.uniform([0, 0], [999, 999]).astype(int))
for k in range(EXPERIMENTS):
    print('current experiment is: ', k)
    np.random.seed(np.round(seeds[k]).astype(int))
    d = []
    x = []
    while len(d) < SIZE:
        # candidate = candidate + np.round(0.4266*153 * np.random.uniform([-1, -1], [1, 1])).astype(int)
        # candidate = np.clip(candidate, [0, 0], [999, 999])
        candidate = candidates[0]
        candidates = candidates[1:]
        if env.grid[candidate[1], candidate[0]] == 0:
            x.append(candidate)
            d.append([candidate, env.maps["t"][candidate[1], candidate[0]]])
    coords = [
        # Coordinator(env.grid, "t", "RQ"),
        Coordinator(env.grid, "t", "RBF"),
        # Coordinator(env.grid, "t", "RBF_N")
    ]
    mu = dict()
    sd = dict()
    for coord in coords:
        coord.initialize_data_gpr(d)
        # mses[coord.k_name].append(coord.get_mse(env.maps['t'].T.flatten()))
        cmu, csd = coord.surrogate(return_std=True, return_sensor=False)
        mu[coord.k_name] = cmu.reshape((1000, 1000)).T
        sd[coord.k_name] = csd.reshape((1000, 1000)).T

        # for nnan in nans:
        #     mu[coord.k_name][nnan[0], nnan[1]] = -1
        mu[coord.k_name] = np.ma.array(mu[coord.k_name], mask=(mu[coord.k_name] == -1))
        # for nnan in nans:
        #     sd[coord.k_name][nnan[0], nnan[1]] = -1
        sd[coord.k_name] = np.ma.array(sd[coord.k_name], mask=(sd[coord.k_name] == -1))
        # print(np.exp(coord.gp.kernel_.theta))
        print(coord.gp.kernel_)
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
    plt.subplot(130 + i)
    # plt.subplot(122)
    # plt.subplot(240 + i)
    plt.imshow(mu[key], origin='lower', cmap='inferno')
    for p in enumerate(x):
        plt.text(p[1][0], p[1][1], p[0])
    plt.plot(x[:, 0], x[:, 1], 'ob')
    CS = plt.contour(mu[key], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                     alpha=0.6, linewidths=1.0)

    plt.clabel(CS, inline=1, fontsize=10)
    plt.title("{} has R2s: {}".format(key, coords[i - 2].get_score(env.maps['t'].T.flatten())))
    # plt.subplot(240 + i + 4)
    plt.subplot(130 + i + 1)
    plt.imshow(sd[key], origin='lower', cmap='viridis')
    plt.plot(x[:, 0], x[:, 1], 'ob')
    plt.title("95% conf. std")
    i += 1
#
plt.show(block=True)
