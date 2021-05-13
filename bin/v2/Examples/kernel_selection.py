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
SIZE = 25
seeds = np.linspace(76842153, 1123581321, 100)

sensors = ["t"]

drones = [SimpleAgent(sensors)]
env = Env(map_path2yaml="E:/ETSI/Proyecto/data/Map/Simple/map.yaml")
env.add_new_map(sensors, file=99)

plt.style.use("seaborn")

plt.subplot(241)
plt.imshow(env.render_maps()["t"], origin='lower', cmap='inferno')
CS = plt.contour(env.render_maps()["t"], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                 alpha=0.6, linewidths=1.0)
plt.clabel(CS, inline=1, fontsize=10)
plt.title("Ground Truth")
plt.show(block=False)
plt.pause(0.01)

mses = {"RBF": [], "RBF_N": [], "RQ": []}
with open('E:/ETSI/Proyecto/data/Databases/numpy_files/nans.npy', 'rb') as g:
    nans = np.load(g)

candidates = [[474, 963],
              [502, 937],
              [468, 874],
              [516, 794],
              [464, 714],
              [527, 630],
              [476, 569],
              [527, 513],
              [468, 455],
              [532, 399],
              [462, 346],
              [528, 303],
              [458, 262],
              [526, 228],
              [591, 196],
              [624, 261],
              [657, 326],
              [690, 391],
              [722, 455],
              [754, 519],
              [694, 477],
              [634, 435],
              [573, 393],
              [522, 446],
              [472, 499],
              [421, 552]]

candidates = [[474, 963],
              [510, 950],
              [417, 933],
              [324, 948],
              [282, 826],
              [235, 688],
              [189, 564],
              [139, 455],
              [215, 370],
              [316, 322],
              [416, 274],
              [512, 228],
              [592, 189],
              [544, 253],
              [496, 317],
              [448, 381],
              [387, 433],
              [342, 498],
              [297, 563],
              [364, 605],
              [432, 647],
              [497, 692],
              [563, 736],
              [490, 769],
              [417, 803],
              [344, 837]]

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
    # plt.subplot(130 + i)
    # plt.subplot(122)
    plt.subplot(240 + i)
    plt.imshow(mu[key], origin='lower', cmap='inferno')
    plt.plot(x[:, 0], x[:, 1], 'ob')
    CS = plt.contour(mu[key], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                     alpha=0.6, linewidths=1.0)

    plt.clabel(CS, inline=1, fontsize=10)
    plt.title("{} has R2s: {}".format(key, coords[i - 2].get_score(env.maps['t'].T.flatten())))
    plt.subplot(240 + i + 4)
    plt.imshow(sd[key], origin='lower', cmap='viridis')
    plt.plot(x[:, 0], x[:, 1], 'ob')
    plt.title("95% conf. std")
    i += 1
#
plt.show(block=True)
