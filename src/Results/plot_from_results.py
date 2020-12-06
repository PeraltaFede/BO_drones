import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn")

# name_files = glob.glob("E:/ETSI/Proyecto/results/csv_results/facq/*.csv")
# name_files = glob.glob("E:/ETSI/Proyecto/results/csv_results/modfacq/*.csv")
name_files = glob.glob("E:/ETSI/Proyecto/results/csv_results/*.csv")

datas = []
dataype = []

# for_comparison = ["gaussian_pimin", "gaussian_eimin", "gaussian_seimin", "maxvalue_entropy_searchmin", "gaussian_pi",
#                   "gaussian_ei", "gaussian_sei", "maxvalue_entropy_search"]
# for_comparison = ["gaussian_pimin", "gaussian_eimin", "gaussian_seimin", "maxvalue_entropy_search"]
# for_comparison = ["gaussian_pimin", "gaussian_pi"]
# for_comparison = ["gaussian_eimin", "gaussian_ei"]
# for_comparison = ["gaussian_seimin", "gaussian_sei"]
# for_comparison = ["maxvalue_entropy_searchmin", "maxvalue_entropy_search"]
# for_comparison = ["gaussian_seimin", "gaussian_sei", "gaussian_eimin"]
# for_comparison = ["maskedxi", "truncatedxi", "split_pathxi", "masked", "truncated", "split_path"]
# for_comparison = ["masked", "split_path"]  # , "truncated"
# for_comparison = ["split_path", "masked", "truncated"]
# for_comparison = ["gaussian_sei,split_path", "gaussian_sei,truncated", "gaussian_sei,masked",
#                   "gaussian_pi,split_path", "gaussian_pi,truncated", "gaussian_pi,masked",
#                   "gaussian_ei,split_path", "gaussian_ei,truncated", "gaussian_ei,masked"]
# for_comparison = ["gaussian_pi,truncated",
#                   "gaussian_ei,truncated"]  # ,
                  # "gaussian_sei,truncated"]
# for_comparison = ["RBF", "Matern", "RQ"]
for_comparison = ["truncated", "GA", "LM"]
# for_comparison = ["truncated"]

for name_file in name_files:
    with open(name_file, 'r') as f:
        f.readline()
        rl = f.readline()  # RBF,gaussian_sei,masked
        for compare in for_comparison:
            if compare in rl:
                dataype.append(compare)
                datas.append(pd.read_csv(name_file, skiprows=2))
                break


mse = dict()
t_dist = dict()

t_dist_mean = dict()
t_dist_std = dict()
mse_mean = dict()
mse_std = dict()

mse_interp = dict()
mse_interp_mean = dict()
mse_interp_std = dict()

for compare in for_comparison:
    mse[compare] = []
    t_dist[compare] = []
    mse_mean[compare] = []
    t_dist_mean[compare] = []
    mse_std[compare] = []
    t_dist_std[compare] = []
    mse_interp[compare] = []
    mse_interp_mean[compare] = []
    mse_interp_std[compare] = []
# print(np.count_nonzero(np.array(dataype) == "gaussian_ei,masked"))
# print(np.count_nonzero(np.array(dataype) == "gaussian_ei,truncated"))
# print(np.count_nonzero(np.array(dataype) == "gaussian_ei,split_path"))
print(np.count_nonzero(np.array(dataype) == "truncated"))
print(np.count_nonzero(np.array(dataype) == "GA"))
print(np.count_nonzero(np.array(dataype) == "LM"))

max_dist = 0

for i in range(len(datas)):
    mse[dataype[i]].append(datas[i]["mse"].values)
    t_dist[dataype[i]].append(datas[i]["t_dist"].values)
    if t_dist[dataype[i]][-1][-1] > max_dist:
        max_dist = t_dist[dataype[i]][-1][-1]
    # if dataype[i] == 'truncated':
    #     plt.plot(t_dist[dataype[i]][-1], mse[dataype[i]][-1], '--b', alpha=0.1)
    # elif dataype[i] == 'GA':
    #     plt.plot(t_dist[dataype[i]][-1], mse[dataype[i]][-1], '--g', alpha=0.1)
    # elif dataype[i] == 'LM':
    #     plt.plot(t_dist[dataype[i]][-1], mse[dataype[i]][-1], '--r', alpha=0.1)
    # else:
    #     plt.plot(t_dist[dataype[i]][-1], mse[dataype[i]][-1], '--g', alpha=0.1)
from scipy import stats

#
for key in mse.keys():
    avgs = []
    for run in t_dist[key]:
        pdists = []
        for i in range(len(run) - 1):
            pdists.append(np.linalg.norm(run[i + 1] - run[i]))
        avgs.append(np.mean(pdists))
    print(key, "mode", stats.mode(avgs), "mean: ", np.mean(avgs), " std: ", np.std(avgs))

x = np.linspace(0, max_dist, np.round(max_dist).astype(int))

for compare in for_comparison:
    for tdistrun, mserun in zip(t_dist[compare], mse[compare]):
        mse_interp[compare].append(np.interp(x, tdistrun, mserun))

# minimum = 100000000000000
# method = 0
# name = "0"
for key in for_comparison:
#     i = 0
#     for inter_run in mse_interp[key]:
#         if np.sum(inter_run) < np.sum(minimum):
#             minimum = inter_run
#             method = i
#             name = key
#         i += 1
    mse_interp[key] = np.array(mse_interp[key]).T.reshape(len(x), -1)
    mse[key] = np.array(mse[key]).T.reshape(21, -1)
    # print(key)
    # print(method)
    # print(minimum)  # [0.78060036 0.77895489 0.77730941 ... 0.00144225 0.00144225 0.00144225]
    # minimum = 100000000000000

for key in for_comparison:
    mse_interp_mean[key] = np.mean(mse_interp[key], axis=1)
    mse_interp_std[key] = np.std(mse_interp[key], axis=1)

for compare in for_comparison:
    mse_mean[compare] = np.mean(mse[compare], axis=1)[:16]
    mse_std[compare] = np.std(mse[compare], axis=1)[:16]

legends = []
for key in for_comparison:
    # plt.fill_between(np.arange(0, 16), mse_mean[key] - mse_std[key] * 1.96,
    #                  mse_mean[key] + mse_std[key] * 1.96, alpha=0.2)
    # plt.plot(np.arange(0, 16), mse_mean[key], linewidth=5)
    legends.append(key)
# plt.show(block=True)
# plt.xlabel("Step")
# plt.ylabel("MSE")
# plt.legend(legends)
# plt.tight_layout()
# plt.show(block=True)
colors = ["#00629B", "#009CA6", "#78BE20", "#FFD100"]

# labels = np.arange(1, 16)
# width = 0.2  # the width of the ba
# i = 0
# for key in for_comparison:
#     plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width, mse_mean[key][1:], width, yerr=mse_std[key][1:],
#             label=key, color=colors[i])
#     #     plt.plot(labels + (i - len(for_comparison) / 2 + 0.5), mse_mean[key])
#     i += 1
# # Add some text for labels, title and custom x-axis tick labels, etc.
# plt.ylabel('MSE', fontsize=30)
# plt.xticks(labels, fontsize=30)
# plt.xlabel("Step", fontsize=30)
# plt.yticks(fontsize=30)
# plt.legend(["PI(x): $\\xi=1.0$", "EI(x): $\\xi=1.0$", "SEI(x): $\\xi=1.0$", "MVES(x): $\\xi=0.01$"],
#            prop={'size': 23})
# plt.tight_layout()
# plt.show(block=True)

# print('masked')
# print(mse_interp_mean["masked"][2499])
# print(mse_interp_std["masked"][2499] * 1.96)
# print('truncated')
# print(mse_interp_mean["truncated"][2499])
# print(mse_interp_std["truncated"][2499] * 1.96)
# print('split_path')
# print(mse_interp_mean["split_path"][2499])
# print(mse_interp_std["split_path"][2499] * 1.96)

# print('maskedxi')
# print(mse_interp_mean["maskedxi"][2499])
# print(mse_interp_std["maskedxi"][2499] * 1.96)
# print('truncatedxi')
# print(mse_interp_mean["truncatedxi"][2499])
# print(mse_interp_std["truncatedxi"][2499] * 1.96)
# print('split_pathxi')
# print(mse_interp_mean["split_pathxi"][2499])
# print(mse_interp_std["split_pathxi"][2499] * 1.96)
#
selected = np.arange(500, 3000, 250).astype(np.int)
width = 50  # the width of the ba
i = 0
for key in for_comparison:
    plt.bar(selected + (i - len(for_comparison) / 2 + 0.5) * width,
            mse_interp_mean[key][selected],
            width,
            yerr=mse_interp_std[key][selected], label=key, color=colors[i])
    # plt.plot(x + i * width, mse_interp_mean[key])

    # print(key)
    # print(mse_interp_mean[key][1500])
    # print(mse_interp_std[key][1500] * 1.96)

    i += 1

# idx = np.arange(0, 3700, 500).astype(np.int)
# for key in for_comparison:
#     for i in idx:

# for key in for_comparison:
#     plt.plot(x[:1750], mse_interp_mean[key][:1750])
#     # legends.append(key)
#     plt.fill_between(x[:1750], mse_interp_mean[key][:1750] - mse_interp_std[key][:1750] * 1.96,
#                      mse_interp_mean[key][:1750] + mse_interp_std[key][:1750] * 1.96, alpha=0.2)

plt.xlabel("Total Distance Travelled", fontsize=30)
plt.ylabel("MSE", fontsize=30)
# plt.title('Comparación por distancia recorrida')
plt.yticks(np.arange(0, 0.9, 0.1), fontsize=30)
plt.xticks(selected, fontsize=30)
# plt.legend(legends, prop={'size': 20})
# plt.legend(["tr-PI(x)", "tr-EI(x)", "tr-SEI(x)"], prop={'size': 30})
plt.legend(["BO", "GA", "LM"], prop={'size': 30})
plt.tight_layout()
plt.show()
