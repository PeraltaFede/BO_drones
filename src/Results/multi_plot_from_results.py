import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn")

name_files = glob.glob("E:/ETSI/Proyecto/results/SAMS/*.csv")
show = "var"

datas = []
dataype = []
# for_comparison = ["decoupled,2,predictive_entropy_search", "coupled,2,predictive_entropy_search",
#                   "decoupled,3,predictive_entropy_search", "coupled,3,predictive_entropy_search",
#                   "decoupled,4,predictive_entropy_search", "coupled,4,predictive_entropy_search"]
for_comparison = ["decoupled,2,gaussian_ei", "coupled,2,gaussian_ei", ]
#                   "decoupled,3,gaussian_ei", "coupled,3,gaussian_ei",
#                   "decoupled,4,gaussian_ei", "coupled,4,gaussian_ei"]
# for_comparison = ["2,coupled,2,gaussian_ei",
#                   "2,coupled,2,predictive_entropy_search",
#                   "3,coupled,3,gaussian_ei",
#                   "3,coupled,3,predictive_entropy_search",
#                   "4,coupled,4,gaussian_ei",
#                   "4,coupled,4,predictive_entropy_search"]

for name_file in name_files:
    with open(name_file, 'r') as f:
        f.readline()
        rl = f.readline()  # RBF,gaussian_sei,masked
        for compare in for_comparison:
            if compare in rl:
                dataype.append(compare)
                datas.append(pd.read_csv(name_file, skiprows=2))
                break

# name_files = glob.glob("E:/ETSI/Proyecto/results/multiagent/1dronms/*.csv")
# for_comparison = ["1,m"]
# for name_file in name_files:
#     with open(name_file, 'r') as f:
#         f.readline()
#         rl = f.readline()  # RBF,gaussian_sei,masked
#         for compare in for_comparison:
#             if compare in rl:
#                 dataype.append(compare)
#                 datas.append(pd.read_csv(name_file, skiprows=2))
#                 break
#
# name_files = glob.glob("E:/ETSI/Proyecto/results/multiagent/1dronnewms/*.csv")
#
# for_comparison = ["1,"]
#
# for name_file in name_files:
#     with open(name_file, 'r') as f:
#         f.readline()
#         rl = f.readline()  # RBF,gaussian_sei,masked
#         for compare in for_comparison:
#             if compare in rl:
#                 dataype.append(compare)
#                 datas.append(pd.read_csv(name_file, skiprows=2))
#                 break
#
# for_comparison = ["1",
#                   "1,m",
#                   "1,"]
for compare in for_comparison:
    print(compare, ": ", np.count_nonzero(np.array(dataype) == compare))

mse = dict()
qty = dict()
score = dict()
variance = dict()
mse_clean = dict()
qty_clean = dict()
score_clean = dict()
variance_clean = dict()
max4key = dict()
mse_mean = dict()
mse_std = dict()
variance_mean = dict()
score_mean = dict()
score_std = dict()
time = dict()
time_clean = dict()
time_mean = dict()
time_std = dict()

for compare in for_comparison:
    mse[compare] = []
    mse_clean[compare] = []
    mse_mean[compare] = []
    mse_std[compare] = []
    qty[compare] = []
    qty_clean[compare] = []
    score[compare] = []
    score_clean[compare] = []
    score_mean[compare] = []
    score_std[compare] = []
    max4key[compare] = -1
    time[compare] = []
    time_clean[compare] = []
    time_mean[compare] = []
    time_std[compare] = []
    variance[compare] = []
    variance_clean[compare] = []
    variance_mean[compare] = []

for i in range(len(datas)):
    mse[dataype[i]].append(datas[i]["avg_mse"].values)
    score[dataype[i]].append(datas[i]["avg_score"].values)
    qty[dataype[i]].append(datas[i]["qty"].values)
    time[dataype[i]].append(datas[i]["time"].values)

    scores = [datas[i][x].values for x in datas[i].columns if "score_" in x]
    # for x in datas[i].columns:
    #     if "score_" in x:
    #         scores += datas[i][x].values
    variance[dataype[i]].append(np.var(scores, axis=0))

for key in for_comparison:
    for run, meas, tim, sco, var in zip(mse[key], qty[key], time[key], score[key], variance[key]):
        muestra, indice = np.unique(meas, return_index=True)
        mse_clean[key].append(list(run[indice.astype(int)]))
        score_clean[key].append(list(sco[indice.astype(int)]))
        variance_clean[key].append(list(var[indice.astype(int)]))
        qty_clean[key].append(list(muestra))
        time_clean[key].append(list(tim[indice]))
        if max4key[key] < len(indice):
            max4key[key] = len(indice)

    for i in range(len(mse_clean[key])):
        if max4key[key] > len(mse_clean[key][i]):
            mse_clean[key][i].extend(list(np.full(max4key[key] - len(mse_clean[key][i]), np.nan)))
            score_clean[key][i].extend(list(np.full(max4key[key] - len(score_clean[key][i]), np.nan)))
            variance_clean[key][i].extend(list(np.full(max4key[key] - len(variance_clean[key][i]), np.nan)))
            qty_clean[key][i].extend(list(np.full(max4key[key] - len(qty_clean[key][i]), np.nan)))
            time_clean[key][i].extend(list(np.full(max4key[key] - len(time_clean[key][i]), np.nan)))

# minimum = 100000000000000
# method = 0
# name = "0"
for key in for_comparison:
    # i = 0
    # for inter_run in mse_clean[key]:
    #     if np.sum(inter_run) < np.sum(minimum):
    #         minimum = inter_run
    #         method = i
    #         name = key
    #     i += 1
    mse_clean[key] = np.array(mse_clean[key]).T.reshape(max4key[key], -1)
    score_clean[key] = np.array(score_clean[key]).T.reshape(max4key[key], -1)
    variance_clean[key] = np.array(variance_clean[key]).T.reshape(max4key[key], -1)
    qty_clean[key] = np.array(qty_clean[key]).T.reshape(max4key[key], -1)
    time_clean[key] = np.array(time_clean[key]).T.reshape(max4key[key], -1)
    # print(key)
    # print(method)
    # print(minimum)
    # minimum = 100000000000000

for compare in for_comparison:
    mse_mean[compare] = np.nanmean(mse_clean[compare], axis=1)
    mse_std[compare] = np.nanstd(mse_clean[compare], axis=1)
    score_mean[compare] = np.nanmean(score_clean[compare], axis=1)
    score_std[compare] = np.nanstd(score_clean[compare], axis=1)
    time_mean[compare] = np.nanmean(time_clean[compare], axis=1)
    time_std[compare] = np.nanstd(time_clean[compare], axis=1)
    variance_mean[compare] = np.nanmean(variance_clean[compare], axis=1)
    # print(compare)
    # print(mse_mean[compare])
    # print(mse_std[compare])
    # print(time_mean[compare])
    # print(time_std[compare])
    # print('--------------------')
    # print(qty_clean[compare])

# print((-mse_mean["2"][-1] + mse_mean["1"][-1]) / mse_mean["1"][-1])
# print((-mse_mean["3"][-1] + mse_mean["2"][-1]) / mse_mean["2"][-1])
# print((-mse_mean["4"][-1] + mse_mean["3"][-1]) / mse_mean["3"][-1])
# print((-mse_mean["5"][-1] + mse_mean["4"][-1]) / mse_mean["4"][-1])
#
# print(((-mse_mean["2"][-1] + mse_mean["1"][-1]) / mse_mean["1"][-1] + (-mse_mean["3"][-1] + mse_mean["2"][-1]) /
#        mse_mean["2"][-1] +
#        (-mse_mean["4"][-1] + mse_mean["3"][-1]) / mse_mean["3"][-1] + (-mse_mean["5"][-1] + mse_mean["4"][-1]) /
#        mse_mean["4"][-1]) / 4
#       )
#
# print(mse_mean["2"][-1] / mse_mean["1"][-1])
# print(mse_mean["3"][-1] / mse_mean["2"][-1])
# print(mse_mean["4"][-1] / mse_mean["3"][-1])
# print(mse_mean["5"][-1] / mse_mean["4"][-1])
#
# print((
#               mse_mean["2"][-1] / mse_mean["1"][-1] +
#               mse_mean["3"][-1] / mse_mean["2"][-1] +
#               mse_mean["4"][-1] / mse_mean["3"][-1] +
#               mse_mean["5"][-1] / mse_mean["4"][-1]
#
#       ) / 4
#       )
legends = []
for key in for_comparison:
    legends.append(key)

# colors = ["#FFD100", "#FFD100AA"]
# colors = ["#00629B", "#009CA6", "#78BE20", "#FFD100"]
colors = ["#3B4D77", "#C09235", "#B72F56", "#91B333", "#00629B", "#009CA6", "#78BE20", "#FFD100", "#3B4D77", "#C09235",
          "#B72F56", "#91B333"]
width = 0.4  # the width of the ba
i = 0
key = for_comparison[-1]

if "mse" in show:
    plt.figure()
    for key in for_comparison:
        labels = np.arange(qty_clean[key][0][0], max4key[key] + qty_clean[key][0][0])
        # print(labels + (i - len(for_comparison) / 2 + 0.5) * width)
        # print(mse_mean[key])
        # print(mse_std[key])
        plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width, mse_mean[key], width,
                yerr=mse_std[key],
                label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')), color=colors[i])
        # label="n_sensors: {0}, fusion: {1}, acq: {3}".format(*key.split(',')), color=colors[i])
        #     plt.plot(labels + (i - len(for_comparison) / 2 + 0.5), mse_mean[key])
        i += 1
    # print('yes')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('MSE', fontsize=30)
    plt.xticks(np.arange(0, max4key[key] + qty_clean[key][0][0]), fontsize=30)
    # plt.xticks(fontsize=30)
    plt.xlabel("Measurements", fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("MSE", fontsize=30)
    # plt.legend(["realnew", "old", "new"], prop={'size': 23})
    plt.legend(prop={'size': 13}, fancybox=True, shadow=True, frameon=True)
    # plt.tight_layout()
i = 0
if "score" in show:
    plt.figure()
    for key in for_comparison:
        labels = np.arange(qty_clean[key][0][0], max4key[key] + qty_clean[key][0][0])
        plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width, score_mean[key], width,
                yerr=score_std[key],
                label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')), color=colors[i])
        # label="n_sensors: {0}, fusion: {1}, acq: {3}".format(*key.split(',')), color=colors[i])
        i += 1
    plt.ylabel('R2Score', fontsize=30)
    plt.xticks(np.arange(0, max4key[key] + qty_clean[key][0][0]), fontsize=30)
    plt.xlabel("Measurements", fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("R2Score", fontsize=30)
    plt.legend(prop={'size': 13}, fancybox=True, shadow=True, frameon=True)
i = 0
if "var" in show:
    plt.figure()
    for key in for_comparison:
        labels = np.arange(qty_clean[key][0][0], max4key[key] + qty_clean[key][0][0])
        plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width, variance_mean[key], width,
                label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')), color=colors[i])
        # label="n_sensors: {0}, fusion: {1}, acq: {3}".format(*key.split(',')), color=colors[i])
        i += 1
    plt.ylabel('Variance', fontsize=30)
    plt.xticks(np.arange(0, max4key[key] + qty_clean[key][0][0]), fontsize=30)
    plt.xlabel("Measurements", fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("Var(x)", fontsize=30)
    plt.legend(prop={'size': 13}, fancybox=True, shadow=True, frameon=True)

i = 0
if "time" in show:
    plt.figure()
    for key in for_comparison:
        labels = np.arange(qty_clean[key][0][0], max4key[key] + qty_clean[key][0][0])
        # print(labels + (i - len(for_comparison) / 2 + 0.5) * width)
        # print(mse_mean[key])
        # print(mse_std[key])
        plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width, time_mean[key], width,
                yerr=time_std[key],
                label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')), color=colors[i])
        # label="n_sensors: {0}, fusion: {1}, acq: {3}".format(*key.split(',')), color=colors[i])
        #     plt.plot(labels + (i - len(for_comparison) / 2 + 0.5), mse_mean[key])
        i += 1
    # print('yes')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Time [s]', fontsize=30)
    plt.xticks(np.arange(0, max4key[key] + qty_clean[key][0][0]), fontsize=30)
    # plt.xticks(fontsize=30)
    plt.xlabel("Measurements", fontsize=30)
    plt.yticks(fontsize=30)
    # plt.title("4 drones", fontsize=30)
    # plt.legend(["realnew", "old", "new"], prop={'size': 23})
    plt.legend(prop={'size': 13}, fancybox=True, shadow=True, frameon=True)

plt.show(block=True)
