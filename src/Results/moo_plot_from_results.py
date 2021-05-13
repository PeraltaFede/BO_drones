import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn")
show = "dist"
for_comparison = ["0.38466", "0.40466", "0.42466", "0.44466"]

datas = []
dataype = []

name_files = glob.glob("E:/ETSI/Proyecto/results/SAMS/*.csv")

for name_file in name_files:
    with open(name_file, 'r') as f:
        f.readline()
        rl = f.readline()  # RBF,gaussian_sei,masked
        rest_all_lines = f.readlines()
        flag = False
        for line in rest_all_lines:
            if "pos:" in line:
                flag = True
                print(name_file)
                break
        if flag:
            continue
        for compare in for_comparison:
            if compare in rl:
                dataype.append(compare)
                datas.append(pd.read_csv(name_file, skiprows=2))
                break
# for_comparison = ["predictive_entropy_search"]
#
# name_files = glob.glob("E:/ETSI/Proyecto/results/pes/*.csv")
# for name_file in name_files:
#     with open(name_file, 'r') as f:
#         f.readline()
#         rl = f.readline()  # RBF,gaussian_sei,masked
#         rest_all_lines = f.readlines()
#         flag = False
#         for line in rest_all_lines:
#             if "pos:" in line:
#                 flag = True
#                 break
#         if flag:
#             continue
#         for compare in for_comparison:
#             if compare in rl:
#                 dataype.append(compare)
#                 # dataype.append(f"old,{compare[1:]}")
#                 datas.append(pd.read_csv(name_file, skiprows=2))
#                 break
# for_comparison = ["max_sum"]
#
# name_files = glob.glob("E:/ETSI/Proyecto/results/ga/*.csv")
# for name_file in name_files:
#     with open(name_file, 'r') as f:
#         f.readline()
#         rl = f.readline()  # RBF,gaussian_sei,masked
#         rest_all_lines = f.readlines()
#         flag = False
#         for line in rest_all_lines:
#             if "pos:" in line:
#                 flag = True
#                 break
#         if flag:
#             continue
#         for compare in for_comparison:
#             if compare in rl:
#                 dataype.append("ga")
#                 # dataype.append(f"old,{compare[1:]}")
#                 datas.append(pd.read_csv(name_file, skiprows=2))
#                 break
# for_comparison = ["pareto", "predictive_entropy_search","ga"]

for compare in for_comparison:
    print(compare, ": ", np.count_nonzero(np.array(dataype) == compare))

qty = dict()
score = dict()
variance = dict()
qty_clean = dict()
score_clean = dict()
variance_clean = dict()
max4key = dict()
mean4key = dict()
variance_mean = dict()
score_mean = dict()
score_std = dict()
time = dict()
time_clean = dict()
time_mean = dict()
time_std = dict()

t_dist = dict()
mse_interp = dict()
mse_interp_mean = dict()
mse_interp_std = dict()
t_dist_clean = dict()
t_dist_mean = dict()
t_dist_std = dict()

for compare in for_comparison:
    qty[compare] = []
    qty_clean[compare] = []
    score[compare] = []
    score_clean[compare] = []
    score_mean[compare] = []
    score_std[compare] = []
    max4key[compare] = -1
    mean4key[compare] = 0
    time[compare] = []
    time_clean[compare] = []
    time_mean[compare] = []
    time_std[compare] = []
    variance[compare] = []
    variance_clean[compare] = []
    variance_mean[compare] = []

    t_dist[compare] = []
    mse_interp[compare] = []
    mse_interp_mean[compare] = []
    mse_interp_std[compare] = []
    t_dist_clean[compare] = []
    t_dist_mean[compare] = []
    t_dist_std[compare] = []

max_dist = 0
for i in range(len(datas)):
    score[dataype[i]].append(datas[i]["avg_score"].values)  # /0.7413447473233803
    scores = [datas[i][x].values for x in datas[i].columns if "score_" in x]
    variance[dataype[i]].append(np.var(scores, axis=0))
    qty[dataype[i]].append(datas[i]["qty"].values)
    time[dataype[i]].append(datas[i]["time"].values)
    t_dist[dataype[i]].append(datas[i]["t_dist"].values)
    if t_dist[dataype[i]][-1][-1] > max_dist:
        max_dist = t_dist[dataype[i]][-1][-1]

max_dist = np.round(max_dist).astype(np.int)
for key in for_comparison:
    for meas, tim, sco, var, dis in zip(qty[key], time[key], score[key], variance[key], t_dist[key]):
        muestra, indice = np.unique(meas, return_index=True)
        score_clean[key].append(list(sco[indice.astype(int)]))
        variance_clean[key].append(list(var[indice.astype(int)]))
        qty_clean[key].append(list(muestra))
        time_clean[key].append(list(tim[indice]))
        t_dist_clean[key].append(list(dis[indice]))
        mean4key[key] += len(qty_clean[key][-1])

        if max4key[key] < len(indice):
            max4key[key] = len(indice)

    for i in range(len(qty_clean[key])):
        if max4key[key] > len(qty_clean[key][i]):
            score_clean[key][i].extend(list(np.full(max4key[key] - len(score_clean[key][i]), np.nan)))
            variance_clean[key][i].extend(list(np.full(max4key[key] - len(variance_clean[key][i]), np.nan)))
            qty_clean[key][i].extend(list(np.full(max4key[key] - len(qty_clean[key][i]), np.nan)))
            time_clean[key][i].extend(list(np.full(max4key[key] - len(time_clean[key][i]), np.nan)))
            t_dist_clean[key][i].extend(list(np.full(max4key[key] - len(t_dist_clean[key][i]), np.nan)))
    mean4key[key] /= len(qty_clean[key])

for key in for_comparison:
    score_clean[key] = np.array(score_clean[key]).T.reshape(max4key[key], -1)
    variance_clean[key] = np.array(variance_clean[key]).T.reshape(max4key[key], -1)
    qty_clean[key] = np.array(qty_clean[key]).T.reshape(max4key[key], -1)
    time_clean[key] = np.array(time_clean[key]).T.reshape(max4key[key], -1)
    t_dist_clean[key] = np.array(t_dist_clean[key]).T.reshape(max4key[key], -1)

for compare in for_comparison:
    score_mean[compare] = np.nanmean(score_clean[compare], axis=1)
    score_std[compare] = np.nanstd(score_clean[compare], axis=1)
    time_mean[compare] = np.nanmean(time_clean[compare], axis=1)
    time_std[compare] = np.nanstd(time_clean[compare], axis=1)
    variance_mean[compare] = np.nanmean(variance_clean[compare], axis=1)
    t_dist_mean[compare] = np.nanmean(t_dist_clean[compare], axis=1)
    t_dist_std[compare] = np.nanstd(t_dist_clean[compare], axis=1)

legends = []
for key in for_comparison:
    legends.append(key)
# colors = ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
#           "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"]
# colors = ["#A6CEE3", "#1F78B4", "#33A02C",
#           "#FF7F00", "#6A3D9A", "#B15928"]
# colors = ["#1F78B4", "#B15928", "#33A02C",
#           "#FF7F00", "#6A3D9A", "#A6CEE3"]
colors = ["#EC9B2E", "#24AAE2", "#73934B"]
titles = for_comparison
# titles = ["Proposed", "PESMOC", "GA"]
if "dist" in show:
    x = np.linspace(0, max_dist, max_dist+1)

    selected = np.arange(250, max_dist+130, 250).astype(np.int)
    width = 90 / len(for_comparison)  # the width of the ba
    i = 0
    for key in for_comparison:
        max_r2s = -1
        fmo4mr2s = -1
        for tdistrun, mserun, idx in zip(t_dist[key], score[key], enumerate(score[key])):
            fmo = np.where(mserun == -1)[0]
            if len(fmo) > 0:
                mse_interp[key].append(np.interp(x, tdistrun[:fmo[0]], mserun[:fmo[0]]))
                # if (max_r2s == -1 or mserun[fmo[0] - 1] > score[key][max_r2s][fmo4mr2s - 1]):
                #     max_r2s = idx[0]
                #     fmo4mr2s = fmo[0]
                    # print(max_r2s)
                    # print(fmo4mr2s)
                # plt.plot(tdistrun[:fmo[0]], mserun[:fmo[0]], color=colors[i], alpha=0.1)
                # plt.plot(tdistrun[fmo[0] - 1], mserun[fmo[0] - 1], '.', color=colors[i])
            else:
                mse_interp[key].append(np.interp(x, tdistrun, mserun))
                if max_r2s == -1 or mserun[- 1] > score[key][max_r2s][-1]:
                    max_r2s = np.where(score[key] == mserun)
                # plt.plot(tdistrun, mserun, color=colors[i], alpha=0.1)
        print(key, max_r2s, fmo4mr2s, score[key][max_r2s][fmo4mr2s - 1])
        print(score[key][max_r2s])
        mse_interp[key] = np.array(mse_interp[key]).T.reshape(len(x), -1)
        mse_interp_mean[key] = np.mean(mse_interp[key], axis=1)
        mse_interp_std[key] = np.std(mse_interp[key], axis=1)
        plt.plot(x, mse_interp_mean[key], label=key, color=colors[i])
        plt.fill_between(x, mse_interp_mean[key] - mse_interp_std[key],
                         mse_interp_mean[key] + mse_interp_std[key], alpha=0.2, color=colors[i])
        # plt.bar(selected + (i - len(for_comparison) / 2 + 0.5) * width,
        #         mse_interp_mean[key][selected],
        #         width,
        #         yerr=mse_interp_std[key][selected],
        #         label=key, color=colors[i])
        # print(key, mse_interp_mean[key][-1])
        i += 1

    plt.ylabel('$R^2(x)$', fontsize=30)
    plt.xticks(selected, [str(format(d * 10, ',')) for d in selected], fontsize=30)
    plt.xlabel("Distance (m)", fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("$R^2(x)$ Score vs Distance", fontsize=30)
    plt.legend(titles, loc='upper left', prop={'size': 25}, fancybox=True, shadow=True,
               frameon=True)

# colors = ["#FFD100", "#FFD100AA"]
# colors = ["#00629B", "#009CA6", "#78BE20", "#FFD100"]
width = 0.95 / len(for_comparison)  # the width of the ba
i = 0
key = for_comparison[-1]
if "score" in show:
    plt.figure()
    for key in for_comparison:
        labels = np.arange(qty_clean[key][0][0], max4key[key] + qty_clean[key][0][0])
        # print(key, np.max(score_mean[key]))
        plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width,
                score_mean[key],
                width,
                yerr=score_std[key],
                color=colors[i],
                label=key)
        # label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')))
        # label="n_sensors: {0}, fusion: {1}, acq: {3}".format(*key.split(',')), color=colors[i])
        i += 1
    plt.ylabel('$R^2(x)$', fontsize=30)
    plt.xticks(np.arange(0, max4key[key] + qty_clean[key][0][0]), fontsize=30)
    plt.xlabel("Measurements", fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("$R^2(x)$ Score vs Measurements", fontsize=30)
    plt.legend(titles, prop={'size': 25}, fancybox=True, shadow=True, frameon=True)
i = 0
if "var" in show:
    plt.figure()
    for key in for_comparison:
        labels = np.arange(qty_clean[key][0][0], max4key[key] + qty_clean[key][0][0])
        plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width, variance_mean[key], width, color=colors[i],
                label=key)
        # label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')))
        # label="n_sensors: {0}, fusion: {1}, acq: {3}".format(*key.split(',')), color=colors[i])
        i += 1
    plt.ylabel('$var(x)$', fontsize=30)
    plt.xticks(np.arange(0, max4key[key] + qty_clean[key][0][0]), fontsize=30)
    plt.xlabel("Measurements", fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("Variance of different $R^2(x)$ Scores", fontsize=30)
    plt.legend(titles, loc='upper right', prop={'size': 25}, fancybox=True, shadow=True,
               frameon=True)
i = 0
if "time" in show:
    plt.figure()
    for key in for_comparison:
        labels = np.arange(qty_clean[key][0][0], max4key[key] + qty_clean[key][0][0])
        # print(labels + (i - len(for_comparison) / 2 + 0.5) * width)
        # print(mse_mean[key])
        # print(key, np.mean(time_mean[key]))
        plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width, time_mean[key], width,
                yerr=time_std[key], color=colors[i],
                label=key)
        # label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')))
        # label="n_sensors: {0}, fusion: {1}, acq: {3}".format(*key.split(',')), color=colors[i])
        #     plt.plot(labels + (i - len(for_comparison) / 2 + 0.5), mse_mean[key])
        i += 1
    # print('yes')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Time (s)', fontsize=30)
    plt.xticks(np.arange(0, max4key[key] + qty_clean[key][0][0]), fontsize=30)
    # plt.xticks(fontsize=30)
    plt.xlabel("Measurements", fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("Computational Time", fontsize=30)
    # plt.legend(["realnew", "old", "new"], prop={'size': 23})
    plt.legend(titles, loc='upper left', prop={'size': 25}, fancybox=True, shadow=True,
               frameon=True)
plt.show(block=True)
