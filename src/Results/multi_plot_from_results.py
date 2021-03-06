import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn")

name_files = glob.glob("E:/ETSI/Proyecto/results/SAMS/old/*.csv")
show = "dist,score,var,time"

datas = []
dataype = []
# for_comparison = ["decoupled,2,predictive_entropy_search", "coupled,2,predictive_entropy_search",
#                   "decoupled,3,predictive_entropy_search", "coupled,3,predictive_entropy_search",
#                   "decoupled,4,predictive_entropy_search", "coupled,4,predictive_entropy_search"]
# for_comparison = [   "decoupled,2,gaussian_ei", "coupled,2,gaussian_ei", ]
# "decoupled,3,gaussian_ei", "coupled,3,gaussian_ei",
# "decoupled,4,gaussian_ei", "coupled,4,gaussian_ei"]
# for_comparison = ["2,coupled", "3,coupled", "4,coupled",
#                   "2,decoupled", "3,decoupled", "4,decoupled"]
for_comparison = ["5,decoupled"]
# for_comparison = ["decoupled,2,gaussian_ei", "coupled,2,gaussian_ei"]
#                   "3,coupled,3,gaussian_ei",
#                   "3,coupled,3,predictive_entropy_search",
#                   "4,coupled,4,gaussian_ei",
#                   "4,coupled,4,predictive_entropy_search"]

# for name_file in name_files:
#     with open(name_file, 'r') as f:
#         f.readline()
#         rl = f.readline()  # RBF,gaussian_sei,masked
#         for compare in for_comparison:
#             if compare in rl:
#                 if "0.5" in rl:
#                     continue
#                     dataype.append(f"0.50,{compare}")
#                 elif "0.75" in rl:
#                     continue
#                     dataype.append(f"0.75,{compare}")
#                 elif "0.25" in rl:
#                     continue
#                     dataype.append(f"ei,{compare}")
#                     # dataype.append(f"0.25,{compare}")
#                 elif "0.375" in rl:
#                     continue
#                     dataype.append(f"0.375,{compare}")
#                 elif "0.125" in rl:
#                     # continue
#                     dataype.append(f"old,0.125,{compare[2:]}")
#                 else:
#                     continue
#                     dataype.append(f"1.00,{compare}")
#                 # dataype.append(compare)
#                 datas.append(pd.read_csv(name_file, skiprows=2))
#                 break

name_files = glob.glob("E:/ETSI/Proyecto/results/SAMS/*.csv")
for_comparison = ["5,decoupled", "5,coupled"]

for name_file in name_files:
    with open(name_file, 'r') as f:
        f.readline()
        rl = f.readline()  # RBF,gaussian_sei,masked
        for compare in for_comparison:
            if compare in rl:
                dataype.append(f"new,0.125,{compare[2:]}")
                datas.append(pd.read_csv(name_file, skiprows=2))
                break

# name_files = glob.glob("E:/ETSI/Proyecto/results/SAMS/pes/*.csv")
#
# for name_file in name_files:
#     with open(name_file, 'r') as f:
#         f.readline()
#         rl = f.readline()  # RBF,gaussian_sei,masked
#         for compare in for_comparison:
#             if compare in rl:
#                 dataype.append(f"pes,{compare}")
#                 datas.append(pd.read_csv(name_file, skiprows=2))
#                 break

# for_comparison = [
#     "110decoupled,2,gaussian_ei", "110coupled,2,gaussian_ei",
#     # "110decoupled,3,gaussian_ei", "110coupled,3,gaussian_ei",
#     # "110decoupled,4,gaussian_ei", "110coupled,4,gaussian_ei",
#     # "220decoupled,2,gaussian_ei", "220coupled,2,gaussian_ei",
#     # "220decoupled,3,gaussian_ei", "220coupled,3,gaussian_ei",
#     # "220decoupled,4,gaussian_ei", "220coupled,4,gaussian_ei",
#     "dyndecoupled,2,gaussian_ei", "dyncoupled,2,gaussian_ei",]
#     # "dyndecoupled,3,gaussian_ei", "dyncoupled,3,gaussian_ei",
# "dyndecoupled,4,gaussian_ei", "dyncoupled,4,gaussian_ei", ]
# for_comparison = [
#     "110decoupled", "110coupled",
#     # "220decoupled", "220coupled",
#     "dyndecoupled", "dyncoupled"]

# for_comparison = ["1.00,2,coupled", "1.00,3,coupled", "1.00,4,coupled",
#                   "1.00,2,decoupled", "1.00,3,decoupled", "1.00,4,decoupled",
#                   "0.75,2,coupled", "0.75,3,coupled", "0.75,4,coupled",
#                   "0.75,2,decoupled", "0.75,3,decoupled", "0.75,4,decoupled",
#                   "0.50,2,coupled", "0.50,3,coupled", "0.50,4,coupled",
#                   "0.50,2,decoupled", "0.50,3,decoupled", "0.50,4,decoupled",
#                   "0.25,2,coupled", "0.25,3,coupled", "0.25,4,coupled",
#                   "0.25,2,decoupled", "0.25,3,decoupled", "0.25,4,decoupled"]


# for_comparison = ["0.25,2,coupled", "0.25,3,coupled", "0.25,4,coupled",
#                   "0.25,2,decoupled", "0.25,3,decoupled", "0.25,4,decoupled",
#                   "0.50,2,coupled", "0.50,3,coupled", "0.50,4,coupled",
#                   "0.50,2,decoupled", "0.50,3,decoupled", "0.50,4,decoupled"
#                   ]
#
# for_comparison = [
#     "1.00,coupled",
#     "1.00,decoupled",
#     "0.75,coupled",
#     "0.75,decoupled",
#     "0.50,coupled",
#     "0.50,decoupled",
#     "0.375,coupled",
#     "0.375,decoupled",
#     "0.25,coupled",
#     "0.25,decoupled",
#     "0.125,coupled",
#     "0.125,decoupled", ]

for_comparison = [
    "new,0.125,decoupled",
    "new,0.125,coupled",
]

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

t_dist = dict()
mse_interp = dict()
mse_interp_mean = dict()
mse_interp_std = dict()
t_dist_clean = dict()
t_dist_mean = dict()
t_dist_std = dict()

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

    t_dist[compare] = []
    mse_interp[compare] = []
    mse_interp_mean[compare] = []
    mse_interp_std[compare] = []
    t_dist_clean[compare] = []
    t_dist_mean[compare] = []
    t_dist_std[compare] = []

max_dist = 0
for i in range(len(datas)):
    mse[dataype[i]].append(datas[i]["avg_mse"].values)
    score[dataype[i]].append(datas[i]["avg_score"].values)
    qty[dataype[i]].append(datas[i]["qty"].values)
    time[dataype[i]].append(datas[i]["time"].values)
    t_dist[dataype[i]].append(datas[i]["t_dist"].values)
    if t_dist[dataype[i]][-1][-1] > max_dist:
        max_dist = t_dist[dataype[i]][-1][-1]
    scores = [datas[i][x].values for x in datas[i].columns if "score_" in x]
    variance[dataype[i]].append(np.var(scores, axis=0))

for key in for_comparison:
    for run, meas, tim, sco, var, dis in zip(mse[key], qty[key], time[key], score[key], variance[key], t_dist[key]):
        muestra, indice = np.unique(meas, return_index=True)
        mse_clean[key].append(list(run[indice.astype(int)]))
        score_clean[key].append(list(sco[indice.astype(int)]))
        variance_clean[key].append(list(var[indice.astype(int)]))
        qty_clean[key].append(list(muestra))
        time_clean[key].append(list(tim[indice]))
        t_dist_clean[key].append(list(dis[indice]))

        if max4key[key] < len(indice):
            max4key[key] = len(indice)

    for i in range(len(mse_clean[key])):
        if max4key[key] > len(mse_clean[key][i]):
            mse_clean[key][i].extend(list(np.full(max4key[key] - len(mse_clean[key][i]), np.nan)))
            score_clean[key][i].extend(list(np.full(max4key[key] - len(score_clean[key][i]), np.nan)))
            variance_clean[key][i].extend(list(np.full(max4key[key] - len(variance_clean[key][i]), np.nan)))
            qty_clean[key][i].extend(list(np.full(max4key[key] - len(qty_clean[key][i]), np.nan)))
            time_clean[key][i].extend(list(np.full(max4key[key] - len(time_clean[key][i]), np.nan)))
            t_dist_clean[key][i].extend(list(np.full(max4key[key] - len(t_dist_clean[key][i]), np.nan)))
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
    t_dist_clean[key] = np.array(t_dist_clean[key]).T.reshape(max4key[key], -1)
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
    t_dist_mean[compare] = np.nanmean(t_dist_clean[compare], axis=1)
    t_dist_std[compare] = np.nanstd(t_dist_clean[compare], axis=1)

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

colors = ["#FFAA00", "#3BAA77", "#B72F56", "#91B333", "#00629B", "#009CA6",
          "#78AA20", "#3B4D77", "#C09235", "#C0AA35", "#B7AA56", "#91AA33"]
if "dist" in show:
    x = np.linspace(0, max_dist, np.round(max_dist).astype(int))
    # print(np.nanmax(t_dist_mean["0.125,coupled"]))
    # print(np.nanmax(t_dist["0.125,coupled"]))
    # print(np.nanmax(t_dist_mean["0.125,decoupled"]))
    # print(np.nanmax(t_dist["0.125,decoupled"]))
    # for key in for_comparison:
    #     print(key)
    #     for a in [mse[key], score[key], variance[key], time[key]]:
    #         for tdistrun, mserun in zip(t_dist[key], a):
    #             fmo = np.where(mserun == -1)[0]
    #             # print(fmo)
    #             if len(fmo) > 0:
    #                 mse_interp[key].append(np.interp(x, tdistrun[:fmo[0]], mserun[:fmo[0]]))
    #             else:
    #                 mse_interp[key].append(np.interp(x, tdistrun, mserun))
    #         mse_interp[key] = np.array(mse_interp[key]).T.reshape(len(x), -1)
    #         mse_interp_mean[key] = np.mean(mse_interp[key], axis=1)
    #         mse_interp_std[key] = np.std(mse_interp[key], axis=1)
    #         print(f"${np.round(mse_interp_mean[key][1500], 4)} \pm {np.round(mse_interp_std[key][1500], 4)}$ & ")
    #         mse_interp[key] = []

    selected = np.arange(250, max_dist, 250).astype(np.int)
    width = 90 / len(for_comparison)  # the width of the ba
    i = 0
    for key in for_comparison:
        # mse = dict()
        # qty = dict()
        # score = dict()
        #  = dict()
        for tdistrun, mserun in zip(t_dist[key], mse[key]):
            fmo = np.where(mserun == -1)[0]
            if len(fmo) > 0:

                mse_interp[key].append(np.interp(x, tdistrun[:fmo[0]], mserun[:fmo[0]]))
                plt.plot(tdistrun[:fmo[0]], mserun[:fmo[0]], color=colors[i], alpha=0.1)
                # plt.plot(tdistrun[fmo[0] - 1], mserun[fmo[0] - 1], '.', color=colors[i])
            else:
                mse_interp[key].append(np.interp(x, tdistrun, mserun))
                plt.plot(tdistrun, mserun, color=colors[i], alpha=0.1)
        mse_interp[key] = np.array(mse_interp[key]).T.reshape(len(x), -1)
        mse_interp_mean[key] = np.mean(mse_interp[key], axis=1)
        mse_interp_std[key] = np.std(mse_interp[key], axis=1)
        plt.plot(x, mse_interp_mean[key], label=key, color=colors[i])
        # plt.fill_between(x, mse_interp_mean[key] - mse_interp_std[key],
        #                  mse_interp_mean[key] + mse_interp_std[key], alpha=0.2, color=colors[i])

        # plt.bar(selected + (i - len(for_comparison) / 2 + 0.5) * width,
        #         mse_interp_mean[key][selected],
        #         width,
        #         yerr=mse_interp_std[key][selected], label=key, color=colors[i])
        i += 1

    plt.ylabel('MSE', fontsize=30)
    plt.xticks(selected, fontsize=30)
    plt.xlabel("Distance", fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("MSE", fontsize=30)
    plt.legend(prop={'size': 13}, fancybox=True, shadow=True, frameon=True)

# colors = ["#FFD100", "#FFD100AA"]
# colors = ["#00629B", "#009CA6", "#78BE20", "#FFD100"]
width = 0.9 / len(for_comparison)  # the width of the ba
i = 0
key = for_comparison[-1]

c_proportion_of_l_s = []
d_proportion_of_l_s = []
c_ratio_score_t_d = []
d_ratio_score_t_d = []

if "mse" in show:
    plt.figure()
    for key in for_comparison:
        # print(key)
        # print(np.mean(score_mean[key][1:] / t_dist_mean[key][1:]))
        if "decoupled" in key:
            # d_proportion_of_l_s.append(1 / float(key[:4]))
            d_proportion_of_l_s.append(float(key[:4]))
            # d_ratio_score_t_d.append(mse_interp_mean[key][1500])
            d_ratio_score_t_d.append(np.mean(mse_mean[key][1:] / t_dist_mean[key][1:]))
        else:
            # c_proportion_of_l_s.append(1 / float(key[:4]))
            c_proportion_of_l_s.append(float(key[:4]))
            # c_ratio_score_t_d.append(mse_interp_mean[key][1500])
            c_ratio_score_t_d.append(np.mean(mse_mean[key][1:] / t_dist_mean[key][1:]))
        # print(np.mean(variance_mean[key][1:] / t_dist_mean[key][1:]))
        # print(mse_mean[key][-1]/t_dist_mean[key][-1])
        # print(variance_mean[key][-1]/t_dist_mean[key][-1])
        # print(np.mean(time_mean[key][1:]))
        labels = np.arange(qty_clean[key][0][0], max4key[key] + qty_clean[key][0][0])
        # print(labels + (i - len(for_comparison) / 2 + 0.5) * width)
        # print(mse_mean[key])
        # print(mse_std[key])
        # plt.bar(labels[1:] + (i - len(for_comparison) / 2 + 0.5) * width, score_mean[key][1:] / t_dist_mean[key][1:],
        #         width,
        #         # yerr=t_dist_std[key],
        #         color=colors[i],
        #         label=key)
        # label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')))
        # label="n_sensors: {0}, fusion: {1}, acq: {3}".format(*key.split(',')), color=colors[i])
        #     plt.plot(labels + (i - len(for_comparison) / 2 + 0.5), mse_mean[key])
        i += 1
    # xx = np.linspace(1.0, 10.0, 1000)
    xx = np.linspace(0.0, 1.0, 1000)
    from scipy.interpolate import splev, splrep

    agh = [
        "1.00,coupled",
        "0.75,coupled",
        "0.50,coupled",
        "0.375,coupled",
        "0.25,coupled",
        "0.125,coupled"]

    for key, xi, yi in zip(agh, d_proportion_of_l_s,
                           d_ratio_score_t_d):
        plt.text(xi, yi + 0.00005, f"$n = {max4key[key]}$", fontsize=20)

    plt.plot(c_proportion_of_l_s, c_ratio_score_t_d, 'or')
    # plt.plot(np.log2(c_proportion_of_l_s), c_ratio_score_t_d, 'or')
    plt.plot(d_proportion_of_l_s, d_ratio_score_t_d, 'og')
    # plt.plot(np.log2(d_proportion_of_l_s), d_ratio_score_t_d, 'og')
    # print(c_proportion_of_l_s, c_ratio_score_t_d)
    tck = splrep(c_proportion_of_l_s[::-1], c_ratio_score_t_d[::-1], s=0)
    # tck = splrep(c_proportion_of_l_s, c_ratio_score_t_d, s=0)
    yy = splev(xx, tck)
    plt.plot(xx, yy, '-r')
    # plt.plot(np.log2(xx), yy, '-r')
    tck = splrep(d_proportion_of_l_s[::-1], d_ratio_score_t_d[::-1], s=0)
    # tck = splrep(d_proportion_of_l_s, d_ratio_score_t_d, s=0)
    yy = splev(xx, tck)
    plt.plot(xx, yy, '-g')
    # plt.plot(np.log2(xx), yy, '-g')
    # print('yes')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('MSE/dist', fontsize=30)
    # plt.xticks(np.arange(2, max4key[key] + qty_clean[key][0][0]), fontsize=30)
    plt.xticks([0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0], fontsize=30)
    plt.xlabel("$\\lambda$", fontsize=30)
    # plt.gca().set_xscale('log', base=2)
    # plt.xlabel("$\\frac{1}{\\lambda}$", fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("MSE/dist", fontsize=30)
    # plt.legend(["realnew", "old", "new"], prop={'size': 23})
    plt.legend(["Coupled", "Decoupled"], prop={'size': 13}, fancybox=True, shadow=True, frameon=True)
    # plt.tight_layout()
i = 0
if "score" in show:
    plt.figure()
    for key in for_comparison:
        labels = np.arange(qty_clean[key][0][0], max4key[key] + qty_clean[key][0][0])
        # print(key, np.round(np.mean(score_mean[key] / labels), 4), np.round(np.std(score_mean[key] / labels), 4))
        plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width,
                score_mean[key],
                width,
                yerr=score_std[key],
                color=colors[i],
                label=key)
        # label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')))
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
        plt.bar(labels + (i - len(for_comparison) / 2 + 0.5) * width, variance_mean[key], width, color=colors[i],
                label=key)
        # label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')))
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
                yerr=time_std[key], color=colors[i],
                label=key)
        # label="n_sensors: {1}, fusion: {0}, acq: {2}".format(*key.split(',')))
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
