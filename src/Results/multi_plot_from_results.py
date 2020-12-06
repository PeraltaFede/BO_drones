import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn")

name_files = glob.glob("E:/ETSI/Proyecto/results/csv_results/*.csv")

datas = []
dataype = []

for name_file in name_files:
    with open(name_file, 'r') as f:
        rl = f.readline()
        if 'split' in rl:
            dataype.append("split")
            datas.append(pd.read_csv(name_file, skiprows=2))
        elif 'truncated' in rl:
            dataype.append("trunc")
            datas.append(pd.read_csv(name_file, skiprows=2))
        elif 'GA' in rl:
            dataype.append("GA")
            datas.append(pd.read_csv(name_file, skiprows=2))
        elif 'LM' in rl:
            dataype.append("LM")
            datas.append(pd.read_csv(name_file, skiprows=2))
        elif 'rq' in rl:
            dataype.append("rq")
            datas.append(pd.read_csv(name_file, skiprows=2))
        elif 'matern' in rl:
            dataype.append("matern")
            datas.append(pd.read_csv(name_file, skiprows=2))
        elif 'mask' in rl:
            dataype.append("mask")
            datas.append(pd.read_csv(name_file, skiprows=2))

bosmse = []
bost_dist = []

bosmse2 = []
bost_dist2 = []

bosmse3 = []
bost_dist3 = []

gasmse = []
gast_dist = []

lmsmse = []
lmst_dist = []

# plt.figure()
# plt.subplot(121)
for i in range(len(datas)):
    if "mask" in dataype[i]:
        # plt.plot(np.arange(1, 21), datas[i]["mse"], '--r', alpha=0.3)
        # plt.plot(datas[i]["t_dist"] - datas[i]["t_dist"][0], datas[i]["mse"], '--r', alpha=0.3)
        bosmse.append(datas[i]["mse"].values)
        bost_dist.append(datas[i]["t_dist"].values)
    if "rq" in dataype[i]:
        # plt.plot(datas[i]["t_dist"]-datas[i]["t_dist"][0], datas[i]["mse"], '-g', alpha=0.2)
        bosmse2.append(datas[i]["mse"].values)
        bost_dist2.append(datas[i]["t_dist"].values)
    # elif "matern" in dataype[i]:
    #     # plt.plot(datas[i]["t_dist"]-datas[i]["t_dist"][0], datas[i]["mse"], '-b', alpha=0.2)
    #     bosmse3.append(datas[i]["mse"].values)
    #     bost_dist3.append(datas[i]["t_dist"].values)
    # elif "GA" in dataype[i]:
    #     # plt.plot(datas[i]["t_dist"] - datas[i]["t_dist"][0], datas[i]["mse"], '--k', alpha=0.3)
    #     # plt.plot(datas[i]["t_dist"] - datas[i]["t_dist"][0], datas[i]["mse"], '--k', alpha=0.3)
    #     gasmse.append(datas[i]["mse"].values)
    #     gast_dist.append(datas[i]["t_dist"].values)
    # elif "LM" in dataype[i]:
    #     # plt.plot(datas[i]["t_dist"] - datas[i]["t_dist"][0], datas[i]["mse"], '--k', alpha=0.3)
    #     # plt.plot(datas[i]["t_dist"] - datas[i]["t_dist"][0], datas[i]["mse"], '--k', alpha=0.3)
    #     lmsmse.append(datas[i]["mse"].values)
    #     lmst_dist.append(datas[i]["t_dist"].values)
# plt.figure()

n_mask = np.count_nonzero(np.array(dataype) == "mask")
n_spli = np.count_nonzero(np.array(dataype) == "rq")
# n_trun = np.count_nonzero(np.array(dataype) == "matern")
# n_ga = np.count_nonzero(np.array(dataype) == "GA")
# n_lm = np.count_nonzero(np.array(dataype) == "LM")

# avgs = []
# stds = []
# for run in bost_dist:
#     pdists = []
#     for i in range(len(run) - 1):
#         pdists.append(np.linalg.norm(run[i + 1] - run[i]))
#     avgs.append(np.mean(pdists))
#     stds.append(np.std(pdists))

bost_dist = np.array(bost_dist).T.reshape(20, -1)
bosmse = np.array(bosmse).T.reshape(20, -1)

bost_dist2 = np.array(bost_dist2).T.reshape(20, -1)
bosmse2 = np.array(bosmse2).T.reshape(20, -1)

# bost_dist3 = np.array(bost_dist3).T.reshape(20, -1)
# bosmse3 = np.array(bosmse3).T.reshape(20, -1)
#
# gast_dist = np.array(gast_dist).T.reshape(20, -1)
# gasmse = np.array(gasmse).T.reshape(20, -1)

# lmst_dist = np.array(lmst_dist).T.reshape(20, -1)
# lmsmse = np.array(lmsmse).T.reshape(20, -1)

bost_dist_mean = np.mean(bost_dist, axis=1)
bosmse_mean = np.mean(bosmse, axis=1)

bost_dist_mean2 = np.mean(bost_dist2, axis=1)
bosmse_mean2 = np.mean(bosmse2, axis=1)

# bost_dist_mean3 = np.mean(bost_dist3, axis=1)
# bosmse_mean3 = np.mean(bosmse3, axis=1)
#
# gast_dist_mean = np.mean(gast_dist, axis=1)
# gasmse_mean = np.mean(gasmse, axis=1)

# lmst_dist_mean = np.mean(lmst_dist, axis=1)
# lmsmse_mean = np.mean(lmsmse, axis=1)

bost_dist_std = np.std(bost_dist, axis=1)
bosmse_std = np.std(bosmse, axis=1)

bost_dist_std2 = np.std(bost_dist2, axis=1)
bosmse_std2 = np.std(bosmse2, axis=1)

# bost_dist_std3 = np.std(bost_dist3, axis=1)
# bosmse_std3 = np.std(bosmse3, axis=1)
#
# gast_dist_std = np.std(gast_dist, axis=1)
# gasmse_std = np.std(gasmse, axis=1)

# lmst_dist_std = np.std(lmst_dist, axis=1)
# lmsmse_std = np.std(lmsmse, axis=1)

# plt.figure()

plt.plot(np.arange(1, 21), bosmse_mean, 'g')
plt.plot(np.arange(1, 21), bosmse_mean2, 'y')
# plt.plot(np.arange(1, 21), bosmse_mean3, 'b')
# plt.plot(np.arange(1, 21), gasmse_mean, 'k')
# plt.plot(np.arange(1, 21), lmsmse_mean, 'r')
plt.fill_between(np.arange(1, 21), bosmse_mean - bosmse_std * 1.96, bosmse_mean + bosmse_std * 1.96, facecolor='green',
                 alpha=0.2)
plt.fill_between(np.arange(1, 21), bosmse_mean2 - bosmse_std2 * 1.96, bosmse_mean2 + bosmse_std2 * 1.96,
                 facecolor='yellow',
                 alpha=0.2)
# plt.fill_between(np.arange(1, 21), bosmse_mean3 - bosmse_std3 * 1.96, bosmse_mean3 + bosmse_std3 * 1.96,
#                  facecolor='blue',
#                  alpha=0.2)
# plt.fill_between(np.arange(1, 21), gasmse_mean - gasmse_std * 1.96, gasmse_mean + gasmse_std * 1.96, facecolor='black',
#                  alpha=0.2)
# plt.fill_between(np.arange(1, 21), lmsmse_mean - lmsmse_std * 1.96, lmsmse_mean + lmsmse_std * 1.96, facecolor='red',
#                  alpha=0.2)
plt.xticks(np.arange(1, 21))
plt.xlabel("Step")
plt.ylabel("MSE")
plt.legend(["$\\mu$ rbf", "$\\mu$ rq", "$\\mu$ mat", "$\\mu$ GA"])
plt.tight_layout()
plt.figure()
plt.plot(bost_dist_mean - bost_dist_mean[0], bosmse_mean, 'g')
plt.plot(bost_dist_mean2 - bost_dist_mean2[0], bosmse_mean2, 'y')
# plt.plot(bost_dist_mean3 - bost_dist_mean3[0], bosmse_mean3, 'b')
# plt.plot(gast_dist_mean - gast_dist_mean[0], gasmse_mean, 'k')
# plt.plot(lmst_dist_mean - lmst_dist_mean[0], lmsmse_mean, 'r')
plt.fill_between(bost_dist_mean - bost_dist_mean[0], bosmse_mean - bosmse_std * 1.96, bosmse_mean + bosmse_std * 1.96,
                 facecolor='green', alpha=0.2)
plt.fill_between(bost_dist_mean2 - bost_dist_mean2[0], bosmse_mean2 - bosmse_std2 * 1.96,
                 bosmse_mean2 + bosmse_std2 * 1.96,
                 facecolor='yellow', alpha=0.3)
# plt.fill_between(bost_dist_mean3 - bost_dist_mean3[0], bosmse_mean3 - bosmse_std3 * 1.96,
#                  bosmse_mean3 + bosmse_std3 * 1.96,
#                  facecolor='blue', alpha=0.3)
# plt.fill_between(gast_dist_mean - gast_dist_mean[0], gasmse_mean - gasmse_std * 1.96, gasmse_mean + gasmse_std * 1.96,
#                  facecolor='black', alpha=0.2)
# plt.fill_between(lmst_dist_mean - lmst_dist_mean[0], lmsmse_mean - lmsmse_std * 1.96, lmsmse_mean + lmsmse_std * 1.96,
#                  facecolor='red', alpha=0.2)
plt.xlabel("Total Distance")
plt.ylabel("MSE")
plt.legend(["$\\mu$ rbf", "$\\mu$ rq", "$\\mu$ mat", "$\\mu$ GA"])
plt.tight_layout()
plt.show()
