import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bin.Coordinators.multi_informed_coordinator import Coordinator
from bin.Environment.simple_env import Env

screen_x0 = -57.38106014642857
screen_xscale = 1.030035714285734E-4
screen_y0 = -25.38342902978723
screen_yscale = 9.33042553191497E-5


def gps2pix(strlatlng):
    latlng = json.loads(strlatlng.replace("\'", "\""))
    screen_y = (latlng["lat"] - screen_y0) / screen_yscale
    screen_x = (latlng["lon"] - screen_x0) / screen_xscale
    return np.round(screen_x).astype(np.int64), np.round(screen_y).astype(np.int64)


with open('E:/ETSI/Proyecto/data/Databases/numpy_files/nans.npy', 'rb') as g:
    nans = np.load(g)


def get_clean(_file):
    for nnan in nans:
        _file[nnan[0], nnan[1]] = -1
    return np.ma.array(_file, mask=(_file == -1))


new_db = pd.read_csv("E:/ETSI/Proyecto/data/Databases/CSV/salida.csv")
env = Env("E:/ETSI/Proyecto/data/Map/Ypacarai/map.yaml")
coo = Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated")

for key in new_db.keys():
    if key == "id" or key == "datatime" or key == "position":
        continue
    else:
        new_db["{}_norm".format(key)] = (new_db[key] - new_db[key].mean()) / new_db[key].std()

# parameter = "tds"
# colors = ['b', 'g', 'r', 'y']
# for skip, clr in zip(np.arange(44, 36, -1), colors):
#     # skip = 38
#     i = 0
#     coo = Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated")
#     print(skip)
#     reads = []
#     for idx, row in new_db.iterrows():
#         if skip == i:
#             x, y = gps2pix(row["position"])
#             reads.append([np.array([x, y]), row["{}_norm".format(parameter)]])
#             i = 0
#         else:
#             i += 1
#
#     coo.initialize_data_gpr(reads)
#
#     Theta0 = np.logspace(0, 2, 1000)
#     LML = np.array(
#         [-coo.gp.log_marginal_likelihood(np.log([Theta0[i]])) for i in range(Theta0.shape[0])])
#     print(np.exp(coo.gp.kernel_.theta))
#     print(Theta0[np.where(LML == np.min(LML))])
#     plt.plot(Theta0, LML, label="skips: {}".format(skip), color=clr)
#     plt.plot(Theta0[np.where(LML == np.min(LML))], np.min(LML), '*', color=clr)

# plt.xscale("log")
# plt.yscale("log")
# plt.legend()
# plt.xlabel("Length-scale")
# plt.ylabel("Log-marginal-likelihood")
# plt.title("Log-marginal-likelihood")

coo = [Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated", k_name="RBF"),
       Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated", k_name="RBF2"),
       Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated", k_name="RBF2"),
       Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated", k_name="RBF")]
# coo = [Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated", k_name="RBF"),
#        Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated", k_name="RQ"),
#        Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated", k_name="RBF"),
#        Coordinator(env.grid, "t", acq="gaussian_ei", acq_mod="truncated", k_name="RQ")]
parameter = ["temperature",
             "tds",
             "ce",
             "ph"]
reads = {parameter[0]: [],
         parameter[1]: [],
         parameter[2]: [],
         parameter[3]: []}
skip = 51
i = 0
for idx, row in new_db.iterrows():
    if skip == i:
        x, y = gps2pix(row["position"])
        # reads[parameter[0]].append([np.array([x, y]), row["{}_norm".format(parameter[0])]])
        # reads[parameter[1]].append([np.array([x, y]), row["{}_norm".format(parameter[1])]])
        # reads[parameter[2]].append([np.array([x, y]), row["{}_norm".format(parameter[2])]])
        # reads[parameter[3]].append([np.array([x, y]), row["{}_norm".format(parameter[3])]])
        reads[parameter[0]].append([np.array([x, y]), row["{}".format(parameter[0])]])
        reads[parameter[1]].append([np.array([x, y]), row["{}".format(parameter[1])]])
        reads[parameter[2]].append([np.array([x, y]), row["{}".format(parameter[2])]])
        reads[parameter[3]].append([np.array([x, y]), row["{}".format(parameter[3])]])
        i = 0
    else:
        i += 1

# np.random.seed(0)
# a = np.arange(0, len(reads[parameter[0]]))
# np.random.shuffle(a)
# for param in parameter:
#     for i in range(len(a)):
#         reads[param][a[i]] = reads[param][i]

coo[0].initialize_data_gpr([reads[parameter[0]][0]])
coo[1].initialize_data_gpr([reads[parameter[1]][0]])
coo[2].initialize_data_gpr([reads[parameter[2]][0]])
coo[3].initialize_data_gpr([reads[parameter[3]][0]])
for i in range(len(reads[parameter[3]][1:])):
    for j in range(len(coo)):
        coo[j].add_data(reads[parameter[j]][i + 1])
        coo[j].fit_data()
        print(reads[parameter[j]][i + 1])
        u, s = coo[j].surrogate(return_std=True)
        # u = get_clean(u.reshape((1000, 1500)).T * new_db[parameter[j]].std() + new_db[parameter[j]].mean())
        u = get_clean(u.reshape((1000, 1500)).T)
        s = get_clean(s.reshape((1000, 1500)).T)
        plt.subplot(241 + j)
        plt.title('$\\mu({})$'.format(parameter[j]))
        plt.imshow(u, origin='lower')
        for read in reads[parameter[j]][:i + 2]:
            # print(read)
            plt.plot(read[0][0], read[0][1], 'o', color='#FFFFFF99')
        plt.subplot(245 + j)
        plt.title('$\\sigma({})$'.format(parameter[j]))
        plt.imshow(s, origin='lower')
        for read in reads[parameter[j]][:i + 2]:
            plt.plot(read[0][0], read[0][1], 'o', color='#FFFFFF99')
    plt.show(block=True)
