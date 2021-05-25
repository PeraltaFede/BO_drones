from copy import copy
from sys import path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from RegscorePy.bic import bic

path.extend([path[0][:path[0].rindex("src") - 1]])
from bin.Coordinators.gym_coordinator import Coordinator
from bin.Environment.simple_env import Env

with open('/data/Databases/CSV/nans.npy', 'rb') as g:
    nans = np.load(g)


def get_clean(_file):
    for nnan in nans:
        _file[nnan[0], nnan[1]] = -1
    return np.ma.array(_file, mask=(_file == -1))


# BO: -840163.6160239156
# GA: -824034.8196472471
# PM:
selection = "PM"
if selection == "BO":
    _bo_xs = np.array([[563, 375],
                       [559, 410],
                       [604, 368],
                       [647, 327],
                       [704, 362],
                       [690, 430],
                       [671, 504],
                       [676, 581],
                       [649, 650],
                       [611, 716],
                       [593, 793],
                       [584, 873],
                       [498, 888],
                       [416, 915],
                       [331, 936],
                       [245, 953],
                       [214, 1023],
                       [259, 958],
                       [276, 1033],
                       [293, 1108],
                       [331, 1039],
                       [334, 1046]])
elif selection == "GA":
    # arriba es bo aca ga y dp pesmoc
    _bo_xs = np.array([[164, 1159],
                       [170, 1122],
                       [180, 1045],
                       [181, 1036],
                       [255, 966],
                       [323, 901],
                       [382, 845],
                       [439, 790],
                       [499, 733],
                       [560, 674],
                       [624, 612],
                       [688, 548],
                       [752, 485],
                       [815, 421],
                       [877, 356],
                       [890, 344],
                       [858, 422],
                       [825, 502],
                       [792, 579],
                       [759, 656],
                       [740, 699], ])
elif selection == "PM":
    _bo_xs = np.array([[880, 539],
                       [844, 545],
                       [825, 585],
                       [802, 633],
                       [770, 698],
                       [735, 768],
                       [655, 789],
                       [575, 811],
                       [562, 896],
                       [557, 984],
                       [474, 979],
                       [483, 1061],
                       [405, 1040],
                       [429, 1122],
                       [368, 1066],
                       [311, 1013],
                       [336, 937],
                       [360, 864],
                       [386, 791],
                       [419, 720],
                       [464, 666]])

sensors = {"s5", "s6"}
environment = Env(map_path2yaml=path[-1] + "/data/Map/Ypacarai/map.yaml")
environment.add_new_map(sensors, file=0, clone4noiseless=False)
coordinator = Coordinator(environment.grid, sensors)

pos = _bo_xs[0, :]
i_pos = pos
_bo_xs = _bo_xs[1:, :]
read = [{"pos": pos, "s5": environment.maps["s5"][pos[1], pos[0]], "s6": environment.maps["s6"][pos[1], pos[0]]}]

xticks = np.arange(0, 1000, 200)
yticks = np.arange(0, 1500, 200)
xnticks = [str(format(num * 10, ",")) for num in xticks]
ynticks = [str(format(num * 10, ",")) for num in yticks]

coordinator.initialize_data_gpr(read)
for pos in _bo_xs:
    read = {"pos": pos, "s5": environment.maps["s5"][pos[1], pos[0]], "s6": environment.maps["s6"][pos[1], pos[0]]}
    coordinator.add_data(read)
    coordinator.fit_data()

fig, axs = plt.subplots(1, 4)  # 4 axs
current_cmap = copy(cm.get_cmap("jet"))
current_cmap.set_bad(color="#eaeaf200")
current_cmap2 = copy(cm.get_cmap("cividis"))
current_cmap2.set_bad(color="#eaeaf200")

tuples = coordinator.surrogate(_x=coordinator.all_vector_pos,
                               return_std=True)  # vector de 2 componentes, cada comp 2 imgs
_map = environment.maps["s5"][~np.isnan(environment.maps["s5"])]

# print(bic(_map, get_clean(tuples[1][0].reshape((1000, 1500)).T)[~np.isnan(environment.maps["s5"])], len(_bo_xs)))
tu = [[], [], [], []]
tu[0] = np.power(environment.maps["s5"] - get_clean(tuples[0][0].reshape((1000, 1500)).T), 2)
tu[1] = np.power(environment.maps["s6"] - get_clean(tuples[1][0].reshape((1000, 1500)).T), 2)
tu[2] = get_clean(tuples[0][1].reshape((1000, 1500)).T)
tu[3] = get_clean(tuples[1][1].reshape((1000, 1500)).T)

# auxmin1 = np.nanmin(tu[0])
# auxmin2 = np.nanmin(tu[1])
# vmin1 = min(auxmin1, auxmin2)
# auxmax1 = np.nanmax(tu[0])
# auxmax2 = np.nanmax(tu[1])
# vmax1 = max(auxmax1, auxmax2)
# auxmin3 = np.nanmin(tu[2])
# auxmin4 = np.nanmin(tu[3])
# vmin2 = min(auxmin3, auxmin4)
# auxmax3 = np.nanmax(tu[2])
# auxmax4 = np.nanmax(tu[3])
# vmax2 = max(auxmax3, auxmax4)


for ax, ts in zip(axs, enumerate(tu)):
    if ts[0] < 2:
        aux = ax.imshow(ts[1], origin='lower', zorder=5, cmap=current_cmap)
    else:
        aux = ax.imshow(ts[1], origin='lower', zorder=5, cmap=current_cmap2)
    ax.plot(_bo_xs[:, 0], _bo_xs[:, 1], '^y', zorder=10)
    ax.plot(i_pos[0], i_pos[1], '^y', zorder=10)
    CS = ax.contour(ts[1], colors='k', alpha=0.6, linewidths=1.3, zorder=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(aux, cax=cax, orientation='vertical')
    ax.grid(True, zorder=0, color="white")
    ax.clabel(CS, inline=1, fontsize=12)
    ax.set_facecolor('#eaeaf2')
    ax.set_xlabel("x (m)", fontsize=17)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xnticks, fontsize=17)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ynticks, fontsize=0)
    if ts[0] % 2 == 0:
        ax.set_title("S5")
    else:
        ax.set_title("S6")
plt.sca(axs[0])
plt.yticks(yticks, labels=ynticks, fontsize=17)
plt.ylabel("y (m)", fontsize=17)
plt.show(block=True)
