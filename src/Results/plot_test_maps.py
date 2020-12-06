import glob
from copy import copy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use("seaborn")
maps = []
for namefile in glob.glob("E:/ETSI/Proyecto/data/Databases/numpy_files/random*"):
    with open(namefile, 'rb') as g:
        print(namefile)
        maps.append(np.load(g))

current_cmap = copy(cm.get_cmap("inferno"))
current_cmap.set_bad(color="#00000000")

i = 0
for fig in maps:
    plt.figure()
    img = plt.imshow(fig, origin='lower', cmap=current_cmap, zorder=5)
    cbar = plt.colorbar(orientation='vertical')
    cbar.ax.tick_params(labelsize=20)
    CS = plt.contour(fig, colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                     alpha=0.6, linewidths=1.0, zorder=10)
    plt.grid(True, zorder=0, color="white")
    # plt..set_facecolor('#eaeaf2')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # .0052 .0017
    plt.show(block=False)
    i += 1
plt.show(block=True)