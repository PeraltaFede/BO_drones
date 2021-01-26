from copy import copy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use("seaborn")
maps = []
# glob.glob("E:/ETSI/Proyecto/data/Databases/numpy_files/random*")
for namefile in ["E:/ETSI/Proyecto/data/Databases/numpy_files/random_12.npy",
                 "E:/ETSI/Proyecto/data/Databases/numpy_files/random_54.npy",
                 "E:/ETSI/Proyecto/data/Databases/numpy_files/random_60.npy"]:
    with open(namefile, 'rb') as g:
        print(namefile)
        maps.append(np.load(g))

current_cmap = copy(cm.get_cmap("plasma"))
current_cmap.set_bad(color="#00000000")
xticks = np.arange(0, 1000, 200)
yticks = np.arange(0, 1500, 200)
xnticks = [str(num * 10) for num in xticks]
ynticks = [str(num * 10) for num in yticks]
i = 0
print(np.nanmax(maps[0]))
print(np.nanmin(maps[0]))
print(np.nanmax(maps[1]))
print(np.nanmin(maps[1]))
print(np.nanmax(maps[2]))
print(np.nanmin(maps[2]))

for fig in maps:
    plt.figure()
    img = plt.imshow(fig, origin='lower', cmap=current_cmap, zorder=5)
    # cbar = plt.colorbar(orientation='vertical')
    # cbar.ax.tick_params(labelsize=20)
    CS = plt.contour(fig, colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                     alpha=0.6, linewidths=1.0, zorder=10)
    plt.grid(True, zorder=0, color="white")
    plt.gca().set_facecolor('#eaeaf2')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel("x [m]", fontsize=20)
    plt.ylabel("y [m]", fontsize=20)
    # plt.xticks(xnticks, fontsize=20)
    # plt.yticks(ynticks, fontsize=20)  # .0052 .0017
    plt.xticks(xticks, labels=xnticks, fontsize=20)
    plt.yticks(yticks, labels=ynticks, fontsize=20)  # .0052 .0017
    plt.tight_layout()
    plt.show(block=False)
    i += 1
plt.show(block=True)
