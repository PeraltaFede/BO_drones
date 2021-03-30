from copy import copy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use("seaborn")
maps = []
# glob.glob("E:/ETSI/Proyecto/data/Databases/numpy_files/random*")

for namefile in ["E:/ETSI/Proyecto/data/Databases/numpy_files/random_0.npy",
                 "E:/ETSI/Proyecto/data/Databases/numpy_files/random_1.npy",
                 "E:/ETSI/Proyecto/data/Databases/numpy_files/random_2.npy",
                 "E:/ETSI/Proyecto/data/Databases/numpy_files/random_3.npy",
                 "E:/ETSI/Proyecto/data/Databases/numpy_files/random_4.npy",
                 "E:/ETSI/Proyecto/data/Databases/numpy_files/random_5.npy",
                 "E:/ETSI/Proyecto/data/Databases/numpy_files/random_6.npy",
                 "E:/ETSI/Proyecto/data/Databases/numpy_files/random_7.npy",
                 ]:
    with open(namefile, 'rb') as g:
        print(namefile)
        maps.append(np.load(g))

current_cmap = copy(cm.get_cmap("pink"))
current_cmap.set_bad(color="#00000000")
xticks = np.arange(0, 1000, 200)
yticks = np.arange(0, 1500, 200)
xnticks = [str(num * 10) for num in xticks]
ynticks = [str(num * 10) for num in yticks]
i = 0

[print(np.nanmin(mapz), np.nanmax(mapz)) for mapz in maps]

# for fig, iid in zip(maps, range(4)):
np.random.seed(76842153)
for fig, name in zip(maps, ["E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_0.npy",
                            "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_1.npy",
                            "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_2.npy",
                            "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_3.npy",
                            "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_4.npy",
                            "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_5.npy",
                            "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_6.npy",
                            "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_7.npy",
                            ]):
    fig += np.random.normal(0, 0.01, fig.shape)
    with open(name, 'wb') as g:
        np.save(g, fig)

for fig in maps:
    # plt.subplot(141 + iid)
    plt.figure()
    img = plt.imshow(fig, origin='lower', cmap=current_cmap, zorder=5)
    # if iid == 2:
    #     cbar = plt.colorbar(orientation='vertical')
    #     cbar.ax.tick_params(labelsize=20)
    CS = plt.contour(fig, colors='k',
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
