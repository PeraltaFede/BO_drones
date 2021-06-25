import matplotlib.pyplot as plt
import numpy as np

from bin.Environment.simple_env import Env
from bin.Utils.voronoi_regions import calc_voronoi

plt.style.use("tableau-colorblind10")
maps = []
# glob.glob("E:/ETSI/Proyecto/data/Databases/numpy_files/random*")

# for namefile in [  # "E:/ETSI/Proyecto/data/Databases/numpy_files/random_0.npy",
#                      # "E:/ETSI/Proyecto/data/Databases/numpy_files/random_1.npy",
#     # "E:/ETSI/Proyecto/data/Databases/numpy_files/noise_random_99.npy",
#     "E:/ETSI/Proyecto/data/Databases/numpy_files/random_3.npy",
#     # "E:/ETSI/Proyecto/data/Databases/numpy_files/random_4.npy",
#     # "E:/ETSI/Proyecto/data/Databases/numpy_files/random_5.npy",
#     # "E:/ETSI/Proyecto/data/Databases/numpy_files/random_6.npy",
#     # "E:/ETSI/Proyecto/data/Databases/numpy_files/random_7.npy",
# ]:
#     with open(namefile, 'rb') as g:
#         print(namefile)
#         maps.append(np.load(g))
env = Env("E:/ETSI/Proyecto/data/Map/Ypacarai/map.yaml")
fig_map = np.full_like(env.grid, np.nan)
for i in range(env.grid.shape[0]):
    for k in range(env.grid.shape[1]):
        if env.grid[i, k] == 0:
            fig_map[i, k] = 0
maps.append(fig_map)

# current_cmap = copy(cm.get_cmap("BuGn_r"))
# current_cmap.set_bad(color="#00000000")
xticks = np.arange(0, 1000, 200)
yticks = np.arange(0, 1500, 200)
xnticks = [str(format(num * 10, ',')) for num in xticks]
ynticks = [str(format(num * 10, ',')) for num in yticks]

i = 0

poses = np.array([[235, 893],
                  [530, 789],
                  [753, 460],
                  [413, 590],
                  [470, 1250]])

# [print(np.nanmin(mapz), np.nanmax(mapz)) for mapz in maps]

# for fig, iid in zip(maps, range(4)):
# np.random.seed(76842153)
# for fig, name in zip(maps, ["E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_0.npy",
#                             "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_1.npy",
#                             "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_2.npy",
#                             "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_3.npy",
#                             "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_4.npy",
#                             "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_5.npy",
#                             "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_6.npy",
#                             "E:/ETSI/Proyecto/data/Databases/numpy_files/noisy_random_7.npy",
#                             ]):
#     fig += np.random.normal(0, 0.01, fig.shape)
#     with open(name, 'wb') as g:
#         np.save(g, fig)

for fig in maps:
    # plt.subplot(141 + iid)
    plt.figure()
    img = plt.imshow(fig, origin='lower', cmap='tab10', zorder=5)
    # if iid == 2:
    #     cbar = plt.colorbar(orientation='vertical')
    #     cbar.ax.tick_params(labelsize=20)
    # CS = plt.contour(fig, colors='k',
    #                  alpha=0.6, linewidths=1.0, zorder=10)
    # for lim in range(len(poses)-1):
    #     _, reg = calc_voronoi(poses[lim], np.vstack([poses[0:lim], poses[lim + 1:-1]]), env.grid)
    #     plt.plot(np.append(reg[:, 0], reg[0, 0]), np.append(reg[:, 1], reg[0, 1]), '-y', zorder=10)
    plt.plot(poses[:, 0], poses[:, 1], 'oy', zorder=10)

    # _, reg = calc_voronoi(poses[0], poses[1:-1], env.grid)
    # plt.plot(np.append(reg[:, 0], reg[0, 0]), np.append(reg[:, 1], reg[0, 1]), '-r', zorder=10)
    plt.plot(poses[0, 0], poses[0, 1], 'oy', zorder=10)

    plt.grid(True, zorder=0, color="white")
    plt.gca().set_facecolor('#eceff1')
    # plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel("x (m)", fontsize=20)
    plt.ylabel("y (m)", fontsize=20)
    plt.xticks(xticks, labels=xnticks, fontsize=20)
    plt.yticks(yticks, labels=ynticks, fontsize=20)  # .0052 .0017
    plt.tight_layout()
    plt.show(block=False)
    i += 1
plt.show(block=True)
