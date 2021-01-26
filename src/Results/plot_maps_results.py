from copy import copy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use("seaborn")
# with open('E:/ETSI/Proyecto/data/Databases/numpy_files/ground_truth_norm.npy', 'rb') as g:
#     _z = np.load(g)

cmap = plt.cm.coolwarm_r
fig, ax = plt.subplots()
# cmap.set_bad(color="#00000000")
# , vmin=np.nanmin(_z),                 vmax=np.nanmin(_z)
import matplotlib.image as image

with open('E:/ETSI/Proyecto/data/Databases/numpy_files/nans.npy', 'rb') as g:
    nans = np.load(g)


def get_clean(_file):
    for nnan in nans:
        _file[nnan[0], nnan[1]] = -1
    return np.ma.array(_file, mask=(_file == -1))


_z = get_clean(np.flipud(image.imread("E:/ETSI/Proyecto/data/Map/Ypacarai/map.png")))

img = plt.imshow(_z[:, :, 0], cmap=cm.coolwarm, zorder=5, origin='lower')
# cbar = plt.colorbar(orientation='vertical')
# cbar.ax.tick_params(labelsize=20)
# CS = plt.contour(_z, colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
#                  alpha=0.6, linewidths=1.0, zorder=10)

plt.grid(True, zorder=0, color="white")
ax.set_facecolor('#eaeaf2')
# cbar.ax.set_xlabel(r'$\mu (x)$', fontsize=30)
# plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel("x [m]", fontsize=20)
plt.ylabel("y [m]", fontsize=20)
xticks = np.arange(0, 1000, 200)
yticks = np.arange(0, 1500, 200)
xnticks = [str(num * 10) for num in xticks]
ynticks = [str(num * 10) for num in yticks]
plt.xticks(xticks, xnticks, fontsize=20)
plt.yticks(yticks, ynticks, fontsize=20)  # .0052 .0017

plt.show(block=True)

with open('E:/ETSI/Proyecto/data/Databases/numpy_files/best_bo.npy', 'rb') as g:
    _bo = get_clean(np.load(g))
    _bo_xs = np.array([[785.0, 757.0],
                       [75.0, 872.0],
                       [554.0, 1366.0],
                       [539.0, 1275.0],
                       [387.0, 1424.0],
                       [497.0, 1231.0],
                       [678.0, 829.0],
                       [622.0, 614.0],
                       [728.0, 423.0],
                       [524.0, 501.0]]).T

with open('E:/ETSI/Proyecto/data/Databases/numpy_files/best_ga.npy', 'rb') as g:
    _ga = get_clean(np.load(g))
    _ga_xs = np.array([[785.0, 757.0],
                       [75.0, 872.0],
                       [524.0, 1168.0],
                       [326.0, 954.0],
                       [180.0, 804.0],
                       [467.0, 758.0],
                       [753.0, 712.0],
                       [792.0, 704.0],
                       [832.0, 671.0],
                       [723.0, 403.0],
                       [694.0, 343.0]]).T

with open('E:/ETSI/Proyecto/data/Databases/numpy_files/best_lm.npy', 'rb') as g:
    _lm = get_clean(np.load(g))
    _lm_xs = np.array([[785.0, 757.0],
                       [75.0, 872.0],
                       [898.0, 225.0],
                       [901.0, 475.0],
                       [819.0, 560.0],
                       [817.0, 310.0],
                       [734.0, 394.0],
                       [734.0, 642.0],
                       [649.0, 807.0],
                       [566.0, 392.0]]).T

i = 0
plt.figure()
for fig, xs in zip([_bo, _ga, _lm], [_bo_xs, _ga_xs, _lm_xs]):
    ax = plt.subplot(131 + i)
    plt.imshow(fig, origin='lower', cmap=current_cmap, vmin=np.nanmin(_z), vmax=np.nanmax(_z), zorder=5)
    plt.plot(xs[0, :], xs[1, :], '^y', markersize=12, alpha=0.5, label="Observations", zorder=6)
    CS = plt.contour(fig, colors='k',
                     alpha=0.6, linewidths=1.0, zorder=10)
    plt.clabel(CS, inline=1, fontsize=10)
    if i == 0:
        plt.ylabel("y [m]", fontsize=20)
    plt.grid(True, zorder=0, color="white")
    ax.set_facecolor('#eaeaf2')
    plt.xlabel("x [m]", fontsize=20)
    plt.legend(loc="lower left", prop={"size": 20})
    plt.xticks(xticks, xnticks, fontsize=20)
    plt.yticks(yticks, ynticks, fontsize=20)
    i += 1

current_cmap = copy(cm.get_cmap("jet"))
current_cmap.set_bad(color="#eaeaf200")

i = 0
plt.figure()
for fig, xs in zip([_bo, _ga, _lm], [_bo_xs, _ga_xs, _lm_xs]):
    ax = plt.subplot(131 + i)
    plt.imshow(np.power(_z - fig, 2), origin='lower', cmap=current_cmap, vmin=0, vmax=6.35767, zorder=5)
    plt.plot(xs[0, :], xs[1, :], '^y', markersize=12, alpha=0.5, label="Observations", zorder=6)
    CS = plt.contour(np.power(_z - fig, 2), colors='k',
                     alpha=0.6, linewidths=1.0, zorder=10)
    plt.clabel(CS, inline=1, fontsize=10)
    if i == 0:
        plt.ylabel("y [m]", fontsize=20)
    plt.grid(True, zorder=0, color="white")
    ax.set_facecolor('#eaeaf2')
    plt.xlabel("x [m]", fontsize=20)
    plt.legend(loc="lower left", prop={"size": 20})
    plt.xticks(xticks, xnticks, fontsize=20)
    plt.yticks(yticks, ynticks, fontsize=20)
    i += 1

# img = plt.imshow(_z, origin='lower', cmap='inferno')  # , vmin=0, vmax=6.357)
# cb = plt.colorbar(img)
# cb.ax.tick_params(labelsize=20)
# cb.ax.set_xlabel(r'$\mu(x)$', fontsize=30)
plt.tick_params()
plt.show(block=True)
