import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt.acquisition import gaussian_ei

from bin.Utils.utils import create_map

_z = create_map(None, None)


def f(_x):
    """
    :param _x: [x, y] position to read
    :return: value of _z at position [x, y]
    """
    # print(_x)
    # print(_z[_x[1], _x[0]])
    if np.isnan(_z[_x[1], _x[0]]):
        return 8
    return _z[_x[1], _x[0]]


res = gp_minimize(f,  # función que queremos optimizar
                  [(0, 999), (0, 1499)],  # límites de cada dimensión
                  acq_func="EI",  # función de adquisición
                  n_calls=15,  # número de evaluaciones
                  n_random_starts=5,  # número de puntos aleatorios iniciales
                  noise=0.05 ** 2,  # nivel de ruido, es opcional
                  random_state=123)  # la semilla

x = np.mgrid[0:1000:1, 0:1500:1].reshape(2, -1).T
x_gp = res.space.transform(x.tolist())

for n_iter in range(5):
    gp = res.models[n_iter]  # modelo usado
    curr_x_iters = res.x_iters[:5 + n_iter]
    curr_func_vals = res.func_vals[:5 + n_iter]
    a = np.array(curr_x_iters)

    y_pred, sigma = gp.predict(x_gp, return_std=True)
    plt.subplot(5, 3, 1 + n_iter*3)
    plt.imshow(y_pred.reshape(1000, 1500).T, origin='lower')
    plt.plot(a[:, 0], a[:, 1], 'or', fillstyle="none", alpha=0.5)
    plt.grid()
    plt.subplot(5, 3, 2 + n_iter*3)
    plt.imshow(sigma.reshape(1000, 1500).T, origin='lower')
    plt.plot(a[:, 0], a[:, 1], 'or', fillstyle="none", alpha=0.5)
    plt.grid()

    plt.subplot(5, 3, 3 + n_iter*3)
    acq = gaussian_ei(x_gp, gp, y_opt=np.min(curr_func_vals))
    plt.imshow(acq.reshape(1000, 1500).T, origin='lower')
    plt.plot(a[:, 0], a[:, 1], 'or', fillstyle="none", alpha=0.5)
    next_x = res.x_iters[5 + n_iter]
    next_acq = gaussian_ei(res.space.transform([next_x]), gp, y_opt=np.min(curr_func_vals))
    print(next_x)
    print(next_acq)
    plt.plot(next_x[0], next_x[1], "bo", markersize=6, label="Next query point")
    # plt.grid()

plt.show(block=True)

    # %% VISUALIZAMOS EL RESULTADO FINAL
    # plt.show()
#
# plt.figure(figsize=(6, 4))
#
# # Plot f(x) + contours
# x = np.linspace(-2, 2, 400).reshape(-1, 1)
# x_gp = res.space.transform(x.tolist())
#
# fx = [f(x_i, _noise_level=0.0) for x_i in x]
# plt.plot(x, fx, "r--", label="True (unknown)")
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
#                          [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
#          alpha=.2, fc="r", ec="None")
#
# # Plot GP(x) + contours
# gp = res.models[-1]  # el último modelo
# y_pred, sigma = gp.predict(x_gp, return_std=True)
#
# plt.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                          (y_pred + 1.9600 * sigma)[::-1]]),
#          alpha=.2, fc="g", ec="None")
#
# # Plot sampled points
# plt.plot(res.x_iters,
#          res.func_vals,
#          "r.", markersize=15, label="Observations")
#
# plt.title(r"$x^* = %.4f, f(x^*) = %.4f$" % (res.x[0], res.fun))
# plt.legend(loc="best", prop={'size': 8}, numpoints=1)
# plt.grid()
plt.show()
