import numpy as np
from skopt.benchmarks import branin


def f(x):
    return branin(x)


from skopt.space import Real

space = [Real(-5, 10, name="x1"), Real(0, 15, name="x2")]

bounds = np.array([[-5, 10], [0, 15]])

X_init = np.array([[9, 2]])
Y_init = f(np.transpose(X_init))

"""
res = gp_minimize(f,                  # función que queremos optimizar
                  space,      # límites de cada dimensión
                  acq_func="EI",      # función de adquisición
                  #acq_func="PI",
                  #acq_func="LCB",
                  n_calls=100,         # número de evaluaciones 
                  n_random_starts=5,  # número de puntos aleatorios iniciales
                  #noise=0.1**2,       # nivel de ruido, es opcional
                  noise = 0,
                  random_state=123)   # la semilla

"""
# %% FUNCIONES DE ADQUISICIÓN

from scipy.stats import norm


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)

    # mu_sample = gpr.predict(np.transpose(X_sample))

    # sigma = sigma.reshape(-1, X_sample.shape[1])

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(Y_sample)
    # mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        if sigma == 0.0:
            return 0.0
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)  # todo: fijarse en como varia el primer miembro con respecto al
        #       segundo, a medida que se encuentran valores

    return ei


# %% OPTIMIZACIÓN DE LA FUNCIÓN DE ADQUISICIÓN

from scipy.optimize import minimize


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)


# %% OPTIMIZACIÓN BAYESIANA

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

m52 = ConstantKernel(1.0) * Matern(length_scale=1,
                                   nu=2.5)  # length scale que se adecua al gradiente del branin y tamaño del mapa
gpr = GaussianProcessRegressor(kernel=m52)

X_sample = X_init
Y_sample = Y_init

n_iter = 30

# para visualizar en 2D
_x = np.arange(bounds[0, 0], bounds[0, 1], 0.1)  #
_y = np.arange(bounds[1, 0], bounds[1, 1], 0.1)  #
_xy = np.meshgrid(_x, _y)  #
_x = _xy[0]  #
_y = _xy[1]  #
_xy = np.array(_xy).reshape(2, -1).T  #
_z = np.fromiter(map(f, zip(_x.flat, _y.flat)), dtype=np.float,  #
                 count=_x.shape[0] * _x.shape[1]).reshape(_x.shape)  #

for i in range(n_iter):
    gpr.fit(X_sample, Y_sample)

    print("EI Minimo 1", expected_improvement(np.array([-np.pi, 12.275]).reshape(1, -1), X_sample, Y_sample, gpr))
    print("EI Minimo 2", expected_improvement(np.array([np.pi, 2.275]).reshape(1, -1), X_sample, Y_sample, gpr))
    print("EI Minimo 3", expected_improvement(np.array([9.42478, 2.475]).reshape(1, -1), X_sample, Y_sample, gpr))

    X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)

    print("EI P Selec.", expected_improvement(X_next.reshape(1, -1), X_sample, Y_sample, gpr))
    print("-----")

    Y_next = f(X_next)
    X_next = np.transpose(X_next)

    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))

import matplotlib.pyplot as plt  #

plt.subplot(131)  #
plt.pcolormesh(_x, _y, _z, shading='auto')  #
_gpz, _stz = gpr.predict(_xy, return_std=True)  #
_gpz = _gpz.reshape((len(_x), len(_y)))
plt.subplot(132)  #
plt.pcolormesh(_x, _y, _gpz, shading='auto')  #
plt.subplot(133)  #
plt.pcolormesh(_x, _y, _stz.reshape((len(_x), len(_y))), shading='auto')  #

print("Valores óptimos de _z: ", np.min(_z))  #
xm, ym = np.where(_z == np.min(_z))  # para encontrar uno de lo minimos segun el mapeo
print("en : ", _x[xm, ym], _y[xm, ym])  #
print("Valores óptimos de _gpz: ", np.min(_gpz))  #
xm, ym = np.where(_gpz == np.min(_gpz))  # encontrando el minimo segun el ultimo fitting
print("en : ", _x[xm, ym], _y[xm, ym])  #

print("Valores óptimos de x: ", X_next)
print("Valor óptimo de f(x): ", Y_next)
plt.show(block=True)
