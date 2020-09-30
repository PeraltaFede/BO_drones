import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import gpr


def evaluate(x, noise=0.1):
    return np.sin(x) + np.random.rand(len(x)).reshape(-1, 1) * noise


class GPR_(object):
    def __init__(self, length_scale=1):
        self.my_x = np.array([])
        self.my_y = np.array([])
        self.kernel = None
        self.L_ = None
        self.alpha = None
        self.length_scale = length_scale
        self.y_mean = 0

    def fit(self, x, y):
        if self.kernel is None:
            self.kernel = np.zeros((len(x), len(x)))
            for i in range(len(self.kernel)):
                for j in range(len(self.kernel[i, :])):
                    self.kernel[i, j] = self.get_norm((x[i] / self.length_scale - x[j] / self.length_scale))
            # self.kernel = self.kernel / np.max(self.kernel)
            self.my_x = x
            self.y_mean = np.mean(y)
            self.my_y = y - self.y_mean

            self.L_ = cholesky(self.kernel, lower=True)
            self.alpha = cho_solve((self.L_, True), self.my_y)

    def predict(self, x):
        ks = np.exp(-.5 * cdist(x / self.length_scale, self.my_x / self.length_scale,
                                metric='sqeuclidean'))

        y_pred = self.y_mean + ks.dot(self.alpha)

        kss = np.zeros((len(x), len(x)))
        for i in range(len(kss)):
            for j in range(len(kss[i, :])):
                kss[i, j] = self.get_norm((x[i] / self.length_scale - x[j] / self.length_scale))
        y_var = np.diag(kss) - np.diag(ks.dot(cho_solve((self.L_, True), ks.T)))

        return y_pred, np.sqrt(y_var)

    @staticmethod
    def get_norm(x):
        return np.exp(-(x ** 2) / 2)


space_x = np.arange(-np.pi, np.pi, 0.1).reshape(-1, 1)
initial_x = np.array([2, 3, 0, 1]).reshape(-1, 1)
data = {"X": initial_x, "Y": evaluate(initial_x, noise=0.01)}
my_gpr = GPR_()
ot_gpr = gpr.GaussianProcessRegressor(normalize_y=True)

my_gpr.fit(data["X"], data["Y"])
ot_gpr.fit(data["X"], data["Y"])

my_obs, my_std = my_gpr.predict(space_x)
ot_obs, ot_std = ot_gpr.predict(space_x, return_std=True)

plt.plot(space_x, evaluate(space_x, noise=0.0))
plt.plot(space_x, ot_obs)
plt.fill_between(space_x.ravel(),
                 my_obs.ravel() + 1.96 * my_std,
                 my_obs.ravel() - 1.96 * my_std,
                 alpha=0.1)
plt.plot(space_x, my_obs)
plt.fill_between(space_x.ravel(),
                 ot_obs.ravel() + 1.96 * ot_std,
                 ot_obs.ravel() - 1.96 * ot_std,
                 alpha=0.1)


plt.scatter(data["X"], data["Y"])
plt.legend(["real", "gp", "gp fs", "st dev", "st dev fs"])

plt.show()
