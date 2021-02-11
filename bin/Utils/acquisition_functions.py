import warnings
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm


def predictive_entropy_search(x, mu, sigma, model, x_star=None, noise=0):
    # diferencias principales con respecto a paper: el pareto set en una sola aproximacion
    # x_star se define como el maximo de la suma de los sigmas (para mejorar la explotacion)
    # Predictive Entropy Search for Efficient Global

    values = np.zeros_like(mu)
    mask = sigma > 0
    # todo: cambiar monte_carlo_samples
    if x_star is None:
        pos = np.where(sigma == np.max(sigma))[0][0]
        x_star = x[pos]
    else:
        pos = np.where(x == x_star)[0][0]
    val = mu[pos]
    if not (x_star == model.X_train_).all():
        model2 = deepcopy(model)
        model2.fit(np.append(model.X_train_, x_star).reshape(-1, 2), np.append(model.y_train_, val))
        _, sigma_cpd = model2.predict(x, return_std=True)
        values[mask] = np.log(sigma[mask] + noise ** 2) - np.log(sigma_cpd[mask] + noise ** 2)
    else:
        print('this sentence will not be printed')
        values[mask] = np.log(sigma[mask] + noise ** 2)

    return values


def maxvalue_entropy_search(x, model, y_opt=0.0, c_point=np.zeros((1, 2)), xi=0.01, masked=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = model.predict(x, return_std=True)
    # mu = mu[0]
    # sigma = sigma[0]
    if y_opt < max(mu) + 5 / 1e6:
        y_opt = max(mu) + 5 / 1e6
    values = np.zeros_like(mu)
    mask = sigma > 0

    normalized_max = (y_opt - mu[mask] - xi) / sigma[mask]
    pdf = norm.pdf(normalized_max)
    cdf = norm.cdf(normalized_max)
    cdf[np.where(cdf == 0.0)] = 1e-30
    values[mask] = (normalized_max * pdf) / (2 * cdf) - np.log(cdf)

    if masked:
        values[mask] *= np.exp(-cdist([c_point], x) / 250).reshape(mu[mask].shape)
    return values


def gaussian_sei(x, model, y_opt=0.0, xi=0.01, c_point=np.zeros((1, 2)), masked=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = model.predict(x, return_std=True)

    # check dimensionality of mu, std so we can divide them below
    if (mu.ndim != 1) or (sigma.ndim != 1):
        raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                         "however both must be 1-dimensional. Did you train "
                         "your model with an (N, 1) vector instead of an "
                         "(N,) vector?"
                         .format(mu.ndim, sigma.ndim))

    # print(c_point)
    # print(x[0, :])
    # print(np.where(cdist([c_point], x) == np.min(cdist([c_point], x))))

    values = np.zeros_like(mu)
    mask = sigma > 0
    imp = y_opt - mu[mask] - xi  # todo ?????
    # imp += np.interp(np.exp(-cdist([c_point], x)).reshape(mu.shape), [-1, 0], [np.nanmin(imp), np.nanmax(imp)])
    # print(np.nanmax(imp))
    # print(np.nanmin(imp))

    Z = imp / sigma[mask]
    cdf = norm.cdf(Z)
    pdf = norm.pdf(Z)

    exploit = imp * cdf
    explore = sigma[mask] * pdf
    values[mask] = exploit + explore
    if masked:
        values[mask] *= np.exp(-cdist([c_point], x) / 250).reshape(mu[mask].shape)

    k = model.kernel.diag(x)
    v = k * ((Z ** 2 + 1) * cdf + Z * pdf) - (values ** 2)

    v[np.where(v < 1e-80)] = 1.0
    # values[np.where(values < 1e-40)] = 0.0

    return np.divide(values, np.sqrt(v))


def gaussian_pi(x, model, y_opt=0.0, xi=0.01, c_point=np.zeros((1, 2)), masked=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, std = model.predict(x, return_std=True)
    if (mu.ndim != 1) or (std.ndim != 1):
        raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                         "however both must be 1-dimensional. Did you train "
                         "your model with an (N, 1) vector instead of an "
                         "(N,) vector?"
                         .format(mu.ndim, std.ndim))

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    values[mask] = norm.cdf(scaled)

    if masked:
        values[mask] *= np.exp(-cdist([c_point], x) / 250).reshape(mu[mask].shape)
    return values


def max_std(x, model, c_point=np.zeros((1, 2)), masked=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, std = model.predict(x, return_std=True)
    mask = std > 0
    if masked:
        std *= np.exp(-cdist([c_point], x) / 250).reshape(mu[mask].shape)
    return std


def gaussian_ei(x, model, y_opt=0.0, xi=0.01, c_point=np.zeros((1, 2)), masked=True):
    if not isinstance(model, tuple):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, std = model.predict(x, return_std=True)
        # check dimensionality of mu, std so we can divide them below
        if (mu.ndim != 1) or (std.ndim != 1):
            raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                             "however both must be 1-dimensional. Did you train "
                             "your model with an (N, 1) vector instead of an "
                             "(N,) vector?"
                             .format(mu.ndim, std.ndim))
    else:
        mu = model[0]
        std = model[1]
    # return std

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore
    if masked:
        values[mask] *= np.exp(-cdist([c_point], x) / 250).reshape(mu[mask].shape)

    # values[np.where(values < 1e-10)] = 0.0

    return values
