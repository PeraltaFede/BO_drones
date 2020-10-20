import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm


def expected_improvement(x, x_sample, y_sample, gpr, xi=0.001):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = gpr.predict(x, return_std=True)

        mu_sample = gpr.predict(x_sample)

    # Needed for noise-based model, otherwise use np.max(Y_sample). See also section 2.4 in [...]

    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def maxvalue_entropy_search(x, model, y_opt=0.0, c_point=np.zeros((1, 2))):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = model.predict(x, return_std=True)
    # mu = mu[0]
    # sigma = sigma[0]
    if y_opt < max(mu) + 5 / 1e6:
        y_opt = max(mu) + 5 / 1e6
    values = np.zeros_like(mu)
    mask = sigma > 0

    normalized_max = (y_opt - mu[mask]) / sigma[mask]
    pdf = norm.pdf(normalized_max)
    cdf = norm.cdf(normalized_max)
    cdf[np.where(cdf == 0.0)] = 1e-30
    values[mask] = (normalized_max * pdf) / (2 * cdf) - np.log(cdf)

    values[mask] *= np.exp(-cdist([c_point], x)/150).reshape(mu[mask].shape)
    return values


def gaussian_sei(x, model, y_opt=0.0, xi=0.001, c_point=np.zeros((1, 2))):
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

    values[mask] *= np.exp(-cdist([c_point], x)/150).reshape(mu[mask].shape)

    k = model.kernel.diag(x)
    v = k * ((Z ** 2 + 1) * cdf + Z * pdf) - (values ** 2)

    v[np.where(v < 1e-80)] = 1.0
    # values[np.where(values < 1e-40)] = 0.0

    return np.divide(values, np.sqrt(v))
