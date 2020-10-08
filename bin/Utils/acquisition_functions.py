import warnings

import numpy as np
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


def gaussian_ei(x, model, y_opt=0.0, xi=0.001):
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
    values = np.zeros_like(mu)
    mask = sigma > 0
    imp = y_opt - mu[mask] - xi
    Z = imp / sigma[mask]
    cdf = norm.cdf(Z)
    pdf = norm.pdf(Z)

    exploit = imp * cdf
    explore = sigma[mask] * pdf
    values[mask] = exploit + explore

    k = model.kernel.diag(x)
    v = k * ((Z ** 2 + 1) * cdf + Z * pdf) - (values ** 2)

    v[np.where(v < 1e-80)] = 1.0
    # values[np.where(values < 1e-40)] = 0.0

    return np.divide(values, np.sqrt(v))
