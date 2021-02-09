import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel, Hyperparameter


class ExpSineSquared(StationaryKernelMixin, NormalizedKernelMixin, Kernel):

    def __init__(self, length_scale=1.0, periodicity=1.0,
                 length_scale_bounds=(1e-5, 1e5),
                 periodicity_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.length_scale_bounds = length_scale_bounds
        self.periodicity_bounds = periodicity_bounds

    @property
    def hyperparameter_length_scale(self):
        """Returns the length scale"""
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_periodicity(self):
        return Hyperparameter(
            "periodicity", "numeric", self.periodicity_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(pdist(X, metric='euclidean'))
            arg = np.pi * dists / self.periodicity
            sin_of_arg = np.sin(arg)
            K = np.exp(- 2 * (sin_of_arg / self.length_scale) ** 2)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric='euclidean')
            K = np.exp(- 2 * (np.sin(np.pi / self.periodicity * dists)
                              / self.length_scale) ** 2)

        if eval_gradient:
            cos_of_arg = np.cos(arg)
            # gradient with respect to length_scale
            if not self.hyperparameter_length_scale.fixed:
                length_scale_gradient = \
                    4 / self.length_scale ** 2 * sin_of_arg ** 2 * K
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:  # length_scale is kept fixed
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))
            # gradient with respect to p
            if not self.hyperparameter_periodicity.fixed:
                periodicity_gradient = \
                    4 * arg / self.length_scale ** 2 * cos_of_arg \
                    * sin_of_arg * K
                periodicity_gradient = periodicity_gradient[:, :, np.newaxis]
            else:  # p is kept fixed
                periodicity_gradient = np.empty((K.shape[0], K.shape[1], 0))

            return K, np.dstack((length_scale_gradient, periodicity_gradient))
        else:
            return K

    def __repr__(self):
        return "{0}(length_scale={1:.3g}, periodicity={2:.3g})".format(
            self.__class__.__name__, self.length_scale, self.periodicity)
