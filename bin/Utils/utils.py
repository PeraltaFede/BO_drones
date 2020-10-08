import numpy as np
from deap import benchmarks

w_obstacles = False

a = []
c = []


def bohachevsky_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.bohachevsky(sol[:2])[0]


def ackley_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.ackley(sol[:2])[0]


def rosenbrock_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.rosenbrock(sol[:2])[0]


def himmelblau_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.himmelblau(sol[:2])[0]


def shekel_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.shekel(sol[:2], a, c)[0]


def create_map(grid, resolution, obstacles_on=False, randomize_shekel=False, no_maxima=3, load_from_db=True):
    if load_from_db:
        with open('E:/ETSI/Proyecto/data/Databases/numpy_files/ground_truth1.npy', 'rb') as g:
            return np.load(g)
    else:
        global w_obstacles, a, c
        w_obstacles = obstacles_on
        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1
        # _z = []
        if randomize_shekel:
            a = []
            c = []
            for i in range(no_maxima):
                a.append([np.random.rand(), np.random.rand()])
                c.append(0.15)
            print(a)
            print(c)
            a = np.array(a)
            c = np.array(c).T
        else:
            a = np.array([[0.16, 1 / 1.5], [0.9, 0.2 / 1.5]])
            c = np.array([0.15, 0.15]).T

        # for i in range(no_minimum):
        xadd = 0  # (np.random.rand() - 0.5) * (xmax - xmin) / 2
        yadd = 0  # (np.random.rand() - 0.5) * (ymax - ymin) / 2

        _x = np.arange(xmin, xmax, resolution * (xmax - xmin) / (grid.shape[1])) + xadd
        _y = np.arange(xmin, xmax, resolution * (ymax - ymin) / (grid.shape[0])) + yadd
        _x, _y = np.meshgrid(_x, _y)
        # if i == 0:
        _z = np.fromiter(map(shekel_arg0, zip(_x.flat, _y.flat, grid.flat)), dtype=np.float,
                         count=_x.shape[0] * _x.shape[1]).reshape(_x.shape)
        # else:
        #     _z1 = np.fromiter(map(shekel_arg0, zip(_x.flat, _y.flat, grid.flat)), dtype=np.float,
        #                       count=_x.shape[0] * _x.shape[1]).reshape(_x.shape)
        #     _z = np.multiply(_z, _z1)
        #
        # if no_minimum > 1:
        # #     _z = _z ** (1/no_minimum)
        # with open('E:/ETSI/Proyecto/data/Databases/numpy_files/ground_truth1.npy', 'wb') as g:
        #     np.save(g, _z)

        #     b = np.load(f)
        return _z
