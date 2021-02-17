from sys import path

import numpy as np
from deap import benchmarks
from skopt.benchmarks import branin as brn

w_obstacles = False

a = []
c = []


# maxz = 0
# meanz = 0


def bohachevsky_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.bohachevsky(sol[:2])[0]


def ackley_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.ackley(sol[:2])[0]


def rosenbrock_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.rosenbrock(sol[:2])[0]


def himmelblau_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.himmelblau(sol[:2])[0]


def branin(sol):
    return np.nan if w_obstacles and sol[2] == 1 else brn(sol[:2])


def shekel_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.shekel(sol[:2], a, c)[0]


def schwefel_arg0(sol):
    return np.nan if w_obstacles and sol[2] == 1 else benchmarks.h1(sol[:2])[0]


def create_map(grid, resolution, obstacles_on=False, randomize_shekel=False, sensor="", no_maxima=10, load_from_db=True,
               file=0):
    if load_from_db:
        if sensor == "s1":
            file = 0
        elif sensor == "s2":
            file = 1
        elif sensor == "s3":
            file = 2
        elif sensor == "s4":
            file = 3
        elif sensor == "s5":
            file = 4
        elif sensor == "s6":
            file = 5
        elif sensor == "s7":
            file = 6
        elif sensor == "s8":
            file = 7
        with open(path[-1] + '/data/Databases/numpy_files/random_{}.npy'.format(file), 'rb') as g:
            # with open('E:/ETSI/Proyecto/data/Databases/numpy_files/ground_truth_norm.npy', 'rb') as g:
            # _z = np.load(g)
            # print(np.nanmax(_z))
            # print(np.nanmin(_z))
            # print(np.nanmean(_z), np.nanstd(_z))
            # _z = _z[~np.isnan(_z)]
            # import matplotlib.pyplot as plt
            #
            # plt.figure()
            # plt.hist(_z)
            # plt.show(block=True)
            return np.load(g)
    else:
        global w_obstacles, a, c
        w_obstacles = obstacles_on
        xmin = -1
        xmax = 1
        ymin = 0
        ymax = 2
        # _z = []
        if randomize_shekel:
            no_maxima = np.random.randint(2, 6)
            xmin = 0
            xmax = 10
            ymin = 0
            ymax = 10
            a = []
            c = []
            for i in range(no_maxima):
                # a.append([2 + np.random.rand() * 6, 2 + np.random.rand() * 6])
                a.append([1.2 + np.random.rand() * 8.8, 1.2 + np.random.rand() * 8.8])
                c.append(5)
            # print(a)
            # print(c)
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
        _z = np.fromiter(map(rosenbrock_arg0, zip(_x.flat, _y.flat, grid.flat)), dtype=np.float,
                         count=_x.shape[0] * _x.shape[1]).reshape(_x.shape)

        # else:
        #     _z1 = np.fromiter(map(shekel_arg0, zip(_x.flat, _y.flat, grid.flat)), dtype=np.float,
        #                       count=_x.shape[0] * _x.shape[1]).reshape(_x.shape)
        #     _z = np.multiply(_z, _z1)
        #
        # if no_minimum > 1:
        # #     _z = _z ** (1/no_minimum)
        # global maxz, meanz
        # maxz = np.nanmax(_z)
        # minz = np.nanmin(_z)
        # _z = _z - (minz + maxz) / 2
        # plt.hist(_z.flatten(), alpha=0.5)
        # print()
        # print(np.nanmax(_z))
        # print(np.nanmin(_z))

        meanz = np.nanmean(_z)
        stdz = np.nanstd(_z)
        _z = (_z - meanz) / stdz

        # print(np.nanmax(_z))
        # print(np.nanmin(_z))
        # _z = _z / np.linalg.norm(_z, ord=2, axis=1, keepdims=True)

        # with open('E:/ETSI/Proyecto/data/Databases/numpy_files/shww.npy', 'wb') as g:
        #     np.save(g, _z)
        return _z


def get_init_pos4(n=1, rotate_rnd=True, expand=False, map_data=None):
    step = np.pi * 2 / n
    if rotate_rnd:
        extra_angle = np.random.rand() * np.pi * 2
    else:
        extra_angle = 0
    angles = [step * i + extra_angle for i in range(n)]
    initial_positions = np.full((n, 3), [500.0, 750.0, 0.0])

    cosine = np.cos(angles)
    sine = np.sin(angles)
    if expand:
        initial_positions[:, 0] += 220 * cosine
        initial_positions[:, 1] += 220 * sine
        for i in range(len(initial_positions[:, 0])):
            while True:
                initial_positions[i, 0] += cosine[i]
                initial_positions[i, 1] += sine[i]
                if map_data[np.round(initial_positions[i, 1]).astype(np.int), np.round(initial_positions[i, 0]).astype(
                        np.int)] == 1:
                    initial_positions[i, 0] -= cosine[i]
                    initial_positions[i, 1] -= sine[i]
                    break
    else:
        initial_positions[:, 0] += 10 * cosine
        initial_positions[:, 1] += 10 * sine
    return np.round(initial_positions).astype(np.int)
