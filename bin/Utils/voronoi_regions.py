import numpy as np

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point


def calc_voronoi(region_center, close_agents, map_data):
    vor_list = [region_center[:2]]

    for generator in close_agents:
        vor_list.append(generator[:2])
    bounding_box = [0, map_data.shape[1], 0, map_data.shape[0]]

    points_left = np.copy(vor_list)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(vor_list)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(vor_list)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(vor_list)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(vor_list,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = Voronoi(points)
    # from scipy.spatial import voronoi_plot_2d
    # import matplotlib.pyplot as plt
    # voronoi_plot_2d(vor)
    # plt.show(block=True)
    region = -1
    for i in range(len(vor.points)):
        if (region_center[:2] == vor.points[i]).all():
            region = i
            break

    return vor, vor.vertices[vor.regions[vor.point_region[region]]]


def pointify(v_pos):
    return [Point(point) for point in v_pos]


def find_cvt_pos4region(all_acq, vect_pos, reg):
    reg_poly = Polygon(reg)
    mass = 0
    x_mass = np.array([0.0, 0.0])
    for (pos, m) in zip(vect_pos, all_acq):
        if reg_poly.contains(Point(pos)):
            mass += m
            x_mass += pos * m

    return np.round(x_mass / mass).astype(np.int)


def find_vect_pos4region(sorted_vect_pos, reg, return_idx=False):
    reg_poly = Polygon(reg)
    if return_idx:
        for i in range(len(sorted_vect_pos)):
            if reg_poly.contains(Point(sorted_vect_pos[i])):
                return sorted_vect_pos[i], i
    else:
        for pos in sorted_vect_pos:
            if reg_poly.contains(Point(pos)):
                return pos
    return None
    # print(slow_method_4_reg)
