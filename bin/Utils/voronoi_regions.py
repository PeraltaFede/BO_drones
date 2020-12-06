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


def create_voronoi_region():
    vor, region = calc_voronoi()
    max_dist = 0
    for num in region:
        if np.linalg.norm(np.subtract(np.array([num[0], num[1]]), formal_gps2pix(
                region_center))) > max_dist and 1.0 < num[0] < 900 and 1.0 < num[1] < 1350:
            max_dist = np.linalg.norm(
                np.subtract(np.array([num[0], num[1]]), formal_gps2pix(region_center)))
    return should_grow


def obtain_shapely_polygon():
    if len(region) == 0:
        reg = Polygon(
            [(0, 0), (0, np.size(map_data, 0)), (np.size(map_data, 1), np.size(map_data, 0)),
             (np.size(map_data, 1), 0)])
    else:
        reg = Polygon(region)
    print('reg is', reg)
    # logging.log(logging.INFO, "DRONE:reg is {}".format(reg))
    return reg


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

    return np.round(x_mass/mass).astype(np.int)


def find_vect_pos4region(sorted_vect_pos, reg):
    reg_poly = Polygon(reg)
    for pos in sorted_vect_pos:
        if reg_poly.contains(Point(pos)):
            return pos
    return None
    # print(slow_method_4_reg)
