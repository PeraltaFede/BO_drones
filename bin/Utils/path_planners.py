import numpy as np


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


def get_nearest_index(nodes_list, rnd):
    dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
             ** 2 for node in nodes_list]
    minind = dlist.index(min(dlist))
    return minind


def collision_check(new_node, parent_node, map_data):
    rounded_nn_x = int(round(new_node.x))
    rounded_nn_y = int(round(new_node.y))
    rounded_pn_x = int(round(parent_node.x))
    rounded_pn_y = int(round(parent_node.y))

    if rounded_nn_x > 999:
        rounded_nn_x = 999
    if rounded_nn_y > 1499:
        rounded_nn_y = 1499

    if map_data[rounded_nn_y, rounded_nn_x] == 1:
        return False

    m = (new_node.y - parent_node.y) / (new_node.x - parent_node.x)
    b = new_node.y - m * new_node.x
    if rounded_nn_x < rounded_pn_x:
        min_x = rounded_nn_x
        max_x = rounded_pn_x
    else:
        min_x = rounded_pn_x
        max_x = rounded_nn_x
    if rounded_nn_y < rounded_pn_y:
        min_y = rounded_nn_y
        max_y = rounded_pn_y
    else:
        min_y = rounded_pn_y
        max_y = rounded_nn_y

    bounds = map_data[min_y:max_y + 1, min_x:max_x + 1]
    i, j = np.where(bounds == 1)
    for k in range(len(i)):
        if i[k] + min_y == int(round(m * (min_x + j[k]) + b)):
            # print(' crosses obstacle')
            return False
    return True


def rrt_star(map_data, current_pose, goal_pose):
    c_pose = Node(current_pose[0], current_pose[1])
    g_pose = Node(goal_pose[0], goal_pose[1])

    node_list = [c_pose]
    y_max, x_max = map_data.shape

    stuck_minima = 0

    while True:

        # switch randomly to random or goal as point
        if np.random.randint(0, 100) > 40:
            rnd = [np.random.uniform(0, x_max), np.random.uniform(
                0, y_max)]
        else:
            rnd = [g_pose.x, g_pose.y]

        # Find nearest node
        nind = get_nearest_index(node_list, rnd)
        nearest_node = node_list[nind]
        if np.round(rnd[1] - nearest_node.y) == 0 and np.round(rnd[0] - nearest_node.x) == 0:
            print('same node')
            continue
        # steering
        theta = np.math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)

        if stuck_minima < 100:
            new_node = Node(node_list[nind].x + np.random.randint(5, 50) * np.math.cos(theta),
                            node_list[nind].y + np.random.randint(5, 50) * np.math.sin(theta))
        elif 101 < stuck_minima < 500:
            print('what')
            new_node = Node(node_list[nind].x + np.random.randint(2, 10) * np.math.cos(theta),
                            node_list[nind].y + np.random.randint(2, 10) * np.math.sin(theta))
        else:
            print('lol')
            new_node = Node(node_list[nind].x + np.random.randint(2, 5) * np.math.cos(theta),
                            node_list[nind].y + np.random.randint(2, 5) * np.math.sin(theta))

        new_node.parent = nind

        # print(new_node.x, ' ', new_node.y)
        if not collision_check(new_node, node_list[nind], map_data):
            stuck_minima += 1
            continue

        stuck_minima = 0
        node_list.append(new_node)
        # check goal
        dx = new_node.x - g_pose.x
        dy = new_node.y - g_pose.y
        d = np.math.sqrt(dx * dx + dy * dy)
        if d <= 50 and collision_check(g_pose, new_node, map_data):
            break

        # plt.plot([new_node.x, node_list[new_node.parent].x],
        #          [new_node.y, node_list[new_node.parent].y], color='blue', marker='o', markersize=0.5)
        # plt.show()
        # figure.canvas.draw()

    path2follow = np.array([np.array([g_pose.x]), np.array([g_pose.y])])
    path2follow = np.expand_dims(path2follow, 0)
    last_index = len(node_list) - 1

    while True:
        node = node_list[last_index]
        to_add = np.array([np.array([node.x]), np.array([node.y])])
        path2follow = np.concatenate([path2follow, [to_add]])
        last_index = node.parent
        if node_list[last_index].parent is None:
            to_add = np.array([np.array([node.x]), np.array([node.y])])
            path2follow = np.concatenate([path2follow, [to_add]])
            break

    return np.round(np.flipud(path2follow).reshape((-1, 2))).astype(np.int)
