from copy import deepcopy

import numpy as np

from bin.Utils.path_planners import rrt_star


class SimpleAgent(object):
    regions = [[[479.8, 967.26], [0., 1183.96], [0, 1500.], [1000., 1500.], [1000., 1141.71043299]],
               [[227.25, 321.73], [0., 56.92], [0, 0], [1000, 0], [1000., 798.35]],
               [[479.79413465, 967.26024708], [227.25213545, 321.73246836], [0., 56.91883525], [0., 1183.96]],
               [[227.25213545, 321.73246836], [479.79413465, 967.26024708], [1000., 1141.71], [1000., 798.35]]]

    def __init__(self, sensors, pose=np.zeros((3, 1)), ignore_obstacles=False, voronoi_reg=-1, _id=0):
        """
        SimpleAgent(sensors)

        Returns a new SimpleAgent with defined sensors.

        Parameters
        ----------
        sensors : string or list of strings
            Sensors on-board, e.g., ``["ph", "temp"]`` or ``"temp"``.

        Returns
        -------
        out : SimpleAgent

        See Also
        --------

        Examples
        --------
        """
        self.sensors = sensors
        self.pose = pose
        self.next_pose = np.zeros((3, 1))
        self.ignore_obstacles = ignore_obstacles
        self.position_flag = True
        self.position_error = np.round(5 * (np.random.rand(3) - 0.5)).astype(np.int)
        self.voronoi_reg = voronoi_reg
        self.agent_env = None
        self.path = None
        self.drone_id = _id
        self.distance_travelled = 0

    def set_agent_env(self, env):
        if self.voronoi_reg == -1:
            self.agent_env = env
        else:
            self.agent_env = env

    def reached_pose(self):
        return self.pose[0] == self.next_pose[0] and self.pose[1] == self.next_pose[1]

    def randomize_pos(self, near=False):
        # np.random.seed(0)
        if near:
            max_x = 10
            max_y = 10
            original_pose = deepcopy(self.pose)
            self.pose = original_pose + np.round(
                np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)
            while self.agent_env.grid[self.pose[1], self.pose[0]] == 1 and not self.ignore_obstacles:
                self.pose = original_pose + np.round(
                    np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)

            self.next_pose = deepcopy(self.pose)
            # np.random.seed(None)
            return
        max_x = self.agent_env.grid.shape[1] - 1
        max_y = self.agent_env.grid.shape[0] - 1

        # np.random.s
        self.pose = np.round(np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)
        while self.agent_env.grid[self.pose[1], self.pose[0]] == 1 and not self.ignore_obstacles:
            self.pose = np.round(np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)

        self.next_pose = deepcopy(self.pose)
        # np.random.seed(None)
        # np.random.seed(seed=None)

    def simulate_step(self, start, finish, m, b):
        # s = start + vo
        deltax = 0.51 * np.sign(finish[0] - start[0])
        delta = [start[0] + deltax, m * (start[0] + deltax) + b, 0]
        # print('---', delta, deltax)
        return np.round(delta).astype(np.int)

    def line_crosses_obstacle(self, start, finish):
        den = (finish[0] - start[0])
        m = (finish[1] - start[1]) / den if den != 0 else 0
        b = start[1] - m * start[0]
        # y = mx + b
        pos = start.copy()

        while pos[0] != finish[0] and pos[1] != finish[1]:
            pos = self.simulate_step(pos, finish, m, b)
            if self.agent_env.grid[pos[1], pos[0]] == 1:
                return True
        return False

    def path_plan(self):

        if self.line_crosses_obstacle(self.pose, self.next_pose):
            print('collision inbound, calculating route via rrt')
            # self.path = None
            path = rrt_star(self.agent_env.grid, self.pose, self.next_pose)
            if path[0, 0] == path[1, 0] and path[0, 1] == path[1, 1]:
                path = path[1:, :]
            indexes = np.unique(path, axis=0, return_index=True)[1]
            path = np.array([path[index] for index in sorted(indexes)])
            c_end = self.pose
            p2f = []
            ma = 10000
            z = 0
            while True:
                z += 1
                for i in range(len(path)):
                    if c_end[0] == path[-i - 1][0] and c_end[1] == path[-i - 1][1] or len(path) == i + 1:
                        p2f.append(path[-i])
                        c_end = path[-i]
                        break
                    if not self.line_crosses_obstacle(c_end[:2], path[-i - 1]):
                        p2f.append(path[-i - 1])
                        c_end = path[-i - 1]
                        break
                if ma < z:
                    # print('pause')
                    print(p2f)
                    print(path)
                    print(self.pose)
                    print(self.next_pose)
                    self.path = p2f
                    break
                if c_end[0] == path[-1][0] and c_end[1] == path[-1][1]:
                    # print('final path is', p2f)
                    self.path = p2f
                    break

        else:
            # print('no collision found')
            self.path = [self.next_pose]

    def step(self):
        if self.agent_env is not None:
            if self.position_flag:
                next_pose = self.next_pose + self.position_error
                self.position_error = np.round(4 * (np.random.rand(3) - 0.5)).astype(np.int)
                while self.agent_env.grid[next_pose[1], next_pose[0]] != 0:
                    # print('retrying')
                    next_pose = self.next_pose + self.position_error
                    self.position_error = np.round(4 * (np.random.rand(3) - 0.5)).astype(np.int)

                self.next_pose = next_pose

            self.path_plan()

            if 0 <= self.next_pose[0] <= self.agent_env.grid.shape[1] and \
                    0 <= self.next_pose[1] <= self.agent_env.grid.shape[0] and (
                    self.agent_env.grid[self.next_pose[1], self.next_pose[0]] == 0 or self.ignore_obstacles):

                for goal in self.path:
                    # print(np.linalg.norm(self.pose[:2] - goal[:2]))
                    # print(goal)
                    # print(self.pose)
                    self.distance_travelled += np.linalg.norm(self.pose[:2] - goal[:2])
                    self.pose = deepcopy(goal)
                # print(self.pose, " vs ", self.next_pose)
                self.pose = deepcopy(self.next_pose)
                print("{}, TD={}, will read {}".format(self.drone_id, self.distance_travelled, self.read()))
                return True
            else:
                print(0 <= self.next_pose[0] <= self.agent_env.grid.shape[1])
                print(0 <= self.next_pose[1] <= self.agent_env.grid.shape[0])
                print(self.agent_env.grid[self.next_pose[1], self.next_pose[0]] == 0)
                print("can't move to ", self.next_pose)
        else:
            print("NO MAP DATA ")

        return False

    def read(self):
        return [self.pose[:2], self.agent_env.maps[self.sensors[0]][self.pose[1], self.pose[0]]]
