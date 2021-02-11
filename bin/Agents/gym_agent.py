from copy import deepcopy

import numpy as np

from bin.Utils.path_planners import rrt_star


class SimpleAgent(object):

    def __init__(self, sensors, pose=np.zeros((3, 1)), ignore_obstacles=False, _id=0):
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
        self.ignore_obstacles = ignore_obstacles
        self.position_flag = True
        self.position_error = np.round(5 * (np.random.rand(3) - 0.5)).astype(np.int)
        self.agent_env = None
        self.path = None
        self.drone_id = _id
        self.distance_travelled = 0
        self.next_pose = np.zeros(3, )
        self.battery_left = 1.0

    @property
    def next_pose(self):
        return self._next_pose

    @next_pose.setter
    def next_pose(self, value):
        if self.agent_env is None:
            self._next_pose = value
            return
        if self.position_flag:
            val = value + self.position_error
            while self.agent_env.grid[np.round(val[1]).astype(np.int), np.round(val[0]).astype(np.int)] != 0:
                # print('retrying')
                self.position_error = np.round(5 * (np.random.rand(3) - 0.5)).astype(np.int)
                val = value + self.position_error
            self._next_pose = val
            self.path_plan()
        else:
            self._next_pose = value

    def set_agent_env(self, env):
        self.agent_env = env

    def reached_pose(self):
        return self.pose[0] == self.next_pose[0] and self.pose[1] == self.next_pose[1] and self.battery_left > 0.0

    def randomize_pos(self, near=False, maxv=10):
        if near:
            max_x = maxv
            max_y = maxv
            original_pose = deepcopy(self.pose)
            self.pose = original_pose + np.round(
                np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)
            while self.agent_env.grid[self.pose[1], self.pose[0]] == 1 and not self.ignore_obstacles:
                self.pose = original_pose + np.round(
                    np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)

            self.next_pose = deepcopy(self.pose)
            return
        max_x = self.agent_env.grid.shape[1] - 1
        max_y = self.agent_env.grid.shape[0] - 1

        self.pose = np.round(np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)
        while self.agent_env.grid[self.pose[1], self.pose[0]] == 1 and not self.ignore_obstacles:
            self.pose = np.round(np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)

        prevflag = self.position_flag
        self.position_flag = False
        self.next_pose = deepcopy(self.pose)
        self.position_flag = prevflag

    @staticmethod
    def simulated_step(start, angle, delta=0.51):
        return start + np.array([delta * np.cos(angle), delta * np.sin(angle)])

    def line_crosses_obstacle(self, start, finish):
        pos = start.copy()[:2]
        rnd_pos = np.round(pos).astype(np.int)
        diff = np.subtract(finish[:2], start[:2])
        angle = np.arctan2(diff[1], diff[0])
        while rnd_pos[0] != finish[0] and rnd_pos[1] != finish[1]:
            pos = self.simulated_step(pos, angle)
            rnd_pos = np.round(pos).astype(np.int)
            if self.agent_env.grid[rnd_pos[1], rnd_pos[0]] == 1:
                return True
        return False

    def path_plan(self):
        if self.line_crosses_obstacle(self.pose, self.next_pose):
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
                    self.path = p2f
                    # raise ValueError
                    break
                if c_end[0] == path[-1][0] and c_end[1] == path[-1][1]:
                    self.path = p2f
                    break

        else:
            self.path = [self.next_pose]

    def step(self, dist_left=None):
        if self.agent_env is not None:
            if 0 <= self.next_pose[0] <= self.agent_env.grid.shape[1] and \
                    0 <= self.next_pose[1] <= self.agent_env.grid.shape[0] and (
                    self.agent_env.grid[self.next_pose[1], self.next_pose[0]] == 0 or self.ignore_obstacles):
                if self.distance_travelled > 1500:
                    print('limited distance')
                    self.battery_left = 0.0
                    return True
                i2remove = 0
                for goal in self.path:
                    d = np.linalg.norm(self.pose[:2] - goal[:2])
                    if dist_left is None or (d <= dist_left):
                        self.distance_travelled += d
                        self.pose = deepcopy(goal)
                        dist_left -= d
                    else:
                        diff = np.subtract(goal[:2], self.pose[:2])
                        angle = np.arctan2(diff[1], diff[0])
                        while dist_left > 0.51:
                            new_pose = self.simulated_step(self.pose[:2], angle)
                            dist_left -= np.linalg.norm(np.subtract(new_pose[:2], self.pose[:2]))
                            self.distance_travelled += np.linalg.norm(np.subtract(new_pose[:2], self.pose[:2]))
                            self.pose = new_pose

                        new_pose = self.simulated_step(self.pose[:2], angle, delta=dist_left)
                        self.distance_travelled += np.linalg.norm(np.subtract(new_pose[:2], self.pose[:2]))
                        self.pose = np.round(np.append(new_pose, 0)).astype(np.int)
                        break
                    i2remove += 1
                del self.path[:i2remove]
                if dist_left is None:
                    self.pose = deepcopy(self.next_pose)
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
        """

        :rtype: dict
        """
        aux = {sensor: self.agent_env.maps[sensor][self.pose[1], self.pose[0]] for sensor in
               self.sensors}
        aux["pos"] = self.pose[:2]
        if self.distance_travelled > 1500:
            self.battery_left = 0.0
        return aux
