from copy import deepcopy

import numpy as np


class SimpleAgent(object):
    agent_env = None

    def __init__(self, sensors, pose=np.zeros((3, 1)), ignore_obstacles=False):
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

    @staticmethod
    def set_agent_env(env):
        SimpleAgent.agent_env = env

    def reached_pose(self):
        return self.pose[0] == self.next_pose[0] and self.pose[1] == self.next_pose[1]

    def randomize_pos(self, near=False):
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
            return
        max_x = self.agent_env.grid.shape[1]-1
        max_y = self.agent_env.grid.shape[0]-1

        # np.random.s
        self.pose = np.round(np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)
        while self.agent_env.grid[self.pose[1], self.pose[0]] == 1 and not self.ignore_obstacles:
            self.pose = np.round(np.multiply(np.random.rand(3), np.array([max_x, max_y, 2 * np.pi]))).astype(np.int)

        self.next_pose = deepcopy(self.pose)
        # np.random.seed(seed=None)

    def step(self):
        if self.agent_env is not None:
            if 0 <= self.next_pose[0] <= self.agent_env.grid.shape[1] and \
                    0 <= self.next_pose[1] <= self.agent_env.grid.shape[0] and (
                    self.agent_env.grid[self.next_pose[1], self.next_pose[0]] == 0 or self.ignore_obstacles):
                self.pose = deepcopy(self.next_pose)
                print("{} did move to and will read {}".format(self, self.read()))
            else:
                print(0 <= self.next_pose[0] <= self.agent_env.grid.shape[1])
                print(0 <= self.next_pose[1] <= self.agent_env.grid.shape[0])
                print(self.agent_env.grid[self.next_pose[1], self.next_pose[0]] == 0)
                print("can't move to ", self.next_pose)
        else:
            print("NO MAP DATA ")

    def read(self):
        return [self.pose[:2], self.agent_env.maps[self.sensors[0]][self.pose[1], self.pose[0]]]
