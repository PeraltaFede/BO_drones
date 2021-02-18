from copy import deepcopy
from sys import path
from time import time

from numpy import mean
from numpy.linalg import norm

from bin.Agents.gym_agent import SimpleAgent as Ga
from bin.Coordinators.gym_coordinator import Coordinator
from bin.Environment.simple_env import Env
from bin.Utils.utils import get_init_pos4
from bin.v2.Communications.simple_sender import Sender


class GymEnvironment(object):
    def __init__(self, map_path2yaml, agents: list, acq="gaussian_ei", saving=False, acq_fusion="max_sum",
                 acq_mod="normal", id_file=0, render2gui=True, initial_pos="circle", name_file=""):
        """

        :param map_path2yaml: file path to mapyaml
        :param agents: list of simple agents. ALL agents have the same set of sensors
        :param acq: str of the acquisition function name
        :param acq_mod: str of the modification of the AF
        :param id_file: int unique file id [0, 1, 2, ... , n]
        :param render2gui: bool if visualization mode is online (performed through MQTT)
        """
        # instancing variables
        self.environment = Env(map_path2yaml=map_path2yaml)
        self.agents = agents
        for agent in self.agents:
            assert isinstance(agent, Ga), "All agents should be instances of gym.SimpleAgent"
        self.file_no = id_file
        self.sensors = set()
        self._init_maps()
        self.coordinator = Coordinator(self.environment.grid, self.sensors, acq=acq, acq_mod=acq_mod,
                                       acq_fusion=acq_fusion)
        self.timestep = 0

        # initializing environment
        self._load_envs_into_agents()
        # initialize drone positions
        initial_positions = get_init_pos4(n=len(agents), map_data=self.environment.grid, expand=True)
        for agent in self.agents:
            aux_f = agent.position_flag
            agent.position_flag = False
            if initial_pos == "circle":
                agent.pose, initial_positions = initial_positions[0, :], initial_positions[1:, :]
            else:
                agent.randomize_pos()
            agent.next_pose = deepcopy(agent.pose)
            agent.position_flag = aux_f
        # initial readings
        reads = []
        [reads.append(read.read()) for read in self.agents]
        self.coordinator.initialize_data_gpr(reads)

        self.saving = saving
        if self.saving:
            self.f = open(
                path[-1] + "/results/SAMS/{}_{}_{}.csv".format(name_file, int(time()), self.file_no), "a")
            self.f.write("n_agent,n_sensors,acq_fusion,kernels,acq,acq_mod\n")
            self.f.write(str(
                "{},{},{},{},{},{}\n".format(len(self.agents), len(self.sensors), acq_fusion, len(self.coordinator.gps),
                                             self.coordinator.acquisition, self.coordinator.acq_mod)))
            mses, scores, keys = self.reward()
            titles = ""
            for sensor in keys:
                titles += f',mse_{sensor},score_{sensor}'
            results = ""
            for mse, score in zip(mses, scores):
                results += f",{mse},{score}"
            self.f.write(
                "step,qty,time,t_dist,avg_mse,avg_score{}\n".format(titles))
            self.f.write("{},{},{},{},{},{}{}\n".format(self.timestep, len(self.coordinator.train_inputs), 0,
                                                        sum(c.distance_travelled for c in self.agents) / len(
                                                            self.agents), mean(mses), mean(scores), results))
            self.t0 = time()

        self.render2gui = render2gui
        if render2gui:
            self.sender = Sender()
            self.sender.send_new_acq_msg(self.coordinator.acquisition)
            for agent in self.agents:
                for sensor in agent.sensors:
                    self.sender.send_new_sensor_msg(
                        str(agent.read()["pos"][0]) + "," + str(agent.read()["pos"][1]) + "," + str(
                            agent.read()[sensor]), sensor=sensor,
                        _id=agent.drone_id)
                self.sender.send_new_drone_msg(agent.pose, agent.drone_id)
                self.sender.send_new_goal_msg(agent.next_pose, agent.drone_id)

    def _init_maps(self):
        """
        Loads the sensor names (str) into a set.
        Loads ground truth maps in the environment for reading.
        """
        for agent in self.agents:
            [self.sensors.add(sensor) for sensor in agent.sensors]
        self.environment.add_new_map(self.sensors, file=self.file_no)

    def _load_envs_into_agents(self):
        """
         After the environment is generated, each drone obtains access to the environment
        """
        for agent in self.agents:
            agent.set_agent_env(self.environment)

    def render(self):
        """

        :return: list of std, u foreach sensor
        """
        return self.coordinator.surrogate()

    def reset(self):
        del self.coordinator

    def _select_next_drone(self):
        idx = -1
        dist2next = 1000000
        for i in range(len(self.agents)):
            future_dist = 0
            if len(self.agents[i].path) == 0:
                break
            for k in range(len(self.agents[i].path) - 1):
                future_dist += norm(self.agents[i].path[k][:2] - self.agents[i].path[k + 1][:2])
            future_dist += norm(self.agents[i].pose[:2] - self.agents[i].path[0][:2])
            if future_dist == 0.0:
                idx = -1
                break
            if future_dist < dist2next:
                idx = i
                dist2next = future_dist
        return idx, dist2next

    def step(self, action):
        for pose, agent in zip(action, self.agents):
            if len(pose) > 0:
                agent.next_pose = pose

        next_idx, dist2_simulate = self._select_next_drone()
        if next_idx == -1:
            self.timestep += 1
            if self.saving:

                results = ""
                for i in range(len(self.sensors)):
                    results += ",-1,-1"
                self.f.write("{},{},{},{},{},{}{}\n".format(self.timestep, len(self.coordinator.train_inputs), 0,
                                                            sum(c.distance_travelled for c in self.agents) / len(
                                                                self.agents), -1, -1, results))
            return -1, -1
        for agent in self.agents:
            if agent.step(dist_left=dist2_simulate):
                if agent.reached_pose():
                    read = agent.read()
                    self.coordinator.add_data(read)
                    if self.render2gui:
                        for sensor in agent.sensors:
                            self.sender.send_new_sensor_msg(
                                str(agent.read()["pos"][0]) + "," + str(agent.read()["pos"][1]) + "," + str(
                                    agent.read()[sensor]), sensor=sensor,
                                _id=agent.drone_id)
                    self.coordinator.fit_data()
            if self.render2gui:
                self.sender.send_new_drone_msg(agent.pose, agent.drone_id)

        self.timestep += 1
        if self.saving:
            mses, scores, keys = self.reward()
            results = ""
            for mse, score in zip(mses, scores):
                results += f",{mse},{score}"
            # print(keys)
            self.f.write(
                "{},{},{},{},{},{}{}\n".format(self.timestep, len(self.coordinator.train_inputs), time() - self.t0,
                                               sum(c.distance_travelled for c in self.agents) / len(
                                                   self.agents), mean(mses), mean(scores), results))
            self.t0 = time()
        return scores

    def reward(self):
        mses = [self.coordinator.get_mse(self.environment.maps[key].flatten(), key) for key in
                self.coordinator.gps.keys()]
        scores = [self.coordinator.get_score(self.environment.maps[key].flatten(), key) for key in
                  self.coordinator.gps.keys()]
        return mses, scores, self.coordinator.gps.keys()
