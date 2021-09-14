from copy import deepcopy
from sys import path
from time import time

from numpy import mean
from numpy.linalg import norm

from bin.Environment.simple_env import Env
from bin.v2.Communications.simple_sender import Sender
from bin.v2.Comparison.moo_multi.mm_ga_coordinator import Coordinator as GACoordinator
from bin.v2.Comparison.moo_multi.mm_lm_coordinator import Coordinator as LMCoordinator


class GymEnvironment(object):
    def __init__(self, map_path2yaml, agents: list, acq="GALM", saving=False, acq_fusion="GALM",
                 acq_mod="truncated", id_file=0, render2gui=True, initial_pos="circle", name_file="", d=1.0):
        """

        :param map_path2yaml: file path to mapyaml
        :param agents: list of simple agents. ALL agents have the same set of sensors
        :param acq: str of the acquisition function name
        :param acq_mod: str of the modification of the AF
        :param id_file: int unique file id 0, 1, 2, ... , n
        :param render2gui: bool if visualization mode is online (performed through MQTT)
        """
        # instancing variables
        self.environment = Env(map_path2yaml=map_path2yaml)
        self.noiseless_maps = False
        self.agents = agents
        self.file_no = id_file
        self.sensors = set()
        self._init_maps()
        if acq == "ga":
            self.coordinator = GACoordinator(self.environment.grid, self.sensors, d=d, no_drones=len(agents))
        elif acq == "lm":
            self.coordinator = LMCoordinator(self.environment.grid, self.sensors, d=d, no_drones=len(agents),
                                             acq=id_file)
        self.timestep = 0

        self._load_envs_into_agents()
        for agent in enumerate(self.agents):
            aux_f = agent[1].position_flag
            agent[1].position_flag = False
            agent[1].pose = self.coordinator.current_goals[agent[0]]
            agent[1].next_pose = deepcopy(agent[1].pose)
            agent[1].position_flag = aux_f
        # initial readings
        # import matplotlib.pyplot as plt
        # for sx, sy, cg in zip(self.coordinator.fpx, self.coordinator.fpy, self.coordinator.current_goals):
        #     plt.plot(sx, sy, '-x')
        #     plt.plot([cg[0], sx[0]], [cg[1], sy[0]], '-x')
        # for a in agents:
        #     plt.plot(a.pose[0], a.pose[1], '*')
        # plt.show(block=True)

        reads = []
        [reads.append(read.read()) for read in self.agents]
        self.coordinator.initialize_data_gpr(reads)

        self.saving = saving
        if self.saving:
            self.f = open(
                path[-1] + "/results/MAMS/{}_{}_{}.csv".format(name_file, int(time()), self.file_no), "a")
            self.f.write("n_agent,n_sensors,acq_fusion,kernels,acq,acq_mod,prop\n")
            self.f.write(str(
                "{},{},{},{},{},{},{}\n".format(len(self.agents), len(self.sensors), acq_fusion,
                                                len(self.coordinator.gps), acq,
                                                acq_mod, d)))
            # mses, scores, keys = self.reward()
            scores, keys = self.reward()
            titles = ""
            for sensor in keys:
                titles += f',score_{sensor}'
                # titles += f',mse_{sensor},score_{sensor}'
            results = ""
            # for mse, score in zip(mses, scores):
            #     results += f",{mse},{score}"
            for bic in scores:
                results += f",{bic}"
            # self.f.write(
            #     "step,qty,time,t_dist,avg_mse,avg_score{}\n".format(titles))
            self.f.write("step,qty,time,t_dist,avg_score{}\n".format(titles))
            # self.f.write("{},{},{},{},{},{}{}\n".format(self.timestep, len(self.coordinator.train_inputs), 0,
            #                                             sum(c.distance_travelled for c in self.agents) / len(
            #                                                 self.agents), mean(mses), mean(scores), results))
            self.f.write(
                "{},{},{},{},{}{}\n".format(self.timestep, len(self.coordinator.train_inputs), 0,
                                            sum(c.distance_travelled for c in self.agents) / len(
                                                self.agents), mean(scores), results))
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
        self.environment.add_new_map(self.sensors, file=self.file_no, clone4noiseless=self.noiseless_maps)

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
                    results += ",-1"
                self.f.write("{},{},{},{},{}{}\n".format(self.timestep, len(self.coordinator.train_inputs), 0,
                                                         sum(c.distance_travelled for c in self.agents) / len(
                                                             self.agents), -1, results))
                #     results += ",-1,-1"
                # self.f.write("{},{},{},{},{},{}{}\n".format(self.timestep, len(self.coordinator.train_inputs), 0,
                #                                             sum(c.distance_travelled for c in self.agents) / len(
                #                                                 self.agents), -1, -1, results))
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
        # mses, scores, keys = self.reward()
        scores, keys = self.reward()
        # print(self.coordinator.train_inputs)
        # print([agent.distance_travelled for agent in self.agents])
        if self.saving:
            results = ""
            for score in scores:
                results += f",{score}"
            # results = ""
            # for mse, score in zip(mses, scores):
            #     results += f",{mse},{score}"
            # self.f.write(
            #     "{},{},{},{},{},{}{}\n".format(self.timestep, len(self.coordinator.train_inputs), time() - self.t0,
            #                                    sum(c.distance_travelled for c in self.agents) / len(
            #                                        self.agents), mean(mses), mean(scores), results))
            self.f.write(
                "{},{},{},{},{}{}\n".format(self.timestep, len(self.coordinator.train_inputs), time() - self.t0,
                                            sum(c.distance_travelled for c in self.agents) / len(
                                                self.agents), mean(scores), results))
            self.t0 = time()
        return scores

    def reward(self):
        if self.noiseless_maps:
            # mses = [self.coordinator.get_mse(self.environment.maps[f"noiseless_{key}"].flatten(), key) for key in
            #         self.coordinator.gps.keys()]
            scores = [self.coordinator.get_score(self.environment.maps[f"noiseless_{key}"].flatten(), key) for key in
                      self.coordinator.gps.keys()]
        else:
            # mses = [self.coordinator.get_mse(self.environment.maps[key].flatten(), key) for key in
            #         self.coordinator.gps.keys()]
            scores = [self.coordinator.get_score(self.environment.maps[key].flatten(), key) for key in
                      self.coordinator.gps.keys()]
        # return mses, scores, self.coordinator.gps.keys()
        return scores, self.coordinator.gps.keys()
