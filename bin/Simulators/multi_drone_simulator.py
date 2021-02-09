import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from bin.Agents import simulated_pathplanning_agent as sppa
from bin.Coordinators.multi_informed_coordinator import Coordinator
from bin.Environment.simple_env import Env
from bin.Utils.utils import get_init_pos4
from bin.v2.Communications.simple_sender import Sender


class Simulator(object):
    def __init__(self, map_path2yaml, agents: list, main_sensor, saving=False, test_name="", acq="gaussian_sei",
                 acq_mod="normal", file=0, initial_pos="circle"):
        """
        Simulator(map_path2yaml, agents, render)

        Returns a new Simulator.

        Parameters
        ----------
        map_path2yaml : string
            Absolute path to map yaml archive  , e.g., ``"C:/data/yaml"``.
        agents : list of sppa.SimpleAgent
            Agents

        Returns
        -------
        out : Simulator

        See Also
        --------

        Examples
        --------
        """
        self.saving = saving
        self.file_no = file

        self.environment = Env(map_path2yaml=map_path2yaml)
        self.agents = agents
        self.sensors = set()

        self.init_maps()
        self.load_envs_into_agents()

        if main_sensor in self.sensors:
            self.main_sensor = main_sensor
        self.sender = Sender()
        # if initial_pos == "none":
        #     print("random initial positions")
        # elif initial_pos == "circle":
        #     print("circle initial positions")
        #     initial_positions = get_init_pos4(n=len(agents))
        # elif initial_pos == "cvt":
        #     print("cvt initial positions")
        initial_positions = get_init_pos4(n=len(agents), map_data=self.environment.grid, expand=True)
        for agent in self.agents:
            aux_f = agent.position_flag
            agent.position_flag = False
            if initial_pos == "circle":
                agent.pose, initial_positions = initial_positions[0, :], initial_positions[1:, :]
            else:
                agent.randomize_pos()
                agent.randomize_pos()
            # agent.randomize_pos(near=False)
            agent.next_pose = deepcopy(agent.pose)
            agent.position_flag = aux_f
            self.sender.send_new_drone_msg(agent.pose, agent.drone_id)
            self.sender.send_new_goal_msg(agent.next_pose, agent.drone_id)

        self.coordinator = Coordinator(self.environment.grid, self.main_sensor, acq=acq, acq_mod=acq_mod)
        self.sender.send_new_acq_msg(self.coordinator.acquisition)

        reads = [
            # [np.array([785, 757]), self.environment.maps["t"][757, 785]],  # CSNB       (757, 785)
            # [np.array([492, 443]), self.environment.maps["t"][443, 492]],  # YVY        (443, 492)
            # [np.array([75, 872]), self.environment.maps["t"][872, 75]],  # PMAregua   (872, 75)
        ]
        [reads.append(read.read()) for read in self.agents]
        for [read, my_id] in zip(reads, [0, 1, 2, 3, 4]):
            self.sender.send_new_sensor_msg(str(read[0][0]) + "," + str(read[0][1]) + "," + str(read[1]),
                                            _id=my_id)

            self.sender.send_new_drone_msg(self.agents[my_id].pose, my_id)
            plt.pause(0.01)
            # plt.plot(reads[0][0][0], reads[0][0][1], '^y', markersize=12, label="Previous Positions")
        self.coordinator.initialize_data_gpr(reads)

        # from copy import copy
        # import matplotlib.cm as cm
        # current_cmap = copy(cm.get_cmap("inferno"))
        # current_cmap.set_bad(color="#eaeaf2")
        #
        # plt.imshow(self.environment.render_maps()["t"], origin='lower', cmap=current_cmap)  # YlGn_r
        # cbar = plt.colorbar(orientation='vertical')
        # cbar.ax.tick_params(labelsize=20)
        # CS = plt.contour(self.environment.render_maps()["t"], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
        #                  alpha=0.6, linewidths=1.0)
        # plt.clabel(CS, inline=1, fontsize=10)
        # # plt.title("Mask", fontsize=30)
        # plt.xlabel("x", fontsize=20)
        # plt.ylabel("y", fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.draw()
        # plt.pause(0.0001)
        # plt.show(block=True)
        if saving:
            self.f = open(
                "E:/ETSI/Proyecto/results/multiagent/{}_{}_{}.csv".format(test_name, int(time.time()), self.file_no),
                "a")
            self.f.write("num,acq,masked\n")
            self.f.write(str(
                "{},{},{}\n".format(len(self.agents), self.coordinator.acquisition, self.coordinator.acq_mod)))
            self.f.write("step,mse,qty,time,t_dist\n")
            mse = self.coordinator.get_mse(self.environment.maps['t'].T.flatten())
            self.f.write("{},{},{},{},{}\n".format(0, mse, len(self.coordinator.data[1]), 0,
                                                   sum(c.distance_travelled for c in self.agents) / len(self.agents)))

    def init_maps(self):
        for agent in self.agents:
            [self.sensors.add(sensor) for sensor in agent.sensors]
        self.environment.add_new_map(self.sensors, file=self.file_no)

    def load_envs_into_agents(self):
        for agent in self.agents:
            agent.set_agent_env(self.environment)

    def select_next_drone(self):
        idx = -1
        dist2next = 1000000
        for i in range(len(self.agents)):
            future_dist = 0
            # print(self.agents[0].pose)
            for k in range(len(self.agents[i].path) - 1):
                future_dist += np.linalg.norm(self.agents[i].path[k][:2] - self.agents[i].path[k + 1][:2])
                # print(self.agents[i].path[k][:2], self.agents[i].path[k + 1][:2])
            future_dist += np.linalg.norm(self.agents[i].pose[:2] - self.agents[i].path[0][:2])
            # print(self.agents[i].pose[:2], self.agents[i].path[0][:2])
            # print(i, future_dist)
            if future_dist < dist2next:
                idx = i
                dist2next = future_dist
            # if np.linalg.norm(self.agents[i].pose[:2] - self.agents[i].next_pose[:2]) < dist2next:
            #     idx = i
            #     dist2next = np.linalg.norm(self.agents[i].pose[:2] - self.agents[i].next_pose[:2])
        return idx, dist2next

    def run_simulation(self):
        imax = 24
        i = 0

        while i < imax:
            # if not self.sender.should_update():
            #     plt.pause(0.5)
            #     continue

            # t0 = time.time()
            for agent in self.agents:
                if agent.reached_pose():
                    print(agent.drone_id, "reached")
                    agent.next_pose = self.coordinator.generate_new_goal(pose=agent.pose, idx=i,
                                                                         other_poses=[agt.pose for agt in self.agents if
                                                                                      agt is not agent])
                    self.sender.send_new_goal_msg(agent.next_pose, agent.drone_id)

                else:
                    print(agent.drone_id, "not reached: ", agent.pose, agent.next_pose, agent.path)

            next_idx, dist2_simulate = self.select_next_drone()
            for agent in self.agents:
                print(agent.drone_id, " is next")
                if agent.step(dist_left=dist2_simulate):
                    if agent.reached_pose():
                        # time.sleep(0.1)
                        read = agent.read()
                        self.coordinator.add_data(read)
                        self.sender.send_new_sensor_msg(
                            str(read[0][0]) + "," + str(read[0][1]) + "," + str(read[1]), agent.drone_id)
                        self.coordinator.fit_data()
                # else:
                #     i -= 1
                self.sender.send_new_drone_msg(agent.pose, agent.drone_id)
            mse = self.coordinator.get_mse(self.environment.maps['t'].T.flatten())
            score = self.coordinator.get_score(self.environment.maps['t'].T.flatten())

            # Theta0 = np.logspace(np.log(self.coordinator.gp.kernel_.theta / 10),
            #                      np.log(self.coordinator.gp.kernel_.theta + 10), 500)
            # LML = np.array(
            #     [-self.coordinator.gp.log_marginal_likelihood(np.log([Theta0[i]])) for i in range(Theta0.shape[0])])
            # plt.plot(Theta0, LML)
            # plt.plot(Theta0[np.where(LML == np.min(LML))], np.min(LML), '*')

            # plt.title("MSE is {}".format(mse))
            # plt.draw()
            # plt.pause(2)
            i += 1
            # if i == 7:
            #     # if self.agents[0].distance_travelled > 1500:
            #     print(mse)
            #     with open('E:/ETSI/Proyecto/data/Databases/numpy_files/best_bo.npy', 'wb') as g:
            #         np.save(g, self.coordinator.surrogate().reshape((1000, 1500)).T)
            #     plt.show(block=True)

            if self.saving:
                self.f.write(
                    "{},{},{},{},{}\n".format(i, mse, len(self.coordinator.data[1]), score,
                                              sum(c.distance_travelled for c in self.agents) / len(self.agents)))  #
        # print("done")
        plt.show(block=True)
        # import os
        # os.system("pause")
        self.sender.client.disconnect()
        if self.saving:
            self.f.close()
