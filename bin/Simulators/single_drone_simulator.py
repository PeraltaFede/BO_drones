import time

import matplotlib.pyplot as plt
import numpy as np

from bin.Agents import pathplanning_agent as ppa
from bin.Agents import simple_agent as sa
from bin.Coordinators.informed_coordinator import Coordinator
from bin.Environment.simple_env import Env
from bin.v2.Communications.simple_sender import Sender


class Simulator(object):
    def __init__(self, map_path2yaml, agents, main_sensor, saving=False, test_name="", acq="gaussian_sei",
                 acq_mod="normal", file=0):
        """
        Simulator(map_path2yaml, agents, render)

        Returns a new Simulator.

        Parameters
        ----------
        map_path2yaml : string
            Absolute path to map yaml archive  , e.g., ``"C:/data/yaml"``.
        agents : list of bin.Environment.simple_agent
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

        for agent in self.agents:
            agent.randomize_pos()
            agent.randomize_pos()
            # agent.randomize_pos()
            # agent.randomize_pos()

        self.sender = Sender()

        self.coordinator = Coordinator(self.environment.grid, self.main_sensor, acq=acq, acq_mod=acq_mod)
        self.sender.send_new_acq_msg(self.coordinator.acquisition)

        self.use_cih_as_initial_points = True
        if self.use_cih_as_initial_points:
            reads = [
                [np.array([785, 757]), self.environment.maps["t"][757, 785]],  # CSNB       (757, 785)
                # [np.array([492, 443]), self.environment.maps["t"][443, 492]],  # YVY        (443, 492)
                [np.array([75, 872]), self.environment.maps["t"][872, 75]],  # PMAregua   (872, 75)
                self.agents[0].read()
            ]
            self.sender.send_new_sensor_msg(str(reads[0][0][0]) + "," + str(reads[0][0][1]) + "," + str(reads[0][1]))
            self.sender.send_new_sensor_msg(str(reads[1][0][0]) + "," + str(reads[1][0][1]) + "," + str(reads[1][1]))
            self.sender.send_new_sensor_msg(str(reads[2][0][0]) + "," + str(reads[2][0][1]) + "," + str(reads[2][1]))
            # plt.plot(reads[0][0][0], reads[0][0][1], '^y', markersize=12, label="Previous Positions")
            # plt.plot(reads[1][0][0], reads[1][0][1], '^y', markersize=12)
            # plt.plot(reads[2][0][0], reads[2][0][1], '^y', markersize=12)

        else:
            reads = [agent.read() for agent in self.agents]

            for i in range(2):
                self.agents[0].randomize_pos()
                reads.append(self.agents[0].read())
            self.sender.send_new_sensor_msg(str(reads[0][0][0]) + "," + str(reads[0][0][1]) + "," + str(reads[0][1]))
            self.sender.send_new_sensor_msg(str(reads[1][0][0]) + "," + str(reads[1][0][1]) + "," + str(reads[1][1]))
            self.sender.send_new_sensor_msg(str(reads[2][0][0]) + "," + str(reads[2][0][1]) + "," + str(reads[2][1]))

        self.coordinator.initialize_data_gpr(reads)
        from copy import copy
        import matplotlib.cm as cm
        current_cmap = copy(cm.get_cmap("inferno"))
        current_cmap.set_bad(color="#eaeaf2")

        plt.imshow(self.environment.render_maps()["t"], origin='lower', cmap=current_cmap)  # YlGn_r
        cbar = plt.colorbar(orientation='vertical')
        cbar.ax.tick_params(labelsize=20)
        CS = plt.contour(self.environment.render_maps()["t"], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                         alpha=0.6, linewidths=1.0)
        plt.clabel(CS, inline=1, fontsize=10)
        # plt.title("Mask", fontsize=30)
        plt.xlabel("x", fontsize=20)
        plt.ylabel("y", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.draw()
        plt.pause(0.0001)
        # plt.show(block=True)

        if saving:
            self.f = open("E:/ETSI/Proyecto/results/csv_results/{}_{}.csv".format(test_name, int(time.time())), "a")
            self.f.write("kernel,acq,masked\n")
            self.f.write(str(
                "{},{},{}\n".format(self.coordinator.k_name, self.coordinator.acquisition, self.coordinator.acq_mod)))
            self.f.write("step,mse,t_dist\n")
            mse = self.coordinator.get_mse(self.environment.maps['t'].T.flatten())
            self.f.write("{},{},{}\n".format(0, mse, self.agents[0].distance_travelled))

    def init_maps(self):
        if isinstance(self.agents, sa.SimpleAgent):
            [self.sensors.add(sensor) for sensor in self.agents.sensors]
        elif isinstance(self.agents, list) and isinstance(self.agents[0], sa.SimpleAgent) or \
                isinstance(self.agents, list) and isinstance(self.agents[0], ppa.SimpleAgent):
            for agent in self.agents:
                [self.sensors.add(sensor) for sensor in agent.sensors]
        self.environment.add_new_map(self.sensors, file=self.file_no)

    def load_envs_into_agents(self):
        for agent in self.agents:
            agent.set_agent_env(self.environment)

    def run_simulation(self):
        imax = 20
        i = 0

        while i < imax:
            if not self.sender.should_update():
                plt.pause(0.5)
                continue
            if isinstance(self.agents, sa.SimpleAgent):
                self.agents.next_pose = self.coordinator.generate_new_goal()
                self.agents.step()
                self.coordinator.add_data(self.agents.read())
                self.coordinator.fit_data()
            elif isinstance(self.agents, list) and isinstance(self.agents[0], sa.SimpleAgent) or \
                    isinstance(self.agents, list) and isinstance(self.agents[0], ppa.SimpleAgent):
                for agent in self.agents:
                    if agent.reached_pose():
                        agent.next_pose = self.coordinator.generate_new_goal(pose=agent.pose, idx=i)
                        if agent.step():
                            time.sleep(1)
                            read = agent.read()
                            self.coordinator.add_data(read)

                            self.sender.send_new_sensor_msg(
                                str(read[0][0]) + "," + str(read[0][1]) + "," + str(read[1]))
                            self.sender.send_new_drone_msg(agent.pose)
                            self.coordinator.fit_data()

                            # dataaa = np.exp(-cdist([agent.pose[:2]],
                            #                        self.coordinator.all_vector_pos) / 150).reshape(1000, 1500).T
                            # plt.imshow(dataaa,
                            #                 origin='lower', cmap='YlGn_r')
                            # if i == 0:
                            #     plt.colorbar(orientation='vertical')
                        else:
                            i -= 1
            mse = self.coordinator.get_mse(self.environment.maps['t'].T.flatten())
            plt.title("MSE is {}".format(mse))
            plt.draw()
            # plt.pause(2)
            i += 1
            # if i == 7:
            #     # if self.agents[0].distance_travelled > 1500:
            #     print(mse)
            #     with open('E:/ETSI/Proyecto/data/Databases/numpy_files/best_bo.npy', 'wb') as g:
            #         np.save(g, self.coordinator.surrogate().reshape((1000, 1500)).T)
            #     plt.show(block=True)

            if self.saving:
                self.f.write("{},{},{}\n".format(i, mse, self.agents[0].distance_travelled))
        print("done")
        plt.show(block=True)
        self.sender.client.disconnect()
        if self.saving:
            self.f.close()
