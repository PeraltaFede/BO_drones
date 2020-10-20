import datetime
import time

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from bin.Agents import pathplanning_agent as ppa
from bin.Agents import simple_agent as sa
from bin.Coordinators.informed_coordinator import Coordinator
from bin.Environment.simple_env import Env
from bin.GUI.simple_mpl_gui import GUI
from bin.v2.Communications.simple_sender import Sender


class Simulator(object):
    def __init__(self, map_path2yaml, agents, main_sensor, render=False):
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
        self.environment = Env(map_path2yaml=map_path2yaml)
        self.agents = agents
        self.render = render
        self.sensors = set()

        self.init_maps()
        self.load_envs_into_agents()

        if main_sensor in self.sensors:
            self.main_sensor = main_sensor

        for agent in self.agents:
            agent.randomize_pos()

        self.sender = Sender()

        self.coordinator = Coordinator(self.agents, self.environment.grid, self.main_sensor)
        reads = [agent.read() for agent in self.agents]

        for i in range(1):
            self.agents[0].randomize_pos()
            reads.append(self.agents[0].read())

        print(reads)

        self.sender.send_new_sensor_msg(str(reads[0][0][0]) + "," + str(reads[0][0][1]) + "," + str(reads[0][1]))
        self.sender.send_new_sensor_msg(str(reads[1][0][0]) + "," + str(reads[1][0][1]) + "," + str(reads[1][1]))
        self.coordinator.initialize_data_gpr(reads)

        plt.imshow(self.environment.render_maps()["t"], origin='lower')
        plt.colorbar(orientation='vertical')
        plt.contour(self.environment.render_maps()["t"], colors='k', alpha=0.3, linewidths=1.0)
        # plt.title("Real Temperature Map")
        plt.draw()
        plt.pause(0.0001)
        # plt.show(block=True)

        if render:
            self.render = render
            self.GUI = GUI(self.environment.maps)
        else:
            self.render = False

    def init_maps(self):
        if isinstance(self.agents, sa.SimpleAgent):
            [self.sensors.add(sensor) for sensor in self.agents.sensors]
        elif isinstance(self.agents, list) and isinstance(self.agents[0], sa.SimpleAgent) or \
                isinstance(self.agents, list) and isinstance(self.agents[0], ppa.SimpleAgent):
            for agent in self.agents:
                [self.sensors.add(sensor) for sensor in agent.sensors]
        self.environment.add_new_map(self.sensors)

    def load_envs_into_agents(self):
        for agent in self.agents:
            agent.set_agent_env(self.environment)

    def export_maps(self, sensors=None, extension='png', render=False):
        data = self.environment.render_maps(sensors)

        mu, std, sensor_name = self.coordinator.surrogate(return_std=True, return_sensor=True)
        # self.coordinator.gp.predict(self.coordinator.vector_pos, return_std=True)

        img.imsave(
            "E:/ETSI/Proyecto/results/Map/{}_{}.{}".format(datetime.datetime.now().timestamp(),
                                                           "{}_u_gp".format(sensor_name), extension),
            1.96 * std.reshape(self.GUI.shape[1], self.GUI.shape[0]).T)
        img.imsave(
            "E:/ETSI/Proyecto/results/Map/{}_{}.{}".format(datetime.datetime.now().timestamp(),
                                                           "{}_gp".format(sensor_name), extension),
            mu.reshape(self.GUI.shape[1], self.GUI.shape[0]).T)

        for key, _map in data.items():
            if _map is not None:
                img.imsave(
                    "E:/ETSI/Proyecto/results/Map/{}_{}.{}".format(datetime.datetime.now().timestamp(), key, extension),
                    _map)
            else:
                print("sensor {} not saved".format(key))

    def run_simulation(self):
        imax = 20
        i = 0

        data = self.environment.render_maps()

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
                        self.sender.send_new_drone_msg(agent.pose)
                        agent.next_pose = self.coordinator.generate_new_goal(pose=agent.pose)
                        if agent.step():
                            time.sleep(1)
                            read = agent.read()
                            self.coordinator.add_data(read)

                            self.sender.send_new_sensor_msg(
                                str(read[0][0]) + "," + str(read[0][1]) + "," + str(read[1]))

                            self.coordinator.fit_data()

                            # plt.imshow(np.exp(-cdist([agent.pose[:2]],
                            #                          self.coordinator.all_vector_pos) / 150).reshape(1000, 1500).T,
                            #            origin='lower')
                        else:
                            i -= 1
            plt.title("MSE is {}".format(self.coordinator.get_mse(self.environment.maps['t'].T.flatten())))
            plt.draw()
            plt.pause(0.001)
            if self.render:
                mu, std, sensor_name = self.coordinator.surrogate(return_std=True, return_sensor=True)
                mse = self.coordinator.get_mse(self.environment.maps['t'].T.flatten())
                # titles = {"MSE": self.coordinator.get_mse(self.environment.maps["temp"].flatten())}
                # print(titles)
                # if isinstance(self.agents, sa.SimpleAgent):
                data["{} gp".format(sensor_name)] = mu
                data["{} gp un".format(sensor_name)] = std
                self.GUI.observe_maps(data)
                print(mse)
                if mse < 0.1:
                    break
            i += 1
        print("done")
        plt.show(block=True)
