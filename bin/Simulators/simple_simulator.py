import datetime

import matplotlib.image as img

from bin.Agents import simple_agent as sa
from bin.Coordinators.informed_coordinator import Coordinator
from bin.Environment.simple_env import Env
from bin.GUI.simple_mpl_gui import GUI


class Simulator(object):
    def __init__(self, map_path2yaml, agents, main_sensor, render=False):
        """
        Simulator(map_path2yaml, agents, render)

        Returns a new Simulator.

        Parameters
        ----------
        map_path2yaml : string
            Absolute path to map yaml archive  , e.g., ``"C:/data/yaml"``.
        agents : bin.Environment.simple_agent
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
        self.agents.randomize_pos()

        self.coordinator = Coordinator(self.agents, self.environment.grid, self.main_sensor)
        self.coordinator.initialize_data_gpr(self.agents.read())
        self.agents.randomize_pos(near=True)
        self.coordinator.add_data(self.agents.read())
        self.coordinator.fit_data()

        if render:
            self.render = render
            self.GUI = GUI(self.environment.maps)
        else:
            self.render = False

    def init_maps(self):
        if isinstance(self.agents, sa.SimpleAgent):
            [self.sensors.add(sensor) for sensor in self.agents.sensors]
        elif isinstance(self.agents, list) and isinstance(self.agents[0], sa.SimpleAgent):
            for agent in self.agents:
                [self.sensors.add(sensor) for sensor in agent.sensors]
        self.environment.add_new_map(self.sensors)

    def load_envs_into_agents(self):
        if isinstance(self.agents, sa.SimpleAgent):
            self.agents.set_agent_env(self.environment)
        elif isinstance(self.agents, list) and isinstance(self.agents[0], sa.SimpleAgent):
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
        imax = 10

        data = self.environment.render_maps()

        for i in range(imax):
            next_pos = self.coordinator.generate_new_goal()
            if isinstance(self.agents, sa.SimpleAgent):
                self.agents.next_pose = next_pos
                self.agents.step()
                self.coordinator.add_data(self.agents.read())
                self.coordinator.fit_data()
            elif isinstance(self.agents, list) and isinstance(self.agents[0], sa.SimpleAgent):
                for agent in self.agents:
                    agent.step()
            if self.render:

                mu, std, sensor_name = self.coordinator.surrogate(return_std=True, return_sensor=True)

                # titles = {"MSE": self.coordinator.get_mse(self.environment.maps["temp"].flatten())}
                # print(titles)
                # if isinstance(self.agents, sa.SimpleAgent):
                data["{} gp".format(sensor_name)] = mu
                data["{} gp un".format(sensor_name)] = std
                self.GUI.observe_maps(data)

        self.GUI.export_maps()
