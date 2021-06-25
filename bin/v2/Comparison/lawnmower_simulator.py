import time

import matplotlib.pyplot as plt

from bin.Agents import pathplanning_agent as ppa
from bin.Agents import simple_agent as sa
from bin.Environment.simple_env import Env
from bin.v2.Comparison.lawmower_coordinator import Coordinator


class Simulator(object):
    def __init__(self, map_path2yaml, agents, main_sensor, saving=False, test_name="", acq=-1):
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

        self.environment = Env(map_path2yaml=map_path2yaml)
        self.agents = agents
        self.sensors = set()

        self.init_maps()
        self.load_envs_into_agents()

        if main_sensor in self.sensors:
            self.main_sensor = main_sensor

        for agent in self.agents:
            agent.randomize_pos()
            # agent.randomize_pos()
            # agent.randomize_pos()
            # agent.randomize_pos()

        if acq == 1:
            acq = "LD"
        elif acq == 2:
            acq = "LU"
        elif acq == 3:
            acq = "RD"
        elif acq == 4:
            acq = "RU"
        else:
            raise Exception

        self.coordinator = Coordinator(self.agents, self.environment.grid, self.main_sensor, acq)
        # self.coordinator.acquisition = "maxvalue_entropy_search"

        for agent in self.agents:
            agent.next_pose = self.coordinator.generate_new_goal(pose=agent.pose)
            agent.step()
            agent.distance_travelled = 0

        self.use_cih_as_initial_points = True
        if self.use_cih_as_initial_points:
            import numpy as np
            reads = [
                [np.array([785, 757]), self.environment.maps["t"][757, 785]],  # CSNB       (757, 785)
                # [np.array([492, 443]), self.environment.maps["t"][443, 492]],  # YVY        (443, 492)
                [np.array([75, 872]), self.environment.maps["t"][872, 75]],  # PMAregua   (872, 75)
                self.agents[0].read()
            ]
            print(reads)

        else:
            reads = [agent.read() for agent in self.agents]

            for i in range(2):
                self.agents[0].randomize_pos()
                reads.append(self.agents[0].read())

        self.coordinator.initialize_data_gpr(reads)

        plt.imshow(self.environment.render_maps()["t"], origin='lower', cmap='inferno')
        plt.colorbar(orientation='vertical')
        CS = plt.contour(self.environment.render_maps()["t"], colors=('gray', 'gray', 'gray', 'k', 'k', 'k', 'k'),
                         alpha=0.6, linewidths=1.0)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title("Ground Truth")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.draw()
        # plt.pause(0.0001)
        plt.show(block=True)

        if saving:
            self.f = open("E:/ETSI/Proyecto/results/csv_results/{}_{}.csv".format(test_name, int(time.time())), "a")
            self.f.write("kernel,acq,masked\n")
            self.f.write(str(
                "LM,{},{}\n".format(self.coordinator.acquisition, self.coordinator.acq_method)))
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
        self.environment.add_new_map(self.sensors, file=0)

    def load_envs_into_agents(self):
        for agent in self.agents:
            agent.set_agent_env(self.environment)

    def run_simulation(self):
        imax = 20
        i = 0

        while i < imax:
            # if not self.sender.should_update():
            #     plt.pause(0.5)
            #     continue
            if isinstance(self.agents, sa.SimpleAgent):
                self.agents.next_pose = self.coordinator.generate_new_goal()
                self.agents.step()
                self.coordinator.add_data(self.agents.read())
                self.coordinator.fit_data()
            elif isinstance(self.agents, list) and isinstance(self.agents[0], sa.SimpleAgent) or \
                    isinstance(self.agents, list) and isinstance(self.agents[0], ppa.SimpleAgent):
                for agent in self.agents:
                    if agent.reached_pose():
                        agent.next_pose = self.coordinator.generate_new_goal(pose=agent.pose)
                        if agent.step():
                            time.sleep(1)
                            read = agent.read()
                            self.coordinator.add_data(read)

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
            # plt.title("MSE is {}".format(mse))
            # plt.draw()
            # plt.pause(0.0001)
            i += 1

            # if self.agents[0].distance_travelled > 1500:
            #     print(mse)
            #     import numpy as np
            #     # with open('E:/ETSI/Proyecto/data/Databases/numpy_files/best_lm.npy', 'wb') as g:
            #     #     np.save(g, self.coordinator.surrogate().reshape((1000, 1500)).T)
            #     plt.show(block=True)

            if self.saving:
                self.f.write("{},{},{}\n".format(i, mse, self.agents[0].distance_travelled))
        print("done")
        if self.saving:
            self.f.close()
