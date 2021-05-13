from sys import path

import numpy as np

path.extend([path[0][:path[0].rindex("bin") - 1]])
from bin.Agents.gym_agent import SimpleAgent
from bin.Simulators.gym_environment import GymEnvironment

seeds = np.linspace(978462, 87549927, 100)
# optimal lengthscale is probably 153
ds = [0.38466, 0.40466, 0.42466, 0.44466, 0.48466]
for d in ds:
    i = 0
    for seed in seeds:
        i += 1
        np.random.seed(np.round(seed).astype(int))
        drones = [SimpleAgent("t", _id=0, limited_distance=False)]
        sim = GymEnvironment(path[-1] + "/data/Map/Simple/map.yaml", agents=drones, id_file=99,
                             acq="gaussian_ei", acq_mod="truncated", render2gui=False, saving=True,
                             name_file="gausian_ei_coupled_{}_1A1S".format(d), d=d, initial_pos="random")
        try:
            for k in range(25):
                next_poses = []
                for agent in sim.agents:
                    if agent.reached_pose():
                        next_poses.append(sim.coordinator.generate_new_goal(pose=agent.pose,
                                                                            other_poses=[agt.pose for agt in
                                                                                         sim.agents
                                                                                         if
                                                                                         agt is not agent]))
                    else:
                        next_poses.append([])
                print(k, sim.step(next_poses))
        except Exception as e:
            sim.f.write(f"pos: {sim.agents[0].pose} " + str(e))
        finally:
            if sim.saving:
                sim.f.close()
        print(i)