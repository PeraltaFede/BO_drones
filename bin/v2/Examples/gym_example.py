import time

import numpy as np

from bin.Agents.gym_agent import SimpleAgent
from bin.Simulators.gym_environment import GymEnvironment

seeds = np.linspace(163343, 3647565, 100)
for sensores in [["s1", "s2"], ["s1", "s2", "s3"], ["s1", "s2", "s3, s4"]]:
    for fusion in ["max_sum", "simple_max"]:
        i = 0
        for seed in seeds:
            i += 1
            np.random.seed(np.round(seed).astype(int))
            drones = [SimpleAgent(sensores, _id=0)]
            sim = GymEnvironment("E:/ETSI/Proyecto/data/Map/Ypacarai/map.yaml", agents=drones, id_file=0,
                                 acq_mod="truncated", render2gui=False, saving=True,
                                 name_file="{}_1A{}S".format(fusion, len(sensores)),
                                 acq_fusion=fusion)
            for k in range(20):
                while True:
                    if sim.render2gui and not sim.sender.should_update():
                        time.sleep(1)
                        continue
                    else:
                        break
                # Selection of best next measurement position occurs here
                next_poses = []
                for agent in sim.agents:
                    print('current pose is', agent.pose)
                    if agent.reached_pose():
                        # TODO: creo que debe ser [] instead
                        next_poses.append(sim.coordinator.generate_new_goal(pose=agent.pose,
                                                                            other_poses=[agt.pose for agt in sim.agents
                                                                                         if
                                                                                         agt is not agent]))
                        print('current goal is', next_poses[-1])
                        if sim.render2gui:
                            sim.sender.send_new_goal_msg(agent.next_pose, agent.drone_id)
                    else:
                        next_poses.append([])

                print(sim.step(next_poses))
                # mus, stds = sim.render()
            sim.f.close()
            if i >= 50:
                break
