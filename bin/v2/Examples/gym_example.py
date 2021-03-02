import time
from sys import path

import numpy as np

path.extend([path[0][:path[0].rindex("bin") - 1]])
from bin.Agents.gym_agent import SimpleAgent
from bin.Simulators.gym_environment import GymEnvironment

# todo hacer simulación
seeds = np.linspace(163343, 3647565, 100)
for d in [0.25]:
    for acq in ["predictive_entropy_search"]:
        for sensores in [["s1", "s2"],
                         ["s5", "s6"],
                         ["s1", "s2", "s3"],
                         ["s5", "s6", "s7"],
                         ["s1", "s2", "s3", "s4"],
                         ["s5", "s6", "s7", "s8"],
                         ["s1", "s2", "s3", "s4", "s5"],
                         ["s5", "s6", "s7", "s8", "s1"]]:
            for fusion in ["coupled", "decoupled"]:
                print(fusion, sensores, acq, d)
                i = 0
                for seed in seeds:
                    i += 1
                    np.random.seed(np.round(seed).astype(int))
                    drones = [SimpleAgent(sensores, _id=0)]
                    sim = GymEnvironment(path[-1] + "/data/Map/Ypacarai/map.yaml", agents=drones, id_file=0,
                                         acq=acq, acq_mod="truncated", render2gui=False, saving=True,
                                         name_file="{}_{}_{}_1A{}S".format(acq, fusion, d, len(sensores)),
                                         acq_fusion=fusion, d=d)
                    for k in range(50):
                        while True:
                            if sim.render2gui and not sim.sender.should_update():
                                time.sleep(1)
                                continue
                            else:
                                break
                        # Selection of best next measurement position occurs here
                        next_poses = []
                        for agent in sim.agents:
                            # print('current pose is', agent.pose)
                            if agent.reached_pose():
                                # TODO: creo que debe ser [] instead
                                next_poses.append(sim.coordinator.generate_new_goal(pose=agent.pose,
                                                                                    other_poses=[agt.pose for agt in
                                                                                                 sim.agents
                                                                                                 if
                                                                                                 agt is not agent]))
                                # print('current goal is', next_poses[-1])
                                if sim.render2gui:
                                    sim.sender.send_new_goal_msg(agent.next_pose, agent.drone_id)
                            else:
                                next_poses.append([])
                        print(sim.step(next_poses))
                        # mus, stds = sim.render()
                    if sim.saving:
                        sim.f.close()
                    print(i)
                    if i >= 50:
                        break
