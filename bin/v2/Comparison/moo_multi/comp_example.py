import time
from sys import path

import numpy as np

path.extend([path[0][:path[0].rindex("bin") - 1]])
from bin.Agents.gym_agent_trimmed import SimpleAgent
from bin.v2.Comparison.moo_multi.comp_environment import GymEnvironment as Env

seeds = np.linspace(163343, 3647565, 100)
ss = [
    ["s1", "s2"],
    ["s5", "s6"],
    ["s1", "s2", "s3"],
    ["s5", "s6", "s7"],
    ["s1", "s2", "s3", "s4"],
    ["s5", "s6", "s7", "s8"]
]
for sensores in ss:
    nro_drones = [4, 3, 2]
    for cant_drones in nro_drones:
        print(cant_drones, sensores, "gaussian_ei", 0.375)
        i = 0
        for seed in seeds:
            i += 1
            np.random.seed(np.round(seed).astype(int))
            drones = [SimpleAgent(sensores, _id=k) for k in range(cant_drones)]
            sim = Env(path[-1] + "/data/Map/Ypacarai/map.yaml", agents=drones, id_file=0,
                      acq="ga", acq_mod="truncated", render2gui=False, saving=True,
                      name_file="{}_{}_{}_{}A{}S".format("ga", "ga", 0.375, cant_drones,
                                                         len(sensores)),
                      acq_fusion="ga", d=0.375)
            try:
                for k in range(50):
                    while True:
                        if sim.render2gui and not sim.sender.should_update():
                            time.sleep(1)
                            continue
                        else:
                            break
                    next_poses = []
                    for agent in enumerate(sim.agents):
                        if agent[1].reached_pose():
                            next_poses.append(sim.coordinator.generate_new_goal(pose=agent[1].pose,
                                                                                agt_id=agent[0]))
                            if sim.render2gui:
                                sim.sender.send_new_goal_msg(agent[1].next_pose, agent[1].drone_id)
                        else:
                            next_poses.append([])
                    print(k, np.mean(sim.step(next_poses)))
            except Exception as e:
                sim.f.write(f"pos: {sim.agents[0].pose} " + str(e))
            finally:
                if sim.saving:
                    sim.f.close()
            print(i)
            if i >= 50:
                break
