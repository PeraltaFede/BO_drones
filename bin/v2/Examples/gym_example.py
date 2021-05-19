import time
from sys import path

import numpy as np

path.extend([path[0][:path[0].rindex("bin") - 1]])
from bin.Agents.gym_agent_trimmed import SimpleAgent
from bin.Simulators.gym_environment import GymEnvironment

seeds = np.linspace(163343, 3647565, 100)
ss = [
    ["s1", "s2"],
    ["s5", "s6"],
    ["s1", "s2", "s3"],
    ["s5", "s6", "s7"],
    ["s1", "s2", "s3", "s4"],
    ["s5", "s6", "s7", "s8"],
    # ["s1", "s2", "s3", "s4", "s5"],
    # ["s5", "s6", "s7", "s8", "s1"],
]
for sensores in ss:
    drones_available = [[SimpleAgent(sensores, _id=0), SimpleAgent(sensores, _id=1)],
                        [SimpleAgent(sensores, _id=0), SimpleAgent(sensores, _id=1), SimpleAgent(sensores, _id=2)],
                        [SimpleAgent(sensores, _id=0), SimpleAgent(sensores, _id=1), SimpleAgent(sensores, _id=2),
                         SimpleAgent(sensores, _id=3)]]
    nro_drones = [2, 3, 4]
    # fs = ["pareto"]
    for cant_drones in nro_drones:
        print(cant_drones, sensores, "gaussian_ei", 0.375)
        i = 0
        for seed in seeds:
            i += 1
            np.random.seed(np.round(seed).astype(int))
            drones = [SimpleAgent(sensores, _id=k) for k in range(cant_drones)]
            sim = GymEnvironment(path[-1] + "/data/Map/Ypacarai/map.yaml", agents=drones, id_file=0,
                                 acq="gaussian_ei", acq_mod="truncated", render2gui=False, saving=True,
                                 name_file="{}_{}_{}_{}A{}S".format("gaussian_ei", "coupled", 0.375, cant_drones,
                                                                    len(sensores)),
                                 acq_fusion="coupled", d=0.375)
            try:
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
                        if agent.reached_pose():
                            next_poses.append(sim.coordinator.generate_new_goal(pose=agent.pose,
                                                                                other_poses=[agt.pose for agt in
                                                                                             sim.agents
                                                                                             if
                                                                                             agt is not agent]))
                            if sim.render2gui:
                                sim.sender.send_new_goal_msg(agent.next_pose, agent.drone_id)
                        else:
                            next_poses.append([])
                    print(k, np.mean(sim.step(next_poses)))
            except Exception as e:
                # raise e
                sim.f.write(f"pos: {sim.agents[0].pose} " + str(e))
            finally:
                if sim.saving:
                    sim.f.close()
            print(i)
            # break
            if i >= 50:
                break
