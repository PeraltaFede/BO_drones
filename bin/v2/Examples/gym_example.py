import time
from sys import path

import numpy as np

path.extend([path[0][:path[0].rindex("bin") - 1]])
from bin.Agents.gym_agent_trimmed import SimpleAgent
from bin.Simulators.gym_environment import GymEnvironment

seeds = np.linspace(163343, 3647565, 100)
for acq in ["predictive_entropy_search"]:
    # ds = [0.25, 0.375, 0.5, 0.75, 1.0] if acq == "gaussian_ei" else [0.25]
    ds = [0.375]
    for d in ds:
        ss = [
            ["s5", "s6"],
        ]
        for sensores in ss:
            fs = ["coupled"]
            for fusion in fs:
                print(fusion, sensores, acq, d)
                i = 0
                for seed in seeds:
                    i += 1
                    np.random.seed(np.round(seed).astype(int))
                    drones = [SimpleAgent(sensores, _id=0)]
                    if i <= 4:
                        continue
                    sim = GymEnvironment(path[-1] + "/data/Map/Ypacarai/map.yaml", agents=drones, id_file=0,
                                         acq=acq, acq_mod="truncated", render2gui=False, saving=False,
                                         name_file="{}_{}_{}_1A{}S".format(acq, fusion, d, len(sensores)),
                                         acq_fusion=fusion, d=d)

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
                                    # TODO: creo que debe ser [] instead
                                    next_poses.append(sim.coordinator.generate_new_goal(pose=agent.pose,
                                                                                        other_poses=[agt.pose for agt in
                                                                                                     sim.agents
                                                                                                     if
                                                                                                     agt is not agent]))
                                    if sim.render2gui:
                                        sim.sender.send_new_goal_msg(agent.next_pose, agent.drone_id)
                                else:
                                    next_poses.append([])
                            print(k, sim.step(next_poses))
                    except Exception as e:
                        sim.f.write(f"pos: {sim.agents[0].pose} " + str(e))
                    finally:
                        if sim.saving:
                            sim.f.close()
                    print(i)
                    if i >= 50:
                        break
