import time
from sys import path

import numpy as np

path.extend([path[0][:path[0].rindex("bin") - 1]])
from bin.Agents.gym_agent_trimmed import SimpleAgent
from bin.v2.Comparison.ga_gym_simulator import GAGymEnvironment

seeds = np.linspace(163343, 3647565, 100)
for d in [0.375]:
    for sensores in [
        ["s1", "s2"],
        ["s5", "s6"],
        ["s1", "s2", "s3"],
        ["s5", "s6", "s7"],
        ["s1", "s2", "s3", "s4"],
        ["s5", "s6", "s7", "s8"],
        ["s1", "s2", "s3", "s4", "s5"],
        ["s5", "s6", "s7", "s8", "s1"]
    ]:
        print(sensores, d)
        i = 0
        for seed in seeds:
            i += 1
            np.random.seed(np.round(seed).astype(int))
            drones = [SimpleAgent(sensores, _id=0)]
            sim = GAGymEnvironment(path[-1] + "/data/Map/Ypacarai/map.yaml", agents=drones, id_file=0,
                                   render2gui=False, saving=True,
                                   name_file="GA_{}_1A{}S".format(d, len(sensores)), d=d)
            try:
                for k in range(50):
                    while True:
                        if sim.render2gui and not sim.sender.should_update():
                            time.sleep(1)
                            continue
                        else:
                            break
                    next_poses = []
                    for agent in sim.agents:
                        # print('current pose is', agent.pose)
                        if agent.reached_pose():
                            next_poses.append(sim.coordinator.generate_new_goal(pose=agent.pose))
                            # print('current goal is', next_poses[-1])
                            if sim.render2gui:
                                sim.sender.send_new_goal_msg(agent.next_pose, agent.drone_id)
                        else:
                            next_poses.append([])
                    print(k, sim.step(next_poses))
                    # mus, stds = sim.render()
            except Exception as e:
                sim.f.write(f"pos: {sim.agents[0].pose} " + str(e))
            finally:
                if sim.saving:
                    sim.f.close()
            print(i)
            if i >= 50:
                break
