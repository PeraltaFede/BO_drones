# v2
# from bin.Agents.simple_agent import SimpleAgent
import numpy as np

from bin.Agents.simulated_pathplanning_agent import SimpleAgent
from bin.Simulators.multi_drone_simulator import Simulator

# 42  -> False 0.002 True 0.0003
# 263 -> False 0.0005
# 260 -> True 0.0006
# 76842153
# 1123581321
# np.round((1+np.sqrt(5))/2* 10000).astype(np.int)\  best 35 0.0957
# seeds = np.linspace(76842153, 1123581321, 100)
seeds = np.linspace(163343, 3647565, 100)
for file in [10, 11, 12, 13, 14]:  #35 ,53, 54, 55, 59, 60
    # file = 0
    i = 0
    for seed in seeds:
        i += 1
        np.random.seed(np.round(seeds[3]).astype(int))
        drones = [SimpleAgent(["t"], _id=0), SimpleAgent(["t"], _id=1)]
        sim = Simulator("E:/ETSI/Proyecto/data/Map/Ypacarai/map.yaml", drones, "t", test_name="testing_test2",
                        saving=False, file=12,
                        acq="gaussian_ei", acq_mod="truncated")
        sim.run_simulation()
        if i >= 50:
            break
