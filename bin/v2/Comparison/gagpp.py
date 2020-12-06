# v2
# from bin.Agents.simple_agent import SimpleAgent
import numpy as np

from bin.Agents.pathplanning_agent import SimpleAgent
from bin.v2.Comparison.ga_simulator import Simulator

# 42  -> False 0.002 True 0.0003
# 263 -> False 0.0005
# 260 -> True 0.0006
# 76842153
# 1123581321
# np.round((1+np.sqrt(5))/2* 10000).astype(np.int) best 23 0.146003
seeds = np.linspace(76842153, 1123581321, 100)
ks = [53, 54, 55, 59, 60, 10, 11, 12, 13, 14]
for k in ks:
    i = 0
    for seed in seeds:
        i += 1
        if i < 89:
            continue
        np.random.seed(np.round(seed).astype(int))
        drones = [SimpleAgent(["t"])]
        sim = Simulator("E:/ETSI/Proyecto/data/Map/Ypacarai/map.yaml", drones, "t",
                        test_name="GAFINALTEST{}".format(ks.index(k) + 1), saving=True, file=k)
        sim.run_simulation()
        print('{} done'.format(i))
        if i >= 98:
            break
