import numpy as np

from bin.Agents.simulated_pathplanning_agent import SimpleAgent
from bin.Simulators.multi_drone_simulator import Simulator

seeds = np.linspace(163343, 3647565, 100)
for file in [53, 54, 55, 59, 60, 10, 11, 12, 13, 14]:  #
    # file = 0
    i = 0
    for seed in seeds:
        i += 1
        np.random.seed(np.round(seed).astype(int))
        drones = [SimpleAgent(["t"], _id=0), SimpleAgent(["t"], _id=1), SimpleAgent(["t"], _id=2),
                  SimpleAgent(["t"], _id=3)]
        sim = Simulator("E:/ETSI/Proyecto/data/Map/Ypacarai/map.yaml", drones, "t", test_name="testing_test4ms",
                        saving=False, file=file,
                        acq="max_std", acq_mod="truncated")
        sim.run_simulation()
        if i >= 50:
            break
