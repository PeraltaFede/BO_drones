# v2
from bin.Agents.simple_agent import SimpleAgent
from bin.Simulators.multi_simulator import Simulator

drones = [SimpleAgent(["t"])]
sim = Simulator("E:/ETSI/Proyecto/data/Map/Ypacarai/map.yaml", drones, "t")
sim.run_simulation()
