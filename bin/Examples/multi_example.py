from bin.Agents.simple_agent import SimpleAgent
from bin.Simulators.multi_simulator import Simulator

drones = [SimpleAgent(["s0"]), SimpleAgent(["s0"])]

sim = Simulator("E:/ETSI/Proyecto/data/Map/Ypacarai/map.yaml", drones, "s0",
                render=False)

# sim.observe_maps()
# sim.export_maps(render=True)
sim.run_simulation()
