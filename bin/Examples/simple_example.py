from bin.Agents.simple_agent import SimpleAgent
from bin.Simulators.simple_simulator import Simulator


sim = Simulator("E:/ETSI/Proyecto/data/Map/Simple/map.yaml", SimpleAgent(["s4"]), "s4", render=True)

# sim.observe_maps()
# sim.export_maps(render=True)
sim.run_simulation()
