"""Two mutually symbiotic species spreading from opposite edges, rendered live (press P to step)."""

import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.rules import SpeciesInteraction
from cellular_automata.Sim import CellularAutomaton

INTERACTIONS = {"bac1": {"bac2": 1.0}, "bac2": {"bac1": 1.0}}


def build_sim(size: tuple[int, int, int], max_steps: int = 1000) -> CellularAutomaton:
    geometry = Geometry(size, "xyz", "")
    rules = [SpeciesInteraction(geometry, time_step=1, interactions=INTERACTIONS)]
    ca_sim = CellularAutomaton(geometry=geometry, rules=rules, max_steps=max_steps)

    b1s = np.zeros(shape=size, dtype=np.uint8)
    b1s[0, 0, :] = 1
    b2s = np.zeros(shape=size, dtype=np.uint8)
    b2s[-1, -1, :] = 1
    ca_sim.state["bac1"] = b1s
    ca_sim.state["bac2"] = b2s
    return ca_sim


def main():
    from cellular_automata.rendering.Renderer import CARenderer

    ca_sim = build_sim((100, 100, 100))

    def update_callback(step=True):
        """Called each frame to get the latest CA state"""
        if step and ca_sim.step() == 0:
            print(f"Step {ca_sim.step_no}/{ca_sim.max_steps}")
        return ca_sim.state["bac1"]

    renderer = CARenderer(
        width=1200,
        height=800,
        init_state=update_callback(step=False),
        update_callback=update_callback,
    )

    try:
        renderer.run()
    except KeyboardInterrupt:
        print("Simulation stopped by user")


if __name__ == "__main__":
    main()
