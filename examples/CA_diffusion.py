"""Substrate diffusion from one corner of the grid, rendered live (press P to step)."""

import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.rules import Diffusion
from cellular_automata.Sim import CellularAutomaton
from cellular_automata.State import State

SUBSTRATE_LAYERS = 6  # one layer per movement direction
STEPS_PER_FRAME = 100


def build_sim(size: tuple[int, int, int], max_steps: int = 1000) -> CellularAutomaton:
    geometry = Geometry(size, "xyz", "")
    rules = [Diffusion(geometry, time_step=1, a=6, p=0.08, target_key="substrate")]
    state = State(
        geometry, cell_keys=["substrate"], key_layers=[SUBSTRATE_LAYERS], random=False
    )
    state["substrate"][0:3, 0:3, 0:3, 0] = 1
    return CellularAutomaton(
        geometry=geometry, rules=rules, state=state, max_steps=max_steps
    )


def main():
    from cellular_automata.rendering.Renderer import CARenderer

    ca_sim = build_sim((50, 50, 50))

    def update_callback(step=True):
        """Called each frame to get the latest CA state"""
        if step:
            for _ in range(STEPS_PER_FRAME):
                if ca_sim.step() != 0:
                    break
            print(f"Step {ca_sim.step_no}/{ca_sim.max_steps}")
        return np.maximum.reduce(ca_sim.state["substrate"], axis=-1)

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
