import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from libs.Geometry import Geometry
from libs.Rule import Rule
from libs.Sim import CellularAutomaton
from libs.rendering.Renderer import CARenderer
from libs.State import State

from libs.neighbourhoods.MooreNeighbourhood import MooreNeighbourhood
from libs.neighbourhoods.VonNeumannNeighbourhood import VonNeumannNeighbourhood


class DeterministicExpansion(Rule):
    def __init__(self, geometry: Geometry):
        super().__init__(geometry)
        self.required_keys = ["alive"]

    def apply_state(self, state) -> State:
        # Fetch the grid using the dictionary indexing defined in State.py
        grid = state["alive"]

        # Use the built-in shift/count mechanism from Rule.py
        counts = self.neighborhood_count(grid)

        # If the cell is already alive OR has >= 1 alive neighbour, it becomes alive
        new_grid = np.where((grid == 1) | (counts > 0), 1, 0)

        # Copy state and update the target key
        new_state = state.copy()
        new_state["alive"] = new_grid.astype(state.dtype)

        return new_state


def main():
    # Setup a 20x20x20 bounding box for both simulations
    size = (20, 20, 20)
    axes = "xyz"
    periodicity = ""  # No wrapping to clearly see the boundaries

    # 1. Initialize Moore CA
    geo_moore = Geometry(size, axes, periodicity, neighbourhood=MooreNeighbourhood())
    ca_moore = CellularAutomaton(
        geometry=geo_moore, rules=[DeterministicExpansion(geo_moore)]
    )

    # 2. Initialize Von Neumann CA
    geo_vn = Geometry(size, axes, periodicity, neighbourhood=VonNeumannNeighbourhood())
    ca_vn = CellularAutomaton(geometry=geo_vn, rules=[DeterministicExpansion(geo_vn)])

    # 3. Seed both simulations exactly in the center
    center = (10, 10, 10)
    ca_moore.state["alive"][center] = 1
    ca_vn.state["alive"][center] = 1

    # Configuration state for the callback loop
    sim_state = {
        "running": True,
        "step_count": 0,
        "max_steps": 10,  # Keep steps low; expansion fills the grid quickly
    }

    def update_callback(step=True):
        """Called each frame by CARenderer to get the latest CA state."""
        if sim_state["running"] and sim_state["step_count"] < sim_state["max_steps"]:
            if step:
                # Step both independent simulations
                ca_moore.step()
                ca_vn.step()
                sim_state["step_count"] += 1
            print(f"Step {sim_state['step_count']}/{sim_state['max_steps']}")

        # Extract the grids
        grid_moore = ca_moore.state["alive"]
        grid_vn = ca_vn.state["alive"]

        # Stitch them together along the X-axis (axis=0) to view side-by-side.
        # The resulting array sent to CARenderer will have a shape of (40, 20, 20).
        combined_grid = np.concatenate((grid_moore, grid_vn), axis=0)

        return combined_grid

    # Initialize renderer with the stitched initial state
    renderer = CARenderer(
        width=1200, height=800, init_state=update_callback(step=False)
    )

    # Run renderer with CA updates
    try:
        renderer.run(update_callback=update_callback)
    except KeyboardInterrupt:
        print("Simulation stopped by user")


if __name__ == "__main__":
    main()
