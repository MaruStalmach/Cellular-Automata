"""Floating bacteria drifting along z while the biofilm stays put, rendered live (press P to step)."""

from cellular_automata.Geometry import Geometry
from cellular_automata.rules import GutDrift
from cellular_automata.Sim import CellularAutomaton

SPECIES_KEYS = ["species_a", "biofilm", "floating_bacteria"]


def build_sim(size: tuple[int, int, int], max_steps: int = 100) -> CellularAutomaton:
    geometry = Geometry(size=size, axes="xyz", periodicity="z")
    drift_rule = GutDrift(
        geometry=geometry,
        time_step=1,
        species_keys=SPECIES_KEYS,
        drift_speed=1,
        target_key="floating_bacteria",
    )
    ca = CellularAutomaton(geometry=geometry, rules=[drift_rule], max_steps=max_steps)

    ca.state["biofilm"][1:5, 1:5, 0:2] = 1
    ca.state["species_a"][1:5, 1:5, 0:2] = 1
    ca.state["floating_bacteria"][1:5, 1:5, 3:5] = 1
    ca.state["species_a"][1:5, 1:5, 3:5] = 1
    return ca


def main():
    from cellular_automata.rendering.Renderer import CARenderer

    ca = build_sim((10, 10, 10))

    def update_callback(step=True):
        if step:
            ca.step()
        return ca.state.key_grid("floating_bacteria", as_bool=True)

    renderer = CARenderer(
        init_state=update_callback(step=False),
        width=1200,
        height=800,
        update_callback=update_callback,
    )

    try:
        renderer.run()
    except KeyboardInterrupt:
        print("Simulation stopped by user")


if __name__ == "__main__":
    main()
