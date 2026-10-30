from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.rules.GutDrift import GutDrift
from libs.Geometry import Geometry
from libs.Sim import CellularAutomaton
from libs.rendering.Renderer import CARenderer


if __name__ == "__main__":
    size = (10, 10, 10)
    axes = "xyz"
    p = "z"

    species_keys = ["species_a", "biofilm", "floating_bacteria"]

    geometry = Geometry(size=size, axes=axes, periodicity=p)

    drift_rule = GutDrift(
        geometry=geometry,
        time_step=1,
        species_keys=species_keys,
        drift_speed=1,
        target_key="floating_bacteria",
    )
    ca = CellularAutomaton(
        geometry=geometry,
        rules=[drift_rule],
    )

    ca.state["biofilm"][1:5, 1:5, 0:2] = 1
    ca.state["species_a"][1:5, 1:5, 0:2] = 1

    ca.state["floating_bacteria"][1:5, 1:5, 3:5] = 1
    ca.state["species_a"][1:5, 1:5, 3:5] = 1

    def update_callback(stp=True):
        if stp:
            ca.step()

        return ca.state.key_grid("floating_bacteria", as_bool=True)

    renderer = CARenderer(
        init_state=update_callback(stp=False),
        width=1200,
        height=800,
        update_callback=update_callback,
    )

    try:
        renderer.run()
    except KeyboardInterrupt:
        print("Simulation stopped by user")
