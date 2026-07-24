from libs.Rule import Rule
from libs.State import State
import numpy as np

class GameOfLife3D(Rule):
    """vectorized 3DGoL based on arrays

    args:
    eb
    eh
    fb
    fh

    """

    def __init__(self, geometry, time_step : float, eb=5, eh=7, fb=6, fh=6, target_key: str = "alive"):
        super().__init__(geometry, time_step)
        self.required_keys.extend([target_key])
        self.eb = eb
        self.eh = eh
        self.fb = fb
        self.fh = fh

    def apply_state(self, state: State) -> State:
        # Use per-key accessors so reads/writes affect the underlying arrays
        alive_grid = state[self.required_keys[0]].astype(np.int8, copy=False)

        alive_counts = np.zeros_like(alive_grid, dtype=np.int8)
        alive_counts += self.neighborhood_count(alive_grid)

        current_alive = state["alive"] == 1
        survives = current_alive & (alive_counts >= self.eb) & (alive_counts <= self.eh)
        born = (~current_alive) & (alive_counts >= self.fb) & (alive_counts <= self.fh)

        updated_alive = np.zeros_like(state["alive"])
        updated_alive[survives | born] = 1
        state["alive"] = updated_alive
        return state