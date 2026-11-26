import numpy as np

from cellular_automata.registry import register_rule
from cellular_automata.Rule import Rule, StepContext
from cellular_automata.State import State


@register_rule
class GameOfLife3D(Rule):
    """vectorized 3DGoL based on arrays

    args:
    eb
    eh
    fb
    fh

    """

    def __init__(
        self,
        geometry,
        time_step: float,
        eb=5,
        eh=7,
        fb=6,
        fh=6,
        target_key: str = "alive",
    ):
        super().__init__(geometry, time_step)
        self.target_key = target_key
        self.required_keys.append(target_key)
        self.eb = eb
        self.eh = eh
        self.fb = fb
        self.fh = fh

    def apply_state(self, state: State, ctx: StepContext | None = None) -> State:
        # Use per-key accessors so reads/writes affect the underlying arrays
        alive_counts = self.neighborhood_count(state[self.target_key])

        current_alive = state[self.target_key] == 1
        survives = current_alive & (alive_counts >= self.eb) & (alive_counts <= self.eh)
        born = (~current_alive) & (alive_counts >= self.fb) & (alive_counts <= self.fh)

        updated_alive = np.zeros_like(state[self.target_key])
        updated_alive[survives | born] = 1
        state[self.target_key] = updated_alive
        return state
