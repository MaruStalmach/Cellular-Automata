
from cellular_automata.Geometry import Geometry
from cellular_automata.registry import register_rule
from cellular_automata.Rule import Rule, StepContext
from cellular_automata.State import State


@register_rule
class BacteriaDecay(Rule):
    def __init__(
        self, geometry: Geometry, time_step: float, target_keys, decay_probability=0.05
    ):
        super().__init__(geometry, time_step)
        self.target_keys = target_keys
        self.required_keys.extend(target_keys)
        self.decay_probability = decay_probability

    def apply_state(self, state: State, ctx: StepContext | None = None):

        for key in self.target_keys:
            grid = state[key].copy()
            decays = grid.astype(bool) & (self.rng.random(size=grid.shape) < self.decay_probability)
            grid[decays] = 0
            state[key] = grid
        return state
