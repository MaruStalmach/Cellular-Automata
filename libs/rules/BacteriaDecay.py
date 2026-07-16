from libs.Rule import Rule
from libs.State import State
from libs.Geometry import Geometry
import numpy as np

class BacteriaDecay(Rule):
    def __init__(self, geometry:Geometry, time_step : float, target_keys, decay_probability):
        super().__init__(geometry, time_step)
        self.target_keys = target_keys
        self.decay_probability = decay_probability
        self.buff = np.zeros(geometry.size, dtype=np.float32)

    def apply_state(self, state:State):

        for key in self.target_keys:
            grid = state[key].copy()

            self.buff = np.random.random(size=grid.shape)

            self.buff = grid.astype(np.bool) & (self.buff<self.decay_probability)

            grid[self.buff==1] = 0

            state[key]=grid
        return state


