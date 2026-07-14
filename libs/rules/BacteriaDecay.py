from libs.Rule import Rule
from libs.State import State
from libs.Geometry import Geometry
import numpy as np

class BacteriaDecay(Rule):
    def __init__(self, geometry:Geometry, target_key, decay_probability):
        super().__init__(geometry)
        self.target_key = target_key
        self.decay_probability = decay_probability
        self.buff = np.zeros(geometry.size, dtype=np.float32)

    def apply_state(self, state:State):

        grid = state[self.target_key].copy()

        self.buff = np.random.random(size=grid.shape)

        self.buff = grid.astype(np.bool) & (self.buff<self.decay_probability)

        grid[self.buff==1] = 0

        state[self.target_key]=grid
        return state


