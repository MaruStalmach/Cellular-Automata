from libs.Rule import Rule
from libs.State import State
from libs.Geometry import Geometry
import numpy as np

class Inoculation(Rule):
    def __init__(self, geometry:Geometry, time_step : float, target_key, location, size):
        super().__init__(geometry, time_step)
        self.target_key = target_key
        self.location = location
        self.size = size

    def apply_state(self, state:State):

        grid = state[self.target_key].copy()

        sel = [slice(self.location[0], self.location[0]+self.size[0]), slice(self.location[1], self.location[1]+self.size[1]), slice(self.location[2], self.location[2]+self.size[2])]
        if len(grid.shape)==4:
            sel += [0]
        grid[tuple(sel)] = 1
            
        state[self.target_key] = grid
        return state


