from libs.Cells import State, Cell
from libs.Geometry import Geometry
import numpy as np


#TODO implement
class Rule():
    
    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.required_keys = []
    
    def apply(self, neighbors: list[Cell], cell: Cell) -> Cell:
        return cell
    
    def neighbors_matrix(self):
        return self.geometry.generate_neighbourhood_matrix()
    
    
    
    
    
    
    
#TODO implement
class TestRule(Rule):
    
    def __init__(self, geometry: Geometry):
        super().__init__(geometry)
        
    def apply(self, neighbors: list[Cell], cell: Cell) -> Cell:
        return cell
    
#TODO make more rules

class GameOfLife3D(Rule):
    """
    test rule that implements game of life in 3D"""
    
    def __init__(self, geometry, eb=5, eh=7, fb=6, fh=6):
        super().__init__(geometry)
        self.required_keys.extend(['alive'])
        self.eb = eb
        self.eh = eh
        self.fb = fb
        self.fh = fh

    def apply(self, neighbors: list[Cell], cell: Cell) -> Cell:
        alive_neighbors = [n_cell['alive'] for n_cell in neighbors] #zero cell handled in Cell.__setitem__
        alive_sum = np.sum(alive_neighbors)

        if cell['alive']==1:
            if alive_sum<self.eb or alive_sum>self.eh:
                cell['alive'] = 0
        else:
            if alive_sum>=self.fb and alive_sum<=self.fh:
                cell['alive'] = 1

        return cell
    
    

class GutDrift(Rule):
    '''defines the drift along z-axis in 3D simulations of the human gut'''
    def __init__(self, geometry, drift_speed: float, detachment_interval: int, base_detachment_rate: float = 0.05, scaling: float = 0.05):
        super().__init__(geometry=geometry)

        assert self.geometry.ndim == 3 # only pushes along z-axis if there are 3 axis present
        assert 'z' in self.geometry.periodicity # only pushes along z-axis if z axis is present
        
        # self.required_keys.extend(['population', 'biofilm_population', 'coord_z', 'distance_to_wall']) #TODO: determine if we need population AND biofilm population

        self.drift_speed = drift_speed
        self.detachment_interval = detachment_interval

        self.base_detachment_rate = base_detachment_rate
        self.scaling = scaling
        self.curr_step = 0

    def apply(self, neighbours: list[Cell], cell: Cell) -> Cell:
        pass
        