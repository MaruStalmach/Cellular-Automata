from libs.Cells import State, Cell
from libs.Geometry import Geometry
import numpy as np


#TODO implement
class Rule():
    
    def __init__(self, geometry: Geometry):
        self.state_geometry = geometry
        self.required_keys = []
    
    def apply(self,state: State):
        pass
    
    def neighbors_matrix(self):
        return self.state_geometry.generate_neighbourhood_matrix()
    
    
    
    
    
    
    
#TODO implement
class TestRule(Rule):
    
    def __init__(self):
        super().__init__()
        
    def apply(self, state: State):
        pass
    
#TODO make more rules

class GameOfLife3D(Rule):
    """
    test rule that implements game of life in 3D"""
    
    def __init__(self, geometry):
        super().__init__(geometry)
        self.required_keys.extend(['alive'])

    def apply(self, neighbors: np.ndarray[Cell], cell: Cell):
        alive_neighbors = [n_cell['alive'] for n_cell in neighbors] #zero cell handled in Cell.__setitem__
        alive_sum = np.sum(alive_neighbors)

        if cell['alive']==1:
            if alive_sum<5 or alive_sum>7:
                cell['alive'] = 0
        else:
            if alive_sum == 5:
                cell['alive'] = 1

        return cell