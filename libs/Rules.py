from libs.Cells import State, Cell
from libs.Geometry import Geometry
import numpy as np


#TODO implement
class Rule():
    
    def __init__(self, geometry: Geometry):
        self.state_geometry = geometry
        self.required_keys = []
    
    def apply(self, neighbors: list[Cell], cell: Cell) -> Cell:
        return cell
    
    def neighbors_matrix(self):
        return self.state_geometry.generate_neighbourhood_matrix()
    
    
    
    
    
    
    
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
    
    def __init__(self, geometry):
        super().__init__(geometry)
        self.required_keys.extend(['alive'])
        self.eb = 5
        self.eh = 7
        self.fb = 6
        self.fh = 6

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
    
    
    