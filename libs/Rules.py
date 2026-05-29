from libs.Cells import State, Cell
from libs.Geometry import Geometry
import numpy as np
from random import random


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
    



class BiofilmDetachment(Rule):
    '''checks how close the cell is to the gut wall and decides whether it should be detached during cell drift
    the closer the cell is to the center of the gut, the more likely it is to detach
    '''
    def __init__(self, geometry, detachment_rate, scaling):
        super().__init__(geometry=geometry)
        self.detachment_probability = detachment_rate * scaling #the bacteria closer to wall is less likely to detach

        #TODO: required keys

    def would_detach(self, cell: Cell) -> bool:
        if cell.coords is not None:
            coord_z = cell.coords[-1]
 
        distance_sq = coord_z ** 2
        probability = min(self.detachment_probability * distance_sq, 1.0)
        
        return random() < probability


    def apply(self, neighbors: list[Cell], cell: Cell) -> Cell:

        if self.would_detach(cell):
            #cell changes state and detaches
            pass

        return cell



class GutDrift(Rule):
    '''defines the drift along z-axis in 3D simulations of the human gut'''
    def __init__(self, geometry, drift_speed: float, detachment_rule: BiofilmDetachment | None = None):
        super().__init__(geometry=geometry)

        #TODO: required keys

        assert self.geometry.ndim == 3 # only pushes along z-axis if there are 3 axis present
        
        self.drift_speed = drift_speed
        self.detachment_rule = detachment_rule
    
    def _is_valid_for_drift(self, cell:Cell) -> bool | None:
        if self.detachment_rule is not None:
            if self.detachment_rule.would_detach(cell):
                return True


    def apply(self, neighbors: list[Cell], cell: Cell) -> Cell:
        #change cell coords here
    
        return cell
        