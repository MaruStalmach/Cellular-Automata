from Rules import Rule
import numpy as np
from Geometry import Geometry






#TODO implement
class CellularAutomaton():
    
    def __init__(self,rules: list[Rule], neighborhood_mask):
        """Heart of the simulation, allows applications of rules to a state and tracks step number

        Args:
            rules (list[Rule]): _description_
            neighborhood_mask (_type_): _description_
        """
        self.rules = rules
        self.neighborhood_mask = neighborhood_mask
        self.step_no = 0
        
    
        
        
    def apply(self,state):

        
        new_state = np.zeros_like(state)
        neighbors_matrix = np.zeros((state.num_cells,np.prod(self.neighborhood_mask.shape))) # use a function to fill the matrix later but now we dont know how we will do it so placeholder
        #TODO: fill neighbors_matrix with actual values with some function
        for cell_idx_1d, neighbors in neighbors_matrix:
            x,y,z = self._dim1_to_dim3_coords(cell_idx_1d,state.shape)
            #TODO: split coords among threads
            for rule in self.rules:
                new_state[x,y,z] += rule.apply(neighbors)
             
        self.step_no+=1
        return new_state
    
    def _dim1_to_dim3_coords(self,dim1_coord,state_shape):
        return (dim1_coord%state_shape[0],
                dim1_coord//state_shape[0] %state_shape[1],
                dim1_coord//(state_shape[0]*state_shape[1]))