from libs.Rules import Rule
import numpy as np
from libs.Geometry import Geometry
from libs.Cells import State, ZERO_CELL, Cell






#TODO implement
class CellularAutomaton():
    """
CA Simulation engine
    """
    
    def __init__(self,geometry: Geometry, rules: list[Rule]):
        """Heart of the CA simulation, allows advancing the simulation by applying rules to the current state. Tracks number of steps since start.

        Args:
            rules (list[Rule]): list of rules to apply to states
        """
        self.rules = rules
        self.geometry = geometry
        self.mask = np.array([
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ],
            [
                [0,1,0],
                [1,1,1],
                [0,1,0]
            ],
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ],
        ])

        # get keys reuired by rules used in the simulation
        keys = []
        for rule in rules:
            keys.extend(rule.required_keys)

        
        self.state = State(geometry,True,keys)
        # self._state_padded = np.pad(self.state._data,((1,1),(1,1),(1,1)),constant_values=ZERO_CELL)
        self.step_no = 0
        
    def step(self):
        """
    advance the simulation one step.
        """
        self._apply(self.state)
        self.step_no+=1
        
        
    def _apply(self,state:State):

        # initialize new state as Cell objects
        new_state = np.empty(state.shape, dtype=object)


        # get neighbors from geometry
        # neighbors_matrix = state.geometry.generate_neighbourhood_matrix().tocsr()
        flattened = state.data.flatten()


        # iterate over all the cells in the state matrix
        #TODO: split among threads
        for cell in range(len(flattened)): 
            neighbors = self._select_neighbors(self._dim1_to_dim3_coords(cell,self.state.shape))

            original_cell = flattened[cell]
            this_cell = Cell(keys=original_cell.keys,random=False)
            for key in original_cell.keys:
                this_cell[key] = original_cell[key]

            
            for rule in self.rules:
                this_cell = rule.apply(neighbors, this_cell)
            x,y,z = self._dim1_to_dim3_coords(cell,state.shape)
            new_state[x, y, z] = this_cell
        
        # thru setter implementation this will set the new values correctly and update the padding
        state.data = new_state
    
    def _dim1_to_dim3_coords(self,dim1_coord,state_shape):
        """converts 1D coordinate to 3D coordinate, given the shape of the state matrix"""
        return np.unravel_index(dim1_coord, state_shape)
        
    def _select_neighbors(self,location):
    
        lx,ly,lz = location
        
        selection = self.state._data[lx:lx+3,ly:ly+3,lz:lz+3] * self.mask

        return selection.flatten().tolist()