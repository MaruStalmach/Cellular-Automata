from libs.Cells import Cell, State
from libs.Geometry import Geometry

def make_predefined_state(geometry:Geometry) -> State:
    '''helper to create state with tag per cell'''
    state = State(geometry, random=False, cell_keys=['tag'])
    
    unpadded_data = state.data
    assert unpadded_data.shape == geometry.size 

    for tag, coords in enumerate(np.ndindex(geometry.size)):
        cell = Cell(keys=['tag'], random=False)
        cell['tag'] = tag
        unpadded_data[coords] = cell
    
    state.data = unpadded_data
    return state

