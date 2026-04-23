from libs.Cells import Cell, State
from libs.Rules import Rule, GameOfLife3D
from libs.Geometry import Geometry
from libs.Sim import CellularAutomaton

def test_state_initialisation():
    geometry2d = Geometry((2,3), axes='xy', periodicity='')
    geometry3d = Geometry((3,4,1), axes='xyz', periodicity='y')
    geometry4d = Geometry((2,1,5,2), axes='xyza', periodicity='a')

    cell_keys = ['alive']

    state2d = State(geometry2d, random=False, cell_keys=cell_keys)
    state3d = State(geometry3d, random=False, cell_keys=cell_keys)
    state4d = State(geometry4d, random=True, cell_keys=cell_keys)
    
    #check unpadded data
    assert state2d.data.shape == (2,3)
    assert state3d.data.shape == (3,4,1)
    assert state4d.data.shape == (2,1,5,2)

    #check cell content
    assert state2d.data[0,0].keys == cell_keys

    #check cell state
    assert state3d.data[0,0,0]['alive'] == 0

    #check if padding is added
    assert state2d._data.shape == (4,5)
    assert state3d._data.shape == (5,6,3)
    assert state4d._data.shape == (4,3,7,4)