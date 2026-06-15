from libs2.Geometry import Geometry
from libs2.Rules import GameOfLife3D
from libs2.Sim import CellularAutomaton

import numpy as np

def test_cellular_automaton_initialisation():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    gol = GameOfLife3D(geometry=geometry)
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    assert ca.rules == [gol]
    assert ca.step_no == 0
    
    #checks for correctness of aggregated rules
    assert 'alive' in ca.state.keys

def test_step_incrementation():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    gol = GameOfLife3D(geometry=geometry)
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    assert ca.step_no == 0
    ca.step()
    assert ca.step_no == 1
    
def test_gol_correctness():
    size=(1,5,5)
    g=Geometry(size=size,axes='xyz',periodicity='')
    ca=CellularAutomaton(g,[GameOfLife3D(g,2,3,3,3)])
    
    def gol_state_from_array(array : np.ndarray):
        return array
        
    
    def array_from_gol_state(state_data):
        arr = state_data['alive']
        return arr
    
    
    # blinker test
    even_state = np.array([[
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0]]],dtype=np.uint8
    )
    odd_state = np.array(
        [[[0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0]]],dtype=np.uint8
    )
    
    
    
    ca.state['alive'] = gol_state_from_array(even_state)
    
    # should see periodic behavior
    ca.step()
    ca_state_arr = array_from_gol_state(ca.state)
    assert np.array_equal(ca_state_arr,odd_state)
    
    ca.step()
    ca_state_arr = array_from_gol_state(ca.state)
    assert np.array_equal(ca_state_arr,even_state)
    
    
    #test 2
    states = [
        np.array([[
        [1,0,0,1,0],
        [0,1,1,0,1],
        [0,0,1,0,0],
        [0,1,1,0,1],
        [0,0,0,1,0]]],dtype=np.uint8),
        np.array([[
        [0,1,1,1,0],
        [0,1,1,0,0],
        [0,0,0,0,0],
        [0,1,1,0,0],
        [0,0,1,1,0]]],dtype=np.uint8),
        np.array([[
        [0,1,0,1,0],
        [0,1,0,1,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,1,1,0]]],dtype=np.uint8),
        np.array([[
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,0,1,0],
        [0,1,0,1,0],
        [0,1,0,1,0]]],dtype=np.uint8),
        np.array([[
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [1,1,0,1,1],
        [0,0,0,0,0]]],dtype=np.uint8),
        np.zeros(shape=(1,5,5),dtype=np.uint8)
    ]
    
    ca.state['alive'] = gol_state_from_array(states[0])
    
    for i in range(1,len(states)):
        ca.step()
        ca_state_arr = array_from_gol_state(ca.state)
        assert np.array_equal(ca_state_arr,states[i])
        
        
if __name__ == "__main__":
    test_gol_correctness()