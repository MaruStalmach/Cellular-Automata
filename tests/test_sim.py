from libs.Sim import CellularAutomaton
from libs.Geometry import Geometry
from libs.Rules import GameOfLife3D
from libs.Cells import Cell
import numpy as np


geometry = Geometry((2,2,2), axes="xy", periodicity="")
gol = GameOfLife3D(geometry=geometry)



def test_cellular_automaton_initialisation():
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    assert ca.rules == [gol]

def test_step_incrementation():
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    assert ca.step_no == 0

    ca.step()

    assert ca.step_no == 1
    
def test_select_neighbors():
    size = (3,3,3)
    g = Geometry(size=size,axes='xyz',periodicity='')
    ca = CellularAutomaton(geometry=g, rules=[])
    
    def gol_state_from_array(array : np.ndarray):
        assert(array.shape==size)
        state = np.empty(size, dtype=object)
        state = state.flatten()
        for i,val in enumerate(array.flatten()):
            state[i] = Cell.from_dict({'val':val})
        state = state.reshape(size)
        return state
    
    init_state = np.zeros(size)
    
    init_state = np.arange(np.prod(size)).reshape(size)
            
    
    
    
    init_state = gol_state_from_array(init_state)
    ca.state.data = init_state
    
    ## General Test: Does the middle cell have 26 neighbors? 
    neighbors = ca._select_neighbors(13,ca.state.data.flatten())
    assert len(neighbors)==26
    
    
    ## Test 2: Edge cell with no periodicity
    neighbors = ca._select_neighbors(0,ca.state.data.flatten())
    neighbors = sorted([cell['val'] for cell in neighbors])
    
    assert neighbors==[1,3,4,9,10,12,13]
    
    ## Test 3: Periodicity on x
    g = Geometry(size=size,axes='xyz',periodicity='x')
    ca = CellularAutomaton(geometry=g, rules=[])
    ca.state.data = init_state
    
    neighbors = ca._select_neighbors(0,ca.state.data.flatten())
    neighbors = sorted([cell['val'] for cell in neighbors])
    
    assert neighbors==[1,2,3,4,5,9,10,11,12,13,14]
    
    ## Test 4: Peridicity on y
    g = Geometry(size=size,axes='xyz',periodicity='y')
    ca = CellularAutomaton(geometry=g, rules=[])
    ca.state.data = init_state
    
    neighbors = ca._select_neighbors(0,ca.state.data.flatten())
    neighbors = sorted([cell['val'] for cell in neighbors])
    
    assert neighbors==[1,3,4,6,7,9,10,12,13,15,16]
    
    ## Test 5: Peridicity on z
    g = Geometry(size=size,axes='xyz',periodicity='z')
    ca = CellularAutomaton(geometry=g, rules=[])
    ca.state.data = init_state
    
    neighbors = ca._select_neighbors(0,ca.state.data.flatten())
    neighbors = sorted([cell['val'] for cell in neighbors])
    
    assert neighbors==[1,3,4,9,10,12,13,18,19,21,22]
    
    ## Test 5: Peridicity on xyz, all cells should be neighbors
    g = Geometry(size=size,axes='xyz',periodicity='xyz')
    ca = CellularAutomaton(geometry=g, rules=[])
    ca.state.data = init_state
    
    neighbors = ca._select_neighbors(0,ca.state.data.flatten())
    neighbors = sorted([cell['val'] for cell in neighbors])
    
    assert neighbors==[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    
def test_simulation():
    size=(1,5,5)
    g=Geometry(size=size,axes='xyz',periodicity='')
    ca=CellularAutomaton(g,[GameOfLife3D(g,2,3,3,3)])
    
    def gol_state_from_array(array : np.ndarray):
        assert(array.shape==size)
        state = np.empty(size, dtype=object)
        d_a = {'alive':1}
        d_d = {'alive':0}
        state = state.flatten()
        for i,val in enumerate(array.flatten()):
            if val == 1:
                state[i] = Cell.from_dict(d_a)
            else:
                state[i] = Cell.from_dict(d_d)
        state = state.reshape(size)
        return state
    
    def array_from_gol_state(state_data):
        arr = np.zeros_like(state_data)
        arr = arr.flatten()
        for i,cell in enumerate(state_data.flatten()):
            arr[i] = cell['alive']
        arr = arr.reshape(size)
        return arr
    
    
    # blinker test
    even_state = np.array([[
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0]]]
    )
    odd_state = np.array(
        [[[0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0]]]
    )
    
    
    
    ca.state.data = gol_state_from_array(even_state)
    
    # should see periodic behavior
    ca.step()
    ca_state_arr = array_from_gol_state(ca.state.data)
    assert np.array_equal(ca_state_arr,odd_state)
    
    ca.step()
    ca_state_arr = array_from_gol_state(ca.state.data)
    assert np.array_equal(ca_state_arr,even_state)
    
    
    #test 2
    states = [
        np.array([[
        [1,0,0,1,0],
        [0,1,1,0,1],
        [0,0,1,0,0],
        [0,1,1,0,1],
        [0,0,0,1,0]]]),
        np.array([[
        [0,1,1,1,0],
        [0,1,1,0,0],
        [0,0,0,0,0],
        [0,1,1,0,0],
        [0,0,1,1,0]]]),
        np.array([[
        [0,1,0,1,0],
        [0,1,0,1,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,1,1,0]]]),
        np.array([[
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,0,1,0],
        [0,1,0,1,0],
        [0,1,0,1,0]]]),
        np.array([[
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [1,1,0,1,1],
        [0,0,0,0,0]]]),
        np.zeros(shape=(1,5,5))
    ]
    
    ca.state.data = gol_state_from_array(states[0])
    
    for i in range(1,len(states)):
        ca.step()
        ca_state_arr = array_from_gol_state(ca.state.data)
        assert np.array_equal(ca_state_arr,states[i])
    
                
if __name__=="__main__":
    print('test')
    test_cellular_automaton_initialisation()
    test_step_incrementation()
    test_select_neighbors()
    test_simulation()