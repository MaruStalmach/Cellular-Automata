from libs2.Geometry import Geometry
from libs2.Rules import GameOfLife3D
from libs2.Sim import CellularAutomaton

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