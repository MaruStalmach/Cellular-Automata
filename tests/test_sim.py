from libs.Sim import CellularAutomaton
from libs.Geometry import Geometry
from libs.Rules import GameOfLife3D


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
