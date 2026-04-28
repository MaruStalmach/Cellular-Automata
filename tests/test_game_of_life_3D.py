from libs.Rules import Rule, GameOfLife3D
from libs.Geometry import Geometry
from libs.Cells import Cell

def make_cell(alive: int) -> Cell:
    '''helper function for creating cells '''
    cell = Cell(keys=["alive"], random=False)
    cell['alive'] = alive
    return cell

# BASIC TESTS

def test_GoL_birth():
    geometry = Geometry((2,2,2), axes="xy", periodicity="y")
    gol = GameOfLife3D(geometry=geometry)

    cell = make_cell(0)
    neighbours = [make_cell(1) for _ in range(6)]
    neighbours += [make_cell(0) for _ in range(20)]

    result =  gol.apply(neighbors=neighbours, cell=cell)

    assert result['alive'] == 1
    assert type(cell) == Cell

def test_GoL_survival():
    geometry = Geometry((2,2,2), axes="xy", periodicity="y")
    gol = GameOfLife3D(geometry=geometry)

    cell = make_cell(1)
    neighbours = [make_cell(1) for _ in range(5)]
    neighbours += [make_cell(0) for _ in range(21)]

    result =  gol.apply(neighbors=neighbours, cell=cell)
    
    assert result['alive'] == 1
    assert type(cell) == Cell
    

def test_GoL_death():
    geometry = Geometry((2,2,2), axes="xy", periodicity="y")
    gol = GameOfLife3D(geometry=geometry)

    cell = make_cell(1)
    neighbours = [make_cell(1) for _ in range(2)]
    neighbours += [make_cell(0) for _ in range(25)]

    result =  gol.apply(neighbors=neighbours, cell=cell)
    
    assert result['alive'] == 0
    assert type(cell) == Cell

    