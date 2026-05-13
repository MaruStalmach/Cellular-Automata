from libs.Rules import Rule, GameOfLife3D
from libs.Geometry import Geometry
from libs.Cells import Cell

def make_cell(alive: int) -> Cell:
    '''helper function for creating cells '''
    cell = Cell(keys=["alive"], random=False)
    cell['alive'] = alive
    return cell

def make_neighbour(n_alive: int, total: int = 26):
    '''helper function for returning a list with n_alive cells out of total'''
    return [make_cell(1)] * n_alive + [make_cell(0)] * (total - n_alive)


def make_gol_4555(geometry: Geometry):
    '''config for Life 4555'''
    gol = GameOfLife3D(geometry=geometry)
    gol.eb = 4
    gol.eh = 5
    gol.fb = 5
    gol.fh = 5

    return gol

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


# TESTS BASED ON https://content.wolfram.com/sites/13/2018/02/01-3-1.pdf

def test_5677_survival_at_lower_bound():
    '''alive cells with exacty 5 neiehghbours survives'''
    geometry = Geometry((2,2,2), axes='xy', periodicity='y')
    gol = GameOfLife3D(geometry=geometry)

    cell = make_cell(1)
    result = gol.apply(neighbors=make_neighbour(5), cell=cell)

    assert result['alive'] == 1

def test_5677_survival_at_upper_bound():
    '''alive cell with exactly 7 neighbours survives'''
    geometry = Geometry((2,2,2), axes='xy', periodicity='y')
    gol = GameOfLife3D(geometry=geometry)

    cell = make_cell(1)
    result = gol.apply(neighbors=make_neighbour(7), cell=cell)

    assert result['alive'] == 1

def test_5677_death_below_lower_bound():
    geometry = Geometry((2,2,2), axes='xy', periodicity='y')
    gol = GameOfLife3D(geometry=geometry)

    cell = make_cell(1)
    result = gol.apply(neighbors=make_neighbour(4), cell=cell)

    assert result['alive'] == 0

def test_5677_death_above_upper_bound():
    geometry = Geometry((2,2,2), axes='xy', periodicity='y')
    gol = GameOfLife3D(geometry=geometry)

    cell = make_cell(1)
    result = gol.apply(neighbors=make_neighbour(8), cell=cell)

    assert result['alive'] == 0

