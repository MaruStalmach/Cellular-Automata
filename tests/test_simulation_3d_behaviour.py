import numpy as np

import pytest

from libs.Cells import Cell, State
from libs.Geometry import Geometry
from libs.Rules import GameOfLife3D
from libs.Sim import CellularAutomaton


@pytest.mark.parametrize(
    "starting_coords_live_cells, expected_coords_live_cells",
    [
        (
            #3D "cross" shape -> survival threshold eb=5 therefore only the center cell survives
            [(1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (0, 1, 1), (2, 1, 1)],
            [(1, 1, 1)] 
        ),
        (
            #2x2x2 grid -> survbival threshold eb=5, eh=7, every cell has exactly 7 neighbours 
            [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)],
            [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
        )
    ]
)
def test_if_step_matches_manual_3d_result(starting_coords_live_cells, expected_coords_live_cells):
    geometry = Geometry(size=(4,4,4), axes='xyz', periodicity='')
    gol = GameOfLife3D(geometry=geometry)
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    state = State(geometry=geometry, random=False, cell_keys=['alive'])
    initial = np.empty(geometry.size, dtype=object)

    for coords in np.ndindex(geometry.size):
        cell = Cell(keys=['alive'], random=False)
        cell['alive'] = 0
        initial[coords] = cell
    
    for coords in starting_coords_live_cells:
        initial[coords]['alive'] = 1
    
    state = initial
    ca.state.data = state

    ca.step()

    expected = np.zeros(geometry.size, dtype=int)
    for coords in expected_coords_live_cells:
        expected[coords] = 1
    
    actual = np.zeros(geometry.size, dtype=int)
    for coords in np.ndindex(geometry.size):
        actual[coords] = ca.state.data[coords]['alive']

    assert np.array_equal(actual, expected)
    assert ca.step_no == 1


@pytest.mark.parametrize(
    "periodicity, survives",
    [
        ('x', True),
        ('xy', True),  
        ('y', False),  
        ('', False)   
    ]
)
def test_periodicity_wrapping_after_step(periodicity, survives):
    geometry = Geometry((4,4,4), axes='xyz', periodicity=periodicity)
    gol = GameOfLife3D(geometry=geometry)
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    state = State(geometry=geometry, random=False, cell_keys=['alive'])
    initial = np.empty(geometry.size, dtype=object)

    for coords in np.ndindex(geometry.size):
        cell = Cell(keys=['alive'], random=False)
        cell['alive'] = 0
        initial[coords] = cell

    #splitting 2x2x2 block of cells across the boundary
    split_block_coords = [
        (0,1,1), (0,1,2), (0,2,1), (0,2,2),
        #moved alongside x axis to the left
        (3,1,1), (3,1,2), (3,2,1), (3,2,2)
    ] 

    for coords in split_block_coords:
        initial[coords]['alive'] = 1

    state.data = initial
    ca.state = state

    ca.step()

    actual = np.zeros(geometry.size, dtype=int)
    for coords in np.ndindex(geometry.size):
        actual[coords] = ca.state.data[coords]['alive']

    expected = np.zeros(geometry.size, dtype=int)
    if survives:
        for coords in split_block_coords:
            expected[coords] = 1 #block stays exactly as it was

    assert np.array_equal(actual, expected)
    assert ca.step_no == 1