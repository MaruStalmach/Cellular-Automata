from libs.Rules import Rule, GameOfLife3D
from libs.Geometry import Geometry
from libs.State import State

import pytest

def setup_gol_test(center_val: int, neighbour_count: int) -> tuple[GameOfLife3D, State, tuple[int, int, int]]:
    geometry = Geometry((3, 3, 3), axes='xyz', periodicity='')
    #default parameters: eb=5, eh=7, fb=6, fh=6
    gol = GameOfLife3D(geometry)
    state = State(geometry, random=False, cell_keys=['alive'])

    center = (1, 1, 1)
    state['alive'][center] = center_val

    for i in range(neighbour_count):
        ox, oy, oz = geometry._offsets[i]
        state['alive'][center[0] + ox, center[1] + oy, center[2] + oz] = 1
        
    return gol, state, center


#test data matrix: (center_state, active_neighbours, expected_result, test_description)
test_data = [
    (0, 6, 1, "birth at exact fertility (fb=6)"),
    (1, 5, 1, "survival at lower bound (eb=5)"),
    (1, 7, 1, "survival at upper bound (eh=7)"),
    (1, 4, 0, "death below lower bound (eb-1=4)"),
    (1, 8, 0, "death above upper bound (eh+1=8)"),
    (0, 5, 0, "no birth below fertility (fb-1=5)"),
    (0, 7, 0, "no birth above fertility (fh+1=7)"),
    (0, 0, 0, "dead cell with 0 neighbours stays dead"),
    (1, 0, 0, "alive cell with 0 neighbours dies"),
]
@pytest.mark.parametrize("center_val, neighbours, expected, desc", test_data)
def test_gol_3d_rules(center_val, neighbours, expected, desc):
    gol, state, center = setup_gol_test(center_val, neighbours)
    
    new_state = gol.apply_state(state)
    
    assert new_state['alive'][center] == expected, f"Failed: {desc}"
    
    
if __name__=="__main__":
    for d in test_data:
        test_gol_3d_rules(*d)