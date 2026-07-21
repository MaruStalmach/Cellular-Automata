import pytest
from itertools import product
import numpy as np
from libs.neighbourhoods.MooreNeighbourhood import MooreNeighbourhood
from libs.neighbourhoods.VonNeumannNeighbourhood import VonNeumannNeighbourhood


@pytest.fixture
def moore():
    return MooreNeighbourhood()

@pytest.fixture
def von_neumann():
    return VonNeumannNeighbourhood()

###### 1D

def test_moore_1d(moore: MooreNeighbourhood):
    '''should shift the index one forward and one back for 1D'''
    offsets = moore.generate_offsets(ndim=1, radius=1)
    expected = np.array([[-1], [1]], dtype=np.int64)
    
    offsets = offsets[np.lexsort(offsets.T)]
    expected = expected[np.lexsort(expected.T)]
    
    assert np.array_equal(offsets, expected)

def test_von_neumann_1d(von_neumann: VonNeumannNeighbourhood):
    '''should shift the index one forward and one back for 1D'''
    offsets = von_neumann.generate_offsets(ndim=1, radius=1)
    expected = np.array([[-1], [1]], dtype=np.int64)
    
    offsets = offsets[np.lexsort(offsets.T)]
    expected = expected[np.lexsort(expected.T)]
    
    np.testing.assert_array_equal(offsets, expected)

###### 2D

def test_moore_2d(moore: MooreNeighbourhood):
    '''should shift the index one forward and one back + 
    one above and one below + diagonals for 2D'''
    offsets = moore.generate_offsets(ndim=2, radius=1)

    assert len(offsets) == 8

    #center should not be in offsets
    for offset in offsets:
        assert not (offset[0] == 0 and offset[1] == 0)

    expected = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    for point in expected:
        assert any(np.array_equal(offset, point) for offset in offsets)

def test_von_neumann_2d(von_neumann: VonNeumannNeighbourhood):
    '''should shift the index one forward and one back + 
    one above and one below for 2D'''
    offsets = von_neumann.generate_offsets(ndim=2, radius=1)
    expected = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
    
    assert len(offsets) == 4
    
    offsets = offsets[np.lexsort(offsets.T)]
    expected = expected[np.lexsort(expected.T)]
    
    assert np.array_equal(offsets, expected)


@pytest.mark.parametrize("ndim, radius", [(0, 1), (2, 0)])
def test_invalid_arguments(
    moore: MooreNeighbourhood, 
    von_neumann: VonNeumannNeighbourhood, 
    ndim: int, 
    radius: int
):
    
    with pytest.raises(AssertionError):
        moore.generate_offsets(ndim=ndim, radius=radius)
        
    with pytest.raises(AssertionError):
        von_neumann.generate_offsets(ndim=ndim, radius=radius)

###### 3D

def test_moore_3d(moore):
    '''should shift the index one forward and one back + 
    one above and one below in each dim + diagonals for 3D'''
    offsets = moore.generate_offsets(ndim=3, radius=1)
    
    assert len(offsets) == 26

    for offset in offsets:
        assert not all(component == 0 for component in offset)

    expected_set = set(product((-1, 0, 1), repeat=3))
    expected_set.remove((0, 0, 0)) 
    
    actual_set = set(tuple(offset) for offset in offsets)
    assert actual_set == expected_set

def test_von_neumann_3d(von_neumann):
    '''should shift the index one forward and one back + 
    one above and one below in each dim for 3D'''
    offsets = von_neumann.generate_offsets(ndim=3, radius=1)
    
    assert len(offsets) == 6
    
    expected_set = {
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1)
    }
    
    actual_set = set(tuple(offset) for offset in offsets)
    assert actual_set == expected_set

def test_radius_scaling_3d(moore, von_neumann):
    radius = 2
    moore_offsets = moore.generate_offsets(ndim=3, radius=radius)
    vn_offsets = von_neumann.generate_offsets(ndim=3, radius=radius)

    expected_moore_len = 3**3 - 1
    assert len(moore_offsets) == expected_moore_len
    
    assert len(vn_offsets) == 6
    
    vn_actual_set = set(tuple(offset) for offset in vn_offsets)
    assert (-radius, 0, 0) in vn_actual_set
    assert (0, radius, 0) in vn_actual_set
    assert (-1, 0, 0) not in vn_actual_set