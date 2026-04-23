from libs.Geometry import Geometry
import pytest

test_data = [
    ((2,3), 'xy', '', 2, 6),
    ((3,4,1), 'xyz', "y", 3, 12),
    ((2,1,5,2), 'xyza', 'a', 4, 20),
    ((5,5), 'xy', '', 2, 25),
    ((10,10,10), 'xyz', '', 3, 1000),
]

@pytest.mark.parametrize("size,axes,periodicity,expected_ndim,expected_cells", test_data)
def test_geometry_initialisation(size, axes, periodicity, expected_ndim, expected_cells):
    geometry = Geometry(size, axes, periodicity)

    #check basic properties
    assert geometry.size == size
    assert geometry.ndim == expected_ndim
    assert geometry.num_cells == expected_cells
    assert geometry.axes == axes.lower()
    assert geometry.periodicity == periodicity.lower()
    
    #check periodic dimensions
    periodic_dims = set()
    for i, ax in enumerate(axes):
        if ax in geometry.periodicity:
            periodic_dims.add(i)
    
    assert geometry.periodic_dims == periodic_dims

# def test_applying_offset():
#     geometry = Geometry((2,2), axes='xy', periodicity='y')

#     adj_matrix = geometry.generate_neighbourhood_matrix()

#     print(ad)