from libs.Geometry import Geometry
import pytest
import numpy as np
from scipy.sparse import csr_matrix


init_test_data = [
    ((10, 10), "xy", "y", 2, 100, {1}, 8),
    ((3, 4, 2), "xyz", "xy", 3, 24, {0, 1}, 26),
    ((5,), "x", "", 1, 5, set(), 2),
    ((2, 2, 2, 2), "xyza", "xyza", 4, 16, {0, 1, 2, 3}, 80),
]


@pytest.mark.parametrize(
    "size, axes, periodicity, expected_ndim, expected_num_cells, expected_periodic_dims, expected_offsets",
    init_test_data,
)
def test_geometry_initialization(
    size,
    axes,
    periodicity,
    expected_ndim,
    expected_num_cells,
    expected_periodic_dims,
    expected_offsets,
):
    geometry = Geometry(size=size, axes=axes, periodicity=periodicity)

    assert geometry.ndim == expected_ndim
    assert geometry.num_cells == expected_num_cells
    assert geometry.periodic_dims == expected_periodic_dims
    assert len(geometry._offsets) == expected_offsets
    assert geometry._offsets.dtype == np.int64


def test_neighbourhood_matrix_2d_non_periodic():
    geometry = Geometry((3, 3), axes="xy", periodicity="")
    matrix = geometry.generate_neighbourhood_matrix()

    assert isinstance(matrix, csr_matrix)
    assert matrix.shape == (9, 9)

    # convert to dense array for further tests
    neighbor_counts = np.array(matrix.sum(axis=1)).flatten()

    # corner cell (0,0) -> flat index 0
    assert neighbor_counts[0] == 3
    # edge cell (0,1) -> flat index 1
    assert neighbor_counts[1] == 5
    # center cell (1,1) -> flat index 4
    assert neighbor_counts[4] == 8
