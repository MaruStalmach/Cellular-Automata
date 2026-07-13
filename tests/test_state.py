from libs.Geometry import Geometry
from libs.State import State
import numpy as np

import pytest

test_data = [
    ((2, 3), "xy", "", (2, 3), (2, 3, 1), (0, 0)),
    ((3, 4, 1), "xyz", "y", (3, 4, 1), (3, 4, 1, 1), (0, 0, 0)),
    ((2, 1, 5, 2), "xyza", "a", (2, 1, 5, 2), (2, 1, 5, 2, 1), (0, 0, 0, 0)),
]


@pytest.mark.parametrize(
    "size, axes, periodicity, expected_shape, expected_data_shape, start_coords",
    test_data,
)
def test_state_initialisation(
    size, axes, periodicity, expected_shape, expected_data_shape, start_coords
):
    geometry = Geometry(size=size, axes=axes, periodicity=periodicity)

    cell_keys = ["alive"]

    state = State(geometry=geometry, random=False, cell_keys=cell_keys)

    # check grid size
    assert state.shape == expected_shape

    # data = grid size+keys
    assert state.data.shape == expected_data_shape

    # check key access
    assert state.keys == tuple(cell_keys)

    # check cell state
    assert state["alive"][start_coords] == 0


def test_state_setters_and_copy():
    geometry = Geometry((3, 3), axes="xy", periodicity="")
    state = State(
        geometry=geometry, random=False, cell_keys=["alive", "oxygen saturation"]
    )

    state["alive"] = np.ones((3, 3))
    assert np.all(state["alive"] == 1)

    state["oxygen saturation"] = np.concatenate(
        (np.ones((2, 3)), np.zeros((1, 3))), axis=0
    )
    assert state["oxygen saturation"][0, 0] == 1
    assert state["oxygen saturation"][2, 2] == 0

    state_copy = state.copy()
    assert np.all(state_copy["alive"] == 1)

    state["alive"][0, 0] = 0
    assert state_copy["alive"][0, 0] == 1


def test_state_custom():
    geometry = Geometry((3, 3), axes="xy", periodicity="")

    def mock_array_generator(size):
        return np.full(size, 5)

    state = State(
        geometry=geometry,
        random=True,
        cell_keys=["HP"],
        random_func=mock_array_generator,
        dtype=np.int32,
        key_dtypes={"HP": np.float32},
    )

    assert state["HP"].dtype == np.float32
    assert state["HP"][0, 0] == 5.0


test_data = [
    ((3, 3), "xy", "", (5, 5), (0, 2), None, None),
    ((3, 4, 1), "xyz", "y", (5, 6, 3), (0, 2, 1), (2, 0, 1), (1, -1, 0)),
    (
        (2, 1, 5, 2),
        "xyza",
        "a",
        (4, 3, 7, 4),
        (0, 1, 3, 1),
        (1, 1, 3, 0),
        (0, 0, 2, -1),
    ),
]


@pytest.mark.parametrize(
    "size, axes, periodicity, expected_padded_size, const_coords, wrap_pad_coords, wrap_orig_coords",
    test_data,
)
def test_state_padding(
    size,
    axes,
    periodicity,
    expected_padded_size,
    const_coords,
    wrap_pad_coords,
    wrap_orig_coords,
):
    geometry = Geometry(size=size, axes=axes, periodicity=periodicity)
    state = State(geometry=geometry, random=False, cell_keys=["value"], dtype=np.int32)

    original_data = np.arange(np.prod(size)).reshape(size)
    state["value"] = original_data

    padded_data = state._pad_data(state["value"], constant_value=-1)
    assert padded_data.shape == expected_padded_size

    assert padded_data[const_coords] == -1

    if wrap_pad_coords is not None:
        assert padded_data[wrap_pad_coords] == original_data[wrap_orig_coords]
