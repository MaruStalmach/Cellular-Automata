import numpy as np
import pytest

from cellular_automata.Geometry import Geometry
from cellular_automata.State import State

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


def test_key_layers_accepts_any_iterable():
    geometry = Geometry((2, 2), axes="xy", periodicity="")

    state = State(geometry, cell_keys=["a", "b"], key_layers=(1, 3))

    assert state["a"].shape == (2, 2)
    assert state["b"].shape == (2, 2, 3)


def test_key_layers_must_match_the_keys():
    geometry = Geometry((2, 2), axes="xy", periodicity="")

    with pytest.raises(ValueError, match="2 cell_keys but 1 key_layers"):
        State(geometry, cell_keys=["a", "b"], key_layers=[1])


def test_copy_keeps_layers_and_dtypes_and_is_independent():
    geometry = Geometry((2, 2), axes="xy", periodicity="")
    state = State(
        geometry, cell_keys=["a", "b"], key_layers=(1, 3), key_dtypes={"b": np.float32}
    )
    state["b"][0, 0, 2] = 0.5

    copied = state.copy()
    state["b"][0, 0, 2] = 0.25

    assert copied["b"].shape == (2, 2, 3)
    assert copied["b"].dtype == np.float32
    assert copied["b"][0, 0, 2] == 0.5


def test_initialiser_is_called_once_with_size_and_cast_to_the_key_dtype():
    geometry = Geometry((2, 3), axes="xy", periodicity="")
    calls = []

    def init(size, value):
        calls.append(size)
        return np.full(size, value)

    state = State(
        geometry, cell_keys=["a"], key_dtypes={"a": np.uint8}, random=True,
        random_func=[init], random_args=[{"value": 7}],
    )

    assert calls == [(2, 3, 1)]
    assert state["a"].dtype == np.uint8
    assert state["a"].tolist() == [[7, 7, 7], [7, 7, 7]]


def test_initialiser_errors_are_not_swallowed():
    # a TypeError inside an initialiser used to trigger retries with fewer arguments
    geometry = Geometry((2, 2), axes="xy", periodicity="")
    calls = []

    def broken(size):
        calls.append(size)
        raise TypeError("bug inside the initialiser")

    with pytest.raises(TypeError, match="bug inside the initialiser"):
        State(geometry, cell_keys=["a"], random=True, random_func=[broken])
    assert len(calls) == 1


def test_initialiser_without_random_args():
    geometry = Geometry((2, 2), axes="xy", periodicity="")

    state = State(
        geometry, cell_keys=["a", "b"], random=True,
        random_func=[lambda size: np.ones(size), None], random_args=[None, None],
    )

    assert state["a"].sum() == 4
    assert state["b"].sum() == 0


def test_integer_initial_values_that_do_not_fit_the_dtype_raise():
    geometry = Geometry((2, 2), axes="xy", periodicity="")

    def too_big(size):
        values = np.zeros(size, dtype=np.int64)
        values[0, 0] = 300
        return values

    with pytest.raises(ValueError, match=r"key 'a' gave values 0\.\.300.*uint8"):
        State(
            geometry, cell_keys=["a"], key_dtypes={"a": np.uint8}, random=True,
            random_func=[too_big],
        )


def test_data_setter_round_trips_layered_keys():
    # np.stack over a layered slice used to move the first grid axis to the end
    geometry = Geometry((2, 3, 4), axes="xyz", periodicity="")
    state = State(geometry, cell_keys=["a", "sub"], key_layers=[1, 3])
    rng = np.random.default_rng(0)
    state["a"][...] = rng.integers(0, 2, size=(2, 3, 4))
    state["sub"][...] = rng.integers(0, 2, size=(2, 3, 4, 3))
    before = {key: state[key].copy() for key in state.keys}

    state.data = state.data

    for key in state.keys:
        assert state[key].shape == before[key].shape
        np.testing.assert_array_equal(state[key], before[key])
