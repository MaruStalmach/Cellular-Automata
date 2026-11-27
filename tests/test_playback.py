import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.State import State
from cellular_automata.tracking.playback import frame_for_keys, parse_keys


def test_parse_keys_splits_spaces_and_commas():
    assert parse_keys(["a,b", "c"], ("a", "b", "c", "d")) == ["a", "b", "c"]


def test_parse_keys_defaults_to_every_key():
    assert parse_keys(None, ("a", "b")) == ["a", "b"]


def test_frame_for_keys_collapses_layers_and_stacks_keys():
    geometry = Geometry((2, 2, 2), axes="xyz", periodicity="")
    state = State(geometry, cell_keys=["alive", "substrate"], key_layers=[1, 3])
    state["alive"][0, 0, 0] = 1
    state["substrate"][1, 1, 1, 2] = 1

    frame = frame_for_keys(state, ["alive", "substrate"], spatial_rank=3)

    assert frame.shape == (2, 2, 2, 2)
    assert frame[0, 0, 0, 0] == 1 and frame[0].sum() == 1
    assert frame[1, 1, 1, 1] == 1 and frame[1].sum() == 1


def test_frame_for_one_key_is_not_stacked():
    geometry = Geometry((2, 2, 2), axes="xyz", periodicity="")
    state = State(geometry, cell_keys=["alive"])

    assert frame_for_keys(state, ["alive"], spatial_rank=3).shape == (2, 2, 2)
    assert np.array_equal(frame_for_keys(state, ["alive"], 3), state["alive"])
