import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.rules.BioreactorBiofilmDetachment import BioreactorBiofilmDetachment
from cellular_automata.State import State

SIZE = (4, 4, 4)


def _setup(rate: float, key: str = "biofilm"):
    geometry = Geometry(SIZE, axes="xyz", periodicity="")
    rule = BioreactorBiofilmDetachment(
        geometry, time_step=1, biofilm_key=key, detachment_rate=rate, scaling=1.0
    )
    state = State(geometry, cell_keys=[key], random=False)
    state[key][...] = 1
    return rule, state


def test_required_keys_is_the_biofilm_key():
    rule, _ = _setup(rate=0.0, key="my_biofilm")

    assert rule.required_keys == ["my_biofilm"]


def test_zero_rate_detaches_nothing():
    rule, state = _setup(rate=0.0)

    state = rule.apply_state(state)

    assert int(state["biofilm"].sum()) == int(np.prod(SIZE))


def test_certain_detachment_clears_everything_except_the_wall_layer():
    # probability grows with z**2 and is 0 at the wall (z = 0)
    rule, state = _setup(rate=1.0)

    state = rule.apply_state(state)

    assert state["biofilm"][:, :, 0].all()
    assert not state["biofilm"][:, :, 1:].any()


def test_custom_key_is_written():
    rule, state = _setup(rate=1.0, key="my_biofilm")

    state = rule.apply_state(state)

    assert not state["my_biofilm"][:, :, 1:].any()
