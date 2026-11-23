"""Rules declare the keys they use, and bad arguments raise ValueError (ARCH-6)."""

import ast
from pathlib import Path

import numpy as np
import pytest

from cellular_automata.Callback import Callback, MultiKeyCallback, ReduceCallback
from cellular_automata.Geometry import Geometry
from cellular_automata.neighbourhoods.MooreNeighbourhood import MooreNeighbourhood
from cellular_automata.rules import (
    BiofilmDetachment,
    BioreactorBiofilmDetachment,
    Diffusion,
    GameOfLife3D,
    GutDrift,
    Inoculation,
)
from cellular_automata.Sim import CellularAutomaton
from cellular_automata.State import State
from tests.rule_fixtures import (
    CTX,
    OTHER_RULE_IDS,
    RULE_IDS,
    make_geometry,
    make_state,
    other_rules,
    species_rules,
)

SRC = Path(__file__).resolve().parent.parent / "src" / "cellular_automata"


class RecordingState:
    """Wraps a State and records every key a rule reads or writes."""

    def __init__(self, state):
        self._state = state
        self.touched = set()

    def __getitem__(self, key):
        self.touched.add(key)
        return self._state[key]

    def __setitem__(self, key, value):
        self.touched.add(key)
        self._state[key] = value

    def __getattr__(self, name):
        return getattr(self._state, name)


def _all_rules(geometry):
    return species_rules(geometry) + other_rules(geometry)


@pytest.mark.parametrize(
    "index", range(len(RULE_IDS + OTHER_RULE_IDS)), ids=RULE_IDS + OTHER_RULE_IDS
)
def test_rule_declares_exactly_the_keys_it_uses(index):
    geometry = make_geometry()
    rule, keys, layers = _all_rules(geometry)[index]
    state = RecordingState(make_state(geometry, keys, layers))

    rule.apply_state(state, CTX)

    assert sorted(rule.required_keys) == sorted(keys)
    assert state.touched == set(rule.required_keys)


def test_every_registered_rule_is_covered():
    from cellular_automata.registry import RULES

    covered = {type(rule).__name__ for rule, _, _ in _all_rules(make_geometry())}

    assert covered == set(RULES.names())


def test_missing_state_key_names_the_rule_and_key():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    rule = GameOfLife3D(geometry, time_step=1, target_key="cells")
    state = State(geometry, cell_keys=["alive"])

    with pytest.raises(ValueError, match="GameOfLife3D needs 'cells'"):
        CellularAutomaton(geometry, [rule], state=state, seed=1)


def test_extra_state_keys_are_allowed():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    rule = GameOfLife3D(geometry, time_step=1)
    state = State(geometry, cell_keys=["alive", "only_tracked"])

    sim = CellularAutomaton(geometry, [rule], state=state, seed=1)

    assert sim.state is state


@pytest.mark.parametrize(
    "make_rule",
    [
        lambda g: BiofilmDetachment(g, time_step=1, detachment_rate=0.1, scaling=1.0),
        lambda g: BioreactorBiofilmDetachment(
            g, time_step=1, biofilm_key="biofilm", detachment_rate=0.1, scaling=1.0
        ),
        lambda g: GutDrift(g, time_step=1, drift_speed=1, species_keys=["A"]),
        lambda g: Diffusion(g, time_step=1, a=7.4, p=0.07),
    ],
    ids=["BiofilmDetachment", "BioreactorBiofilmDetachment", "GutDrift", "Diffusion"],
)
def test_3d_only_rules_reject_other_grids_when_built(make_rule):
    with pytest.raises(ValueError, match="only works on 3D grids, got 2D"):
        make_rule(Geometry((4, 4), axes="xy", periodicity=""))


def test_inoculation_fills_layer_0_of_a_layered_key_by_default():
    geometry = Geometry((4, 4, 4), axes="xyz", periodicity="")
    rule = Inoculation(geometry, time_step=1, target_key="s", location=[1, 1, 3], size=[2, 2, 1])
    state = State(geometry, cell_keys=["s"], key_layers=[6])

    state = rule.apply_state(state, CTX)

    assert state["s"][1:3, 1:3, 3, 0].min() == 1
    assert state["s"].sum() == 4


def test_inoculation_layer_and_2d_box():
    geometry = Geometry((4, 4), axes="xy", periodicity="")
    rule = Inoculation(geometry, time_step=1, target_key="s", location=[0, 2], size=[1, 2], layer=2)
    state = State(geometry, cell_keys=["s"], key_layers=[3])

    state = rule.apply_state(state, CTX)

    assert state["s"][0, 2:4, 2].tolist() == [1, 1]
    assert state["s"].sum() == 2


def test_inoculation_layer_must_exist():
    geometry = Geometry((4, 4, 4), axes="xyz", periodicity="")
    with pytest.raises(ValueError, match="layer"):
        Inoculation(
            geometry, time_step=1, target_key="s", location=[0, 0, 0], size=[1, 1, 1], layer=-1
        )
    rule = Inoculation(
        geometry, time_step=1, target_key="s", location=[0, 0, 0], size=[1, 1, 1], layer=6
    )

    with pytest.raises(ValueError, match="layer 6 but 's' has 6 layers"):
        rule.apply_state(State(geometry, cell_keys=["s"], key_layers=[6]))


def test_inoculation_box_needs_one_value_per_axis():
    geometry = Geometry((4, 4, 4), axes="xyz", periodicity="")

    with pytest.raises(ValueError, match="one value per axis"):
        Inoculation(geometry, time_step=1, target_key="s", location=[0, 0], size=[1, 1])


def test_callback_needs_a_simulation():
    with pytest.raises(TypeError, match="CellularAutomaton"):
        Callback("alive", object())


def test_reduce_callback_needs_a_layered_key():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    sim = CellularAutomaton(geometry, [GameOfLife3D(geometry, time_step=1)], seed=1)

    with pytest.raises(ValueError, match="several layers"):
        ReduceCallback("alive", sim)
    with pytest.raises(TypeError, match="ufunc"):
        ReduceCallback("alive", sim, reduce_op=max)


def test_library_has_no_assert_statements():
    # asserts vanish under `python -O`, so validation must raise real exceptions
    offenders = [
        f"{path.relative_to(SRC)}:{node.lineno}"
        for path in SRC.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Assert)
    ]

    assert offenders == []


def test_neighbourhood_arguments_raise_value_error():
    with pytest.raises(ValueError, match="radius"):
        MooreNeighbourhood().generate_offsets(3, radius=0)
    assert np.array_equal(MooreNeighbourhood().generate_offsets(1), np.array([[-1], [1]]))


def test_multi_key_callback_reduces_layered_keys_on_2d_grids():
    geometry = Geometry((3, 3), axes="xy", periodicity="")
    state = State(geometry, cell_keys=["alive", "s"], key_layers=[1, 4])
    sim = CellularAutomaton(geometry, [GameOfLife3D(geometry, time_step=1)], state=state, seed=1)

    callback = MultiKeyCallback(["alive", "s"], sim)

    assert [type(cb) for cb in callback.callbacks] == [Callback, ReduceCallback]
