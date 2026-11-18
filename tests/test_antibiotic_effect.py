"""AntibioticEffect: constants come from the geometry, probabilities are exact (BUG-8)."""

from types import SimpleNamespace

import numpy as np
import pytest

from cellular_automata.Geometry import Geometry
from cellular_automata.neighbourhoods.VonNeumannNeighbourhood import (
    VonNeumannNeighbourhood,
)
from cellular_automata.rules import AntibioticEffect
from cellular_automata.State import State

SIZE = (4, 4, 4)


def _rule(size=SIZE, neighbourhood=None, **kwargs):
    axes = "xyz"[: len(size)]
    geometry = Geometry(size, axes=axes, periodicity=axes[:2], neighbourhood=neighbourhood)
    args = {"time_step": 1800, "bacteria_keys": ["bacteria"], "biofilm_key": "biofilm",
            "effectiveness": [0.0], "base_prob": 1.041e-6}
    args.update(kwargs)
    return AntibioticEffect(geometry, **args)


def _state(rule, step_no=0, max_steps=10):
    state = State(rule.geometry, cell_keys=["bacteria", "biofilm"])
    state["bacteria"] = np.ones(rule.geometry.size)
    state.sim = SimpleNamespace(step_no=step_no, max_steps=max_steps)
    return state


def _no_biofilm(rule):
    return np.zeros(rule.geometry.size, dtype=np.uint8)


class FixedRolls:
    """Stands in for rule.rng: every roll is `value`."""

    def __init__(self, value):
        self.value = value

    def random(self, shape):
        return np.full(shape, self.value)


def test_death_probability_is_exact_at_calibrated_magnitudes():
    # float16 stored base_prob 1.041e-6 as 1.013e-6 (-2.7%), and 1800 // 0.01 gave 179999
    rule = _rule()

    probability = rule.death_probability(_no_biofilm(rule))

    p = 1.041e-6 * 1800 / 0.01
    assert probability[0, 0, 1] == pytest.approx(p, rel=1e-12)
    # bottom layer: 9 of the 26 Moore neighbours are wall, out of 27 cells
    assert probability[0, 0, 0] == pytest.approx(p / (1 + 9 / 27), rel=1e-12)


def test_time_steps_below_the_reference_scale_down():
    # used to become a fixed 0.1 instead of time_step / 0.01
    rule = _rule(time_step=0.005, base_prob=0.1)

    assert rule.death_probability(_no_biofilm(rule))[0, 0, 1] == pytest.approx(0.05)


def test_wall_protection_follows_the_neighbourhood():
    rule = _rule(neighbourhood=VonNeumannNeighbourhood(), time_step=0.01, base_prob=0.1)

    probability = rule.death_probability(_no_biofilm(rule))

    # 1 of the 6 von Neumann neighbours is wall, out of 7 cells
    assert probability[0, 0, 0] == pytest.approx(0.1 / (1 + 1 / 7))
    assert probability[0, 0, 1] == pytest.approx(0.1)


def test_wall_is_below_the_last_axis_in_2d():
    rule = _rule(size=(4, 4), time_step=0.01, base_prob=0.1)

    probability = rule.death_probability(_no_biofilm(rule))

    # 2D Moore: 3 of 8 neighbours are wall, out of 9 cells
    assert probability[1, 0] == pytest.approx(0.1 / (1 + 3 / 9))
    assert probability[1, 1] == pytest.approx(0.1)


def test_biofilm_neighbours_protect():
    rule = _rule()
    biofilm = _no_biofilm(rule)
    biofilm[2, 2, 3] = 1  # one neighbour for (2, 2, 2), none for (0, 0, 2)

    probability = rule.death_probability(biofilm)

    assert probability[2, 2, 2] == pytest.approx(probability[0, 0, 2] / (1 + 1 / 27))


@pytest.mark.parametrize(
    "effectiveness, dies",
    [(0.0, False), (0.4, False), (1.0, True), (-1.0, False)],
)
def test_effectiveness_scales_the_death_probability(effectiveness, dies):
    # death probability away from wall and biofilm is 0.2 * (1 + effectiveness)
    rule = _rule(time_step=0.01, base_prob=0.2, effectiveness=[effectiveness])
    rule.rng = FixedRolls(0.3)

    state = rule.apply_state(_state(rule))

    above_the_wall = state["bacteria"][:, :, 1:]
    assert (above_the_wall.sum() == 0) == dies


@pytest.mark.parametrize("step_no, max_steps", [(4, 10), (6, 10), (9, 10)])
def test_dosing_window(step_no, max_steps):
    rule = _rule(base_prob=1.0, inoculation_start=5)

    state = rule.apply_state(_state(rule, step_no=step_no, max_steps=max_steps))

    dosed = 5 <= step_no < max_steps
    assert (state["bacteria"].sum() == 0) == dosed


def test_open_dosing_window_follows_the_current_max_steps():
    # the end used to be fixed at the max_steps of the first run that applied the rule
    rule = _rule(base_prob=1.0)
    rule.apply_state(_state(rule, step_no=0, max_steps=10))

    state = rule.apply_state(_state(rule, step_no=15, max_steps=20))

    assert state["bacteria"].sum() == 0


def test_mismatched_keys_and_effectiveness_raise():
    with pytest.raises(ValueError, match="effectiveness"):
        _rule(bacteria_keys=["a", "b"], effectiveness=[0.5])


def test_effectiveness_below_minus_one_raises():
    with pytest.raises(ValueError, match=">= -1"):
        _rule(effectiveness=[-1.5])
