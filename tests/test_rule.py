"""Tests for the shared helpers on the Rule base class."""

import numpy as np
import pytest

from cellular_automata.Geometry import Geometry
from cellular_automata.Rule import Rule


class _NoOpRule(Rule):
    def apply_state(self, state, ctx=None):
        return state


def _rule(size=(3, 3, 3), periodicity="") -> Rule:
    return _NoOpRule(Geometry(size, axes="xyz", periodicity=periodicity), time_step=1)


def test_a_rule_without_apply_state_cannot_be_built():
    class Incomplete(Rule):
        pass

    with pytest.raises(TypeError, match="apply_state"):
        Incomplete(Geometry((3, 3, 3), axes="xyz", periodicity=""), time_step=1)


def test_the_base_rule_cannot_be_built():
    with pytest.raises(TypeError, match="abstract"):
        Rule(Geometry((3, 3, 3), axes="xyz", periodicity=""), time_step=1)


def test_neighborhood_count_does_not_overflow_with_dense_layers():
    # 26 Moore neighbours x 6 layers = 156 particles around the centre cell
    rule = _rule()
    substrate = np.ones((3, 3, 3, 6), dtype=np.uint8)

    count = rule.neighborhood_count(substrate)

    assert count[1, 1, 1] == 156
    assert count.min() >= 0


def test_neighborhood_count_does_not_overflow_on_single_layer_values():
    rule = _rule()
    grid = np.full((3, 3, 3), 10, dtype=np.uint8)

    count = rule.neighborhood_count(grid)

    assert count[1, 1, 1] == 260


def _shift_of_single_cell(periodicity: str, offset: tuple[int, int, int]) -> np.ndarray:
    rule = _rule(size=(5, 1, 1), periodicity=periodicity)
    grid = np.zeros((5, 1, 1), dtype=np.int32)
    grid[2, 0, 0] = 1
    return rule._shift_for_offset(grid, offset)[:, 0, 0]


def test_shift_reads_the_neighbour_at_offset_on_non_periodic_axis():
    # result[i] == grid[i + offset]: the cell at 2 is the +1 neighbour of cell 1
    assert _shift_of_single_cell("", (1, 0, 0)).tolist() == [0, 1, 0, 0, 0]


def test_shift_reads_the_neighbour_at_offset_on_periodic_axis():
    assert _shift_of_single_cell("x", (1, 0, 0)).tolist() == [0, 1, 0, 0, 0]


def test_shift_wraps_only_on_periodic_axis():
    rule = _rule(size=(3, 1, 1), periodicity="x")
    grid = np.array([1, 0, 0], dtype=np.int32).reshape(3, 1, 1)
    assert rule._shift_for_offset(grid, (1, 0, 0))[:, 0, 0].tolist() == [0, 0, 1]

    rule = _rule(size=(3, 1, 1), periodicity="")
    assert rule._shift_for_offset(grid, (1, 0, 0))[:, 0, 0].tolist() == [0, 0, 0]


def test_shift_handles_offsets_larger_than_one_on_non_periodic_axis():
    assert _shift_of_single_cell("", (2, 0, 0)).tolist() == [1, 0, 0, 0, 0]
    assert _shift_of_single_cell("", (-2, 0, 0)).tolist() == [0, 0, 0, 0, 1]
