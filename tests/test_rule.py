"""Tests for the shared helpers on the Rule base class."""

import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.Rule import Rule


def _rule(size=(3, 3, 3), periodicity="") -> Rule:
    return Rule(Geometry(size, axes="xyz", periodicity=periodicity), time_step=1)


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
