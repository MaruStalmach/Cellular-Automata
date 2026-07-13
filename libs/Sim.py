from __future__ import annotations

import numpy as np

from libs.Geometry import Geometry
from libs.State import State
from libs.Rules import Rule


class CellularAutomaton:
    """Simulation engine

    args:
    - geometry - Geometry object
    - rules - list of Rule objects used within the siimulation
    """

    def __init__(self, geometry: Geometry, rules: list[Rule], state: State = None):
        self.rules = rules
        self.geometry = geometry
        self.neighbours = None
        self._neighbours_idx = None

        # collect necessary keys for rules
        keys: list[str] = []
        for rule in rules:
            for key in rule.required_keys:
                if key not in keys:
                    keys.append(key)

        if state is not None:
            self.state = state
            # TODO: check if state keys are the same as rule keys
            try:
                assert keys == self.state.keys
            except AssertionError:
                print("key mismatch between rules and state arrays (soft error)") #TODO: comparing tuple to list will always thorw a mismacth
        else:
            self.state = State(geometry, random=True, cell_keys=keys)
        self.step_no = 0

    def _apply_cellwise(self, rule: Rule):

        # builds neighbourhood index list
        if self._neighbours_idx is None:
            self.neighbours = self.geometry.generate_neighbourhood_matrix()
            self._neighbours_idx = [
                self.neighbours.indices[
                    self.neighbours.indptr[i] : self.neighbours.indptr[i + 1]
                ]
                for i in range(self.geometry.num_cells)
            ]

        source = self.state.data.copy()
        target = np.empty_like(source)
        # flatten to 2D
        flat_source = source.reshape(-1, source.shape[-1])
        flat_target = target.reshape(-1, target.shape[-1])

        # per-cell apply
        for cell_idx, neighbours_idx in enumerate(self._neighbours_idx):
            neighbours = flat_source[neighbours_idx]
            cell = flat_source[cell_idx].copy()
            coords = tuple(
                int(coord) for coord in np.unravel_index(cell_idx, self.geometry.size)
            )
            flat_target[cell_idx] = rule.apply(neighbours, cell, coords)

        self.state.data = target

    def step(self):
        """increments the timestep of the simulation and applies all"""
        for rule in self.rules:
            if hasattr(rule, "apply_state"):
                self.state = rule.apply_state(self.state)
            else:
                self._apply_cellwise(rule)

        self.step_no += 1
