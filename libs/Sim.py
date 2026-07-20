from __future__ import annotations

import numpy as np

from libs.Geometry import Geometry
from libs.State import State
from libs.Rule import Rule


class CellularAutomaton:
    """Simulation engine

    args:
    - geometry - Geometry object
    - rules - list of Rule objects used within the siimulation
    """

    def __init__(self, geometry: Geometry, rules: list[Rule], state : State = None, max_steps:int=100):
        self.rules = rules
        self.geometry = geometry
        self.neighbours = None
        self._neighbours_idx = None
        self.max_steps=max_steps
        self.time = 0.0
        self.max_dt = 0

        # collect necessary keys for rules and find the smallest dt in rules
        keys: list[str] = []
        for rule in rules:
            if rule.dt > self.max_dt:
                self.max_dt = rule.dt
            for key in rule.required_keys:
                if key not in keys:
                    keys.append(key)
                    
        self.rule_exec_n_times = []
        #calculate amount execution times for each rule
        for rule in self.rules:
            times_to_execute = int(np.clip(self.max_dt//rule.dt,1,500))
            self.rule_exec_n_times.append(times_to_execute)                    
    

        if state is not None:
            self.state = state
            # TODO: check if state keys are the same as rule keys
            try:
                assert keys == tuple(self.state.keys)
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
        '''increments the timestep of the simulation and applies all '''
        if self.step_no>=self.max_steps:
            print('simulation ended')
            quit()
            return
        
        times_to_execute = self.rule_exec_n_times.copy()
        breakpoint()
        while sum(times_to_execute)>0:
            for i,rule in enumerate(self.rules):
                if times_to_execute[i] <= 0:
                    continue
                if hasattr(rule, "apply_state"):
                    self.state = rule.apply_state(self.state)
                else:
                    self._apply_cellwise(rule)
                times_to_execute[i] -= 1

        self.step_no += 1
