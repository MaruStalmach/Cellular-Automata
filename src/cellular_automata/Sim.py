from __future__ import annotations

import operator
import secrets
import warnings
from time import time

import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.Rule import Rule
from cellular_automata.State import State
from cellular_automata.tracking.StateTracker import StateTracker

# upper bound on how often one rule runs per step, to keep a step's cost bounded
MAX_RUNS_PER_STEP = 500


def resolve_seed(seed: int | None) -> int:
    """Return `seed`, or a fresh random seed if it is None (so every run has one).

    Fresh seeds are 63-bit so they survive a round trip through JSON tools.
    """
    if seed is None:
        return secrets.randbits(63)
    try:
        value = operator.index(seed)
    except TypeError:
        value = None
    if isinstance(seed, (bool, np.bool_)) or value is None or value < 0:
        raise ValueError(f"sim.seed must be a non-negative integer, got {seed!r}")
    return int(value)


def state_rng(seed: int) -> np.random.Generator:
    """Generator for the initial state of a simulation seeded with `seed`."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0,)))


def rule_rng(seed: int, index: int) -> np.random.Generator:
    """Generator for the rule at `index`; independent of the other rules' streams."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1, index)))


class CellularAutomaton:
    """Simulation engine

    args:
    - geometry - Geometry object
    - rules - list of Rule objects used within the siimulation
    - seed - seeds each rule's `rng` (and the state, if the simulation creates
      it; a `state` passed in keeps its own values). If None a random seed is
      picked and stored in `self.seed`
    """

    def __init__(
        self,
        geometry: Geometry,
        rules: list[Rule],
        state: State = None,
        max_steps: int = 100,
        tracker: StateTracker | None = None,
        output_filename: str | None = None,
        seed: int | None = None,
    ):
        self.seed = resolve_seed(seed)
        self.rules = rules
        self.geometry = geometry
        self.max_steps = max_steps
        self.tracker = tracker
        self.output_filename = output_filename
        self.start_time = 0.0
        self._saved_at_step: int | None = None
        self.max_dt = 0

        # collect the keys the rules need; one step lasts as long as the largest rule dt
        keys: list[str] = []
        for rule in rules:
            if rule.dt > self.max_dt:
                self.max_dt = rule.dt
            for key in rule.required_keys:
                if key not in keys:
                    keys.append(key)

        for index, rule in enumerate(rules):
            rule.rng = rule_rng(self.seed, index)

        self.rule_exec_n_times = [self._runs_per_step(rule) for rule in self.rules]

        if state is not None:
            self.state = state
            # TODO: check if state keys are the same as rule keys
            try:
                assert keys == tuple(self.state.keys)
            except AssertionError:
                print(
                    "key mismatch between rules and state arrays (soft error)"
                )  # TODO: comparing tuple to list will always thorw a mismacth
        else:
            self.state = State(
                geometry, random=True, cell_keys=keys, rng=state_rng(self.seed)
            )
        self.state._get_sim_reference(self)

        self.step_no = 0
        if self.tracker:
            self.tracker.seed = self.seed
            self.tracker.record_step(self.state, self.step_no)

    def _runs_per_step(self, rule: Rule) -> int:
        """How many times `rule` runs during one step, which lasts `max_dt`."""
        ratio = self.max_dt / rule.dt
        runs = round(ratio)
        if not np.isclose(ratio, runs, rtol=1e-9, atol=0.0):
            runs = int(np.floor(ratio))
            warnings.warn(
                f"{type(rule).__name__}: the step length {self.max_dt} is not a whole "
                f"multiple of its time_step {rule.dt}; it runs {runs} times per step",
                stacklevel=3,
            )
        if runs > MAX_RUNS_PER_STEP:
            warnings.warn(
                f"{type(rule).__name__}: would run {runs} times per step "
                f"(step length {self.max_dt} / time_step {rule.dt}) but is capped at "
                f"{MAX_RUNS_PER_STEP}, so it advances slower than its time_step implies",
                stacklevel=3,
            )
        return int(np.clip(runs, 1, MAX_RUNS_PER_STEP))

    def step(self):
        '''increments the timestep of the simulation and applies all '''
        if self.step_no>=self.max_steps:
            print(f'simulation ended. time={time()-self.start_time:.2f}s')
            return -1
        if self.step_no==0:
            self.start_time = time()
        
        times_to_execute = self.rule_exec_n_times.copy()
        while sum(times_to_execute) > 0:
            for i, rule in enumerate(self.rules):
                if times_to_execute[i] <= 0:
                    continue
                self.state = rule.apply_state(self.state)
                times_to_execute[i] -= 1

        self.step_no += 1
        if self.tracker:
            self.tracker.record_step(self.state, self.step_no)

        if self.step_no >= self.max_steps:
            self.save_tracker()
        return 0

    def save_tracker(self) -> None:
        """Save the tracked history so far, e.g. when a run is interrupted.

        Called automatically once max_steps is reached; does nothing without a
        tracker and output filename, or if this step was already saved.
        """
        if not (self.tracker and self.output_filename):
            return
        if self._saved_at_step == self.step_no:
            return
        self.tracker.save(self.output_filename)
        self._saved_at_step = self.step_no
