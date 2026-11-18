from dataclasses import dataclass

import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.State import State


@dataclass(frozen=True)
class StepContext:
    """What a rule may know about the run it is part of, passed to apply_state."""

    step_no: int
    max_steps: int


def require_3d(rule_name: str, geometry: Geometry) -> None:
    """Raise ValueError unless `geometry` is 3D (for rules written for 3D grids only)."""
    if geometry.ndim != 3:
        raise ValueError(f"{rule_name} only works on 3D grids, got {geometry.ndim}D")


class Rule:
    def __init__(self, geometry: Geometry, time_step: float):
        self.geometry = geometry
        self.dt = time_step
        self.required_keys: list[str] = []
        # replaced by a seeded generator when the rule joins a CellularAutomaton
        self.rng = np.random.default_rng()

    def _cell_mask(self, grid: np.ndarray) -> np.ndarray:
        if grid.ndim == self.geometry.ndim + 1:
            return np.any(grid > 0, axis=-1)
        return grid > 0

    def _shift_for_offset(
        self, grid: np.ndarray, offset: tuple[int, ...]
    ) -> np.ndarray:
        """Return `result` with `result[i] == grid[i + offset]` (the neighbour at `offset`).

        Periodic axes wrap around; on non-periodic axes cells past the edge read as 0.
        """
        shifted = grid

        for axis, shift in enumerate(offset):
            if shift != 0 and axis in self.geometry.periodic_dims:
                shifted = np.roll(shifted, shift=-shift, axis=axis)

        non_periodic_axes = [
            axis
            for axis in range(self.geometry.ndim)
            if axis not in self.geometry.periodic_dims
        ]
        if non_periodic_axes:
            pad_width = [(0, 0)] * shifted.ndim
            for axis in non_periodic_axes:
                pad = abs(offset[axis])
                pad_width[axis] = (pad, pad)
            shifted = np.pad(shifted, pad_width, mode="constant")

            slices = []
            for axis, shift in enumerate(offset):
                if axis in self.geometry.periodic_dims:
                    slices.append(slice(None))
                else:
                    start = abs(shift) + shift
                    stop = start + grid.shape[axis]
                    slices.append(slice(start, stop))
            shifted = shifted[tuple(slices)]

        return shifted

    def _layer_count(self, grid: np.ndarray) -> np.ndarray:
        if grid.ndim == self.geometry.ndim + 1:
            return (grid > 0).sum(axis=-1)
        return grid.astype(np.int32, copy=False)

    def neighborhood_count(self, grid: np.ndarray):
        dim_count = self._layer_count(grid)
        # int32: a 3D Moore neighbourhood of a 6-layer key holds up to 156 particles
        count = np.zeros_like(dim_count, dtype=np.int32)
        for offset in self.geometry._offsets:
            count += self._shift_for_offset(
                dim_count, tuple(int(value) for value in offset)
            )
        return count

    def apply_state(self, state: State, ctx: StepContext | None = None) -> State:
        """Apply the rule to the whole state and return it. Every rule must implement this.

        `ctx` describes the current step; CellularAutomaton always passes it, and rules
        that don't need it may be called without it (e.g. in tests).
        """
        raise NotImplementedError()
