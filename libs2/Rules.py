from __future__ import annotations

from random import random

import numpy as np

from libs.Geometry import Geometry
from libs2.Cells import State


class Rule:
    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.required_keys: list[str] = []

    def apply(self, neighbors, cell, coords=None):
        return cell





class GameOfLife3D(Rule):
    """vectorized 3D Game of Life rule for the array backend"""

    def __init__(self, geometry, eb=5, eh=7, fb=6, fh=6):
        super().__init__(geometry)
        self.required_keys.extend(["alive"])
        self.eb = eb
        self.eh = eh
        self.fb = fb
        self.fh = fh

    def _shift_for_offset(self, grid: np.ndarray, offset: tuple[int, ...]) -> np.ndarray:
        shifted = grid

        for axis, shift in enumerate(offset):
            if shift != 0 and axis in self.geometry.periodic_dims:
                shifted = np.roll(shifted, shift=shift, axis=axis)

        non_periodic_axes = [axis for axis in range(self.geometry.ndim) if axis not in self.geometry.periodic_dims]
        if non_periodic_axes:
            pad_width = [(0, 0)] * shifted.ndim
            for axis in non_periodic_axes:
                pad_width[axis] = (1, 1)
            shifted = np.pad(shifted, pad_width, mode="constant")

            slices = []
            for axis, shift in enumerate(offset):
                if axis in self.geometry.periodic_dims:
                    slices.append(slice(None))
                else:
                    start = 1 + shift
                    stop = start + grid.shape[axis]
                    slices.append(slice(start, stop))
            shifted = shifted[tuple(slices)]

        return shifted

    def apply_state(self, state: State) -> State:
        alive_idx = state.key_to_index["alive"]
        alive_grid = state.data[..., alive_idx].astype(np.int8, copy=False)
        alive_counts = np.zeros_like(alive_grid, dtype=np.int16)

        for offset in self.geometry._offsets:
            alive_counts += self._shift_for_offset(alive_grid, tuple(int(value) for value in offset))

        current_alive = state.data[..., alive_idx] == 1
        survives = current_alive & (alive_counts >= self.eb) & (alive_counts <= self.eh)
        born = (~current_alive) & (alive_counts >= self.fb) & (alive_counts <= self.fh)

        updated_alive = np.zeros_like(state.data[..., alive_idx])
        updated_alive[survives | born] = 1
        state.data[..., alive_idx] = updated_alive
        return state


class BiofilmDetachment(Rule):
    def __init__(self, geometry, detachment_rate, scaling):
        super().__init__(geometry=geometry)
        self.detachment_probability = detachment_rate * scaling

    def would_detach(self, coords) -> bool:
        if coords is not None:
            coord_z = coords[-1]
        else:
            coord_z = 0

        distance_sq = coord_z**2
        probability = min(self.detachment_probability * distance_sq, 1.0)

        return random() < probability

    def apply(self, neighbors, cell, coords=None):
        if self.would_detach(coords):
            return np.zeros_like(cell)
        return cell


class GutDrift(Rule):
    def __init__(self, geometry, drift_speed: int, detachment_rule: BiofilmDetachment | None = None):
        super().__init__(geometry=geometry)
        self.drift_speed = drift_speed
        self.detachment_rule = detachment_rule
        assert self.geometry.ndim == 3

    def apply(self, neighbors, cell, coords=None):
        return cell
