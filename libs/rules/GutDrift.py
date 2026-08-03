import numpy as np

from libs.Rule import Rule
from libs.State import State


class GutDrift(Rule):
    """periodically moves floating bacteria down the z-axis and flushes some of the biofilm down the z-axis"""

    def __init__(
        self,
        geometry,
        time_step: float,
        drift_speed: int,
        species_keys: list,
        target_key: str = "floating_bacteria",
    ):
        super().__init__(geometry, time_step)
        self.drift_speed = drift_speed
        self.species_keys = species_keys
        self.target_key = target_key # target key for the layer to be drifted

        self.required_keys.append(target_key)
        self.required_keys.extend(species_keys)

        assert self.geometry.ndim == 3  # only works for 3D

    def _shift_z(self, grid: np.ndarray) -> np.ndarray:
        axis = self.geometry.ndim - 1

        if axis in self.geometry.periodic_dims:
            return np.roll(grid, shift=self.drift_speed, axis=axis)

        shifted = np.zeros_like(grid)
        if self.drift_speed > 0:
            src = [slice(None)] * grid.ndim
            dst = [slice(None)] * grid.ndim
            src[axis] = slice(0, grid.shape[axis] - self.drift_speed)
            dst[axis] = slice(self.drift_speed, grid.shape[axis])
        else:
            shift = abs(self.drift_speed)
            src = [slice(None)] * grid.ndim
            dst = [slice(None)] * grid.ndim
            src[axis] = slice(shift, grid.shape[axis])
            dst[axis] = slice(0, grid.shape[axis] - shift)

        shifted[tuple(dst)] = grid[tuple(src)]
        return shifted



    def apply_state(self, state: State) -> State:
        if self.drift_speed == 0:
            return state

        #floating bacteria per key
        floating_mask = state[self.target_key].astype(bool)

        #attached bacteria per key
        attached_mask = np.zeros(self.geometry.size, dtype=bool)
        for sp in self.species_keys:
            attached_mask |= (state[sp].astype(bool) & ~floating_mask)

        shifted_floating_mask = self._shift_z(floating_mask.astype(np.uint8)).astype(bool)
        surviving_floating_mask = shifted_floating_mask & ~attached_mask

        for species_key in self.species_keys:
            grid = state[species_key]

            mask = floating_mask[..., None] if grid.ndim == floating_mask.ndim + 1 else floating_mask
            survive_mask = (
                surviving_floating_mask[..., None]
                if grid.ndim == surviving_floating_mask.ndim + 1
                else surviving_floating_mask
            )

            floating_species = np.where(mask, grid, 0)
            attached_species = np.where(mask, 0, grid)

            shifted_floating_species = self._shift_z(floating_species)
            surviving_shifted_species = np.where(survive_mask, shifted_floating_species, 0)

            state[species_key] = np.maximum(attached_species, surviving_shifted_species)

        state[self.target_key] = surviving_floating_mask.astype(state[self.target_key].dtype)
        
        return state
        
