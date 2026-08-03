from libs.Rule import Rule
import numpy as np
from operator import itemgetter


class Utilization(Rule):
    def __init__(
        self,
        geometry,
        time_step: float,
        bacteria_keys: list[str],
        substrate_key: str,
        substitute_constant: float = 0.37,
        Ks: float = 10,
        Sb: float = 15,
    ):
        super().__init__(geometry, time_step)
        self.b_keys = bacteria_keys
        self.sub_key = substrate_key

        self.required_keys.extend([*bacteria_keys, substrate_key])

        self.sub_const = substitute_constant
        self.Ks = Ks
        self.Sb = Sb

        self.mask = np.zeros(shape=geometry.size, dtype=bool)

    def _shift_spatial(self, grid: np.ndarray, offset: tuple[int, ...]) -> np.ndarray:
        shifted = grid

        for axis, shift in enumerate(offset):
            if shift != 0 and axis in self.geometry.periodic_dims:
                shifted = np.roll(shifted, shift=shift, axis=axis)

        non_periodic_axes = [
            axis
            for axis in range(self.geometry.ndim)
            if axis not in self.geometry.periodic_dims
        ]
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

    def _neighbour_candidates(self, coords: tuple[int, ...]) -> list[tuple[int, ...]]:
        candidates: list[tuple[int, ...]] = []
        for offset in self.geometry._offsets:
            target = []
            valid = True
            for axis, (coord, delta) in enumerate(zip(coords, offset, strict=True)):
                value = coord + int(delta)
                if axis in self.geometry.periodic_dims:
                    value %= self.geometry.size[axis]
                elif value < 0 or value >= self.geometry.size[axis]:
                    valid = False
                    break
                target.append(value)

            if valid:
                candidates.append(tuple(target))

        return candidates

    def _substrate_candidates_for_cell(
        self,
        substrate_grid: np.ndarray,
        coords: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], int]]:
        candidates: list[tuple[tuple[int, ...], int]] = []
        for offset in self.geometry._offsets:
            neighbour = []
            valid = True
            for axis, (coord, delta) in enumerate(zip(coords, offset, strict=True)):
                value = coord + int(delta)
                if axis in self.geometry.periodic_dims:
                    value %= self.geometry.size[axis]
                elif value < 0 or value >= self.geometry.size[axis]:
                    valid = False
                    break
                neighbour.append(value)

            if not valid:
                continue

            neighbour_coords = tuple(neighbour)
            if substrate_grid.ndim == self.geometry.ndim + 1:
                active_layers = np.flatnonzero(substrate_grid[neighbour_coords] > 0)
                for layer in active_layers:
                    candidates.append((neighbour_coords, int(layer)))
            elif substrate_grid[neighbour_coords] > 0:
                candidates.append((neighbour_coords, 0))

        return candidates

    def _consume_substrate_particle(
        self, substrate_grid: np.ndarray, coords: tuple[int, ...]
    ) -> None:
        if substrate_grid.ndim == self.geometry.ndim + 1:
            substrate_grid[coords] = 0
            return

        substrate_grid[coords[:-1]] = 0

    def apply(self, neighbors, cell, coords=None):
        return super().apply(neighbors, cell, coords)

    def apply_state(self, state):
        # get data from simulation
        substrate_grid = state[self.sub_key].copy()
        bacteria_grids = {key: state[key].copy() for key in self.b_keys}

        # which cells are occupied by bacteria?
        self.mask.fill(0)
        for grid in bacteria_grids.values():
            self.mask |= self._cell_mask(grid)

        # counts amt of substrate particles in each cell neighborhood
        substrate_count = self.neighborhood_count(substrate_grid).astype(
            np.int8, copy=False
        )
        ###################################################################################################
        # equations based on 'QUANTITATIVE CELLULAR AUTOMATON MODEL FOR BIOFILMS' by pizarro
        # reusing the same array for calculations
        utilization_prob = self.sub_const * (
            (self.Sb * substrate_count / 27)
            / (self.Ks + (self.Sb * substrate_count / 27))
        )
        # p=r *dt=q*(S/(10+S))*40*dt
        # S=15*neighbors/max
        # r=q*(S/(10+S))*40
        utilization_prob = np.clip(utilization_prob, 0.0, 1.0)
        utilization_prob = np.where(self.mask, utilization_prob, 0.0)
        # utilization prob is the probabilty that a
        # a bacteria cell uses/eats a substrate particle

        # roll which bacteria cells eat a substrate particle
        consumption_draw = np.random.random(state.shape)
        consumed_cells = self.mask & (consumption_draw < utilization_prob)

        # a substrate particle should be claimed by one bacteria only (no double spending)
        # this lets every bacteria cell claim an uneaten particle and remove it from the 'plate',
        # so other bacteria dont eat the particle again
        substrate_claims: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for coords in map(tuple, np.argwhere(consumed_cells)):
            candidates = self._substrate_candidates_for_cell(substrate_grid, coords)
            if not candidates:
                continue

            target_cell, target_layer = candidates[
                int(np.random.randint(len(candidates)))
            ]
            substrate_claims.append(
                (float(np.random.random()), coords, target_cell + (target_layer,))
            )

        substrate_claims.sort(key=itemgetter(0), reverse=True)
        claimed_substrate_targets: set[tuple[int, ...]] = set()

        for _, _, substrate_coords in substrate_claims:
            if substrate_coords in claimed_substrate_targets:
                continue

            self._consume_substrate_particle(substrate_grid, substrate_coords)
            claimed_substrate_targets.add(substrate_coords)

        state[self.sub_key] = substrate_grid

        return state
