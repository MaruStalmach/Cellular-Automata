from libs.Rule import Rule
import numpy as np
from operator import itemgetter


class BacteriaGrowth(Rule):
    def __init__(
        self,
        geometry,
        time_step: float,
        bacteria_keys,
        substrate_key,
        Yca = 6.94e-6,
        base_prob = 0.02,
        time_scale = 4320000,
    ):
        super().__init__(geometry, time_step)
        self.b_keys = bacteria_keys
        self.sub_key = substrate_key

        self.Yca = Yca
        self.base_prob = base_prob
        self.time_scale = (
            time_scale  # ratio between larger time step and smaller time step
        )

        self.required_keys.extend([*bacteria_keys, substrate_key])

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

    def apply_state(self, state):
        substrate_grid = state[self.sub_key].copy()
        bacteria_grids = {key: state[key].copy() for key in self.b_keys}

        # which cells are occupied by bacteria?
        occupied_mask = np.zeros(state.shape, dtype=bool)
        for grid in bacteria_grids.values():
            occupied_mask |= self._cell_mask(grid)

        # counts amt of substrate particles in each cell
        substrate_count = self.neighborhood_count(substrate_grid).astype(
            np.int8, copy=False
        )

        # equations based on QUANTITATIVE CELLULAR AUTOMATON MODEL FOR BIOFILMS by pizarro

        growth_prob = np.clip(
            self.time_scale * self.Yca * self.base_prob * (substrate_count), 0.0, 1.0
        )

        # the code below prevents two diffrent bacteria from growing into the same empty cell
        proposals: list[tuple[float, str, tuple[int, ...], tuple[int, ...]]] = []
        for key, grid in bacteria_grids.items():
            key_mask = self._cell_mask(grid)
            dividing_cells = key_mask & (np.random.random(state.shape) < growth_prob)

            for parent_coords in map(tuple, np.argwhere(dividing_cells)):
                empty_targets = [
                    target
                    for target in self._neighbour_candidates(parent_coords)
                    if not occupied_mask[target]
                ]
                if not empty_targets:
                    continue

                target = empty_targets[int(np.random.randint(len(empty_targets)))]
                proposals.append(
                    (float(np.random.random()), key, parent_coords, target)
                )

        proposals.sort(key=itemgetter(0), reverse=True)
        claimed_targets: set[tuple[int, ...]] = set()

        for _, key, parent_coords, target_coords in proposals:
            # dont grow if spot taken
            if target_coords in claimed_targets or occupied_mask[target_coords]:
                continue

            bacteria_grids[key][target_coords] = state[key][parent_coords]
            occupied_mask[target_coords] = True
            claimed_targets.add(target_coords)

        for key, grid in bacteria_grids.items():
            state[key] = grid

        return state
