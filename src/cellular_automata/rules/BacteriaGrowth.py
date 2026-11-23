import numpy as np

from cellular_automata.registry import register_rule
from cellular_automata.Rule import Rule, StepContext


@register_rule
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

    def apply_state(self, state, ctx: StepContext | None = None):
        substrate_grid = state[self.sub_key].copy()
        bacteria_grids = {key: state[key].copy() for key in self.b_keys}

        # which cells are occupied by bacteria?
        occupied_mask = np.zeros(state.shape, dtype=bool)
        for grid in bacteria_grids.values():
            occupied_mask |= self._cell_mask(grid)

        # counts amt of substrate particles in each cell
        substrate_count = self.neighborhood_count(substrate_grid)

        # equations based on QUANTITATIVE CELLULAR AUTOMATON MODEL FOR BIOFILMS by pizarro

        growth_prob = np.clip(
            self.time_scale * self.Yca * self.base_prob * (substrate_count), 0.0, 1.0
        )

        # each dividing cell picks an empty neighbour uniformly; a cell picked by several
        # parents (of any species) goes to one of them, chosen uniformly
        keys, parents, targets = [], [], []
        for key_index, grid in enumerate(bacteria_grids.values()):
            key_mask = self._cell_mask(grid)
            dividing_cells = key_mask & (self.rng.random(state.shape) < growth_prob)
            key_parents = np.argwhere(dividing_cells)

            neighbours, valid = self._neighbour_coords(key_parents)
            valid &= ~occupied_mask[tuple(np.moveaxis(neighbours, -1, 0))]
            column, has_target = self._pick_uniform(valid)
            chosen = neighbours[np.arange(len(key_parents)), column][has_target]

            keys.append(np.full(len(chosen), key_index))
            parents.append(key_parents[has_target])
            targets.append(chosen)

        keys_all = np.concatenate(keys)
        parents_all = np.concatenate(parents)
        targets_all = np.concatenate(targets)
        winners = self._first_claims(
            np.ravel_multi_index(tuple(targets_all.T), state.shape)
        )

        for key_index, key in enumerate(bacteria_grids):
            won = winners[keys_all[winners] == key_index]
            parent_idx = tuple(parents_all[won].T)
            target_idx = tuple(targets_all[won].T)
            bacteria_grids[key][target_idx] = state[key][parent_idx]

        for key, grid in bacteria_grids.items():
            state[key] = grid

        return state
