import numpy as np

from cellular_automata.registry import register_rule
from cellular_automata.Rule import Rule, StepContext


@register_rule
class Utilization(Rule):
    # consuming cells handled per batch, to bound memory on large grids
    CHUNK = 65536

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

    def apply_state(self, state, ctx: StepContext | None = None):
        # get data from simulation
        substrate_grid = state[self.sub_key].copy()
        bacteria_grids = {key: state[key].copy() for key in self.b_keys}

        # which cells are occupied by bacteria?
        self.mask.fill(0)
        for grid in bacteria_grids.values():
            self.mask |= self._cell_mask(grid)

        # counts amt of substrate particles in each cell neighborhood
        substrate_count = self.neighborhood_count(substrate_grid)
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
        consumption_draw = self.rng.random(state.shape)
        consumed_cells = self.mask & (consumption_draw < utilization_prob)

        # each consuming cell picks one substrate particle next to it, uniformly over the
        # particles (a neighbour with 3 particles is 3 times as likely); a particle picked
        # by several cells is eaten once, and the others go without this time
        layered = substrate_grid.ndim == self.geometry.ndim + 1
        has_particle = substrate_grid > 0
        if not layered:
            has_particle = has_particle[..., None]
        layers = has_particle.shape[-1]

        consumers = np.argwhere(consumed_cells)
        eaten = []
        for start in range(0, len(consumers), self.CHUNK):
            chunk = consumers[start:start + self.CHUNK]
            neighbours, valid = self._neighbour_coords(chunk)
            slots = has_particle[tuple(np.moveaxis(neighbours, -1, 0))] & valid[..., None]
            column, has_target = self._pick_uniform(slots.reshape(len(chunk), -1))
            offset_index, layer = np.divmod(column[has_target], layers)
            cells = neighbours[np.flatnonzero(has_target), offset_index]
            eaten.append(np.column_stack([cells, layer]))

        if eaten:
            eaten_slots = np.concatenate(eaten)
            if layered:
                substrate_grid[tuple(eaten_slots.T)] = 0
            else:
                substrate_grid[tuple(eaten_slots[:, :-1].T)] = 0

        state[self.sub_key] = substrate_grid

        return state
