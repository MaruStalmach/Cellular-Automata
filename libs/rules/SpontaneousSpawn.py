import numpy as np

from libs.Rule import Rule
from libs.State import State


class SpontaneousSpawn(Rule):
    """allows species to appear on empty grid cells independent of neighbours,
    so a species that starts at 0.0 abundance (or has no existing individuals
    nearby) still has a way to enter the simulation.

    each empty cell gets an independent Bernoulli(spawn_probabilities[sp]) draw
    per species per step. species are processed in random order each call so
    that repeated runs don't systematically favor whichever species happens
    to be listed first when two species would otherwise claim the same cell.
    """

    def __init__(
        self,
        geometry,
        time_step: float,
        species_keys: list,
        spawn_probabilities: dict,
    ):
        super().__init__(geometry, time_step)
        self.species_keys = species_keys
        # species with no configured probability simply never spontaneously spawn
        self.spawn_probabilities = {
            sp: float(spawn_probabilities.get(sp, 0.0)) for sp in species_keys
        }

        self.required_keys.extend(species_keys)

    def apply_state(self, state: State) -> State:
        occupied = np.zeros(state.shape, dtype=bool)
        for sp in self.species_keys:
            occupied |= state[sp].astype(bool)
        empty = ~occupied

        order = list(self.species_keys)
        np.random.shuffle(order)

        for sp in order:
            prob = self.spawn_probabilities[sp]
            if prob <= 0.0 or not empty.any():
                continue

            spawn_mask = empty & (np.random.random(state.shape) < prob)
            if spawn_mask.any():
                state[sp] = np.where(
                    spawn_mask.reshape(state[sp].shape), 1, state[sp]
                ).astype(state[sp].dtype)
                empty &= ~spawn_mask  # claimed cells can't also spawn a different species this step

        return state