from .. import Rule, State
import numpy as np

class BiofilmDetachment(Rule):
    """the closer the biofilm is to the wall of the gut, the harder it is for it to get detached
    moves detached cells from biofilm layer to detached bacteria layer"""

    def __init__(self, geometry, detachment_rate: float, scaling: float):
        super().__init__(geometry=geometry)
        self.detachment_probability = detachment_rate * scaling

        self.target_keys = ["biofilm", "floating_bacteria"]
        self.required_keys.extend(self.target_keys)

    def apply_state(self, state: State) -> State:

        assert self.geometry.ndim == 3  # only works for 3D

        biofilm_grid, floating_bact_grid = state["biofilm"], state["floating_bacteria"]
        total_layers = biofilm_grid.shape[-1]

        z_indices = np.arange(total_layers)
        # chooses the chance of detachment based on the distance from the wall of the gut
        # TODO: check the distance to the wall at both sides
        chance_detachment = np.clip(
            self.detachment_probability * (z_indices**2), 0.0, 1.0
        )

        chances = np.random.random(biofilm_grid.shape)

        # checks for biofilm on square and checks the prob of detachment
        detached_mask = biofilm_grid & (chances < chance_detachment)
        biofilm_grid[detached_mask] = False

        floating_bact_grid |= (
            detached_mask  # bitwise or adds detached cells to floating bact layer
        )

        state["biofilm"] = biofilm_grid
        state["floating_bacteria"] = floating_bact_grid

        return state