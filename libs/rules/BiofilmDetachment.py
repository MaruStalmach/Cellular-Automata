from libs.Rule import Rule
from libs.State import State
import numpy as np


class BiofilmDetachment(Rule):
    """the closer the biofilm is to the wall of the gut, the harder it is for it to get detached
    moves detached cells from biofilm layer to detached bacteria layer"""

    def __init__(
        self, geometry, time_step: float, biofilm_key : str, detachment_rate: float, scaling: float
    ):
        super().__init__(geometry, time_step)
        self.detachment_probability = detachment_rate * scaling

        self.target_key = biofilm_key
        self.required_keys.extend(self.target_key)

    def apply_state(self, state: State) -> State:

        assert self.geometry.ndim == 3  # only works for 3D

        biofilm_grid = state[self.target_key].copy()
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
        biofilm_grid[detached_mask] = 0


        state["biofilm"] = biofilm_grid

        return state
