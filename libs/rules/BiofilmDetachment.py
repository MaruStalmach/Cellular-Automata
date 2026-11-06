import numpy as np

from libs.Rule import Rule
from libs.State import State


class BiofilmDetachment(Rule):
    """the closer the biofilm is to the wall of the gut, the harder it is for it to get detached
    moves detached cells from biofilm layer to detached bacteria layer"""

    def __init__(
        self, 
        geometry, 
        time_step: float, 
        detachment_rate: float, 
        scaling: float,
        species_keys: list | None = None,
        attachment_probability: float = 1.0,
    ):
        super().__init__(geometry, time_step)
        
        self.detachment_probability = detachment_rate * scaling
        self.target_keys = ["biofilm", "floating_bacteria"]
        self.required_keys.extend(self.target_keys)

        self.species_keys = species_keys or []
        self.attachment_probability = attachment_probability
        self.required_keys.extend(self.species_keys)

        self.y_axis = geometry.axes.index("y")
        y_length = geometry.size[self.y_axis]
        y_distances = np.arange(y_length, dtype=np.float32) / max(1, y_length - 1)
        self.y_probabilities = np.clip(
            self.detachment_probability * (y_distances ** 2), 0.0, 1.0
        )


    def apply_state(self, state: State) -> State:
        assert self.geometry.ndim == 3

        biofilm_grid = state["biofilm"]
        floating_bact_grid = state["floating_bacteria"]

        active_coords = np.nonzero(biofilm_grid)
        num_active = len(active_coords[0])

        if num_active == 0:
            return state

        if num_active > 0:
            active_y_indices = active_coords[1]
            active_probs = self.y_probabilities[active_y_indices]
 
            randoms = np.random.random(num_active)
            detach_mask = randoms < active_probs
 
            detached_coords = (
                active_coords[0][detach_mask],
                active_coords[1][detach_mask],
                active_coords[2][detach_mask],
            )
 
            biofilm_grid[detached_coords] = False
            floating_bact_grid[detached_coords] = True

        if self.species_keys and self.attachment_probability > 0.0:
            biofilm_bool = biofilm_grid.astype(bool)
 
            occupied = np.zeros(self.geometry.size, dtype=bool)
            for sp in self.species_keys:
                occupied |= state[sp].astype(bool)
 
            below_is_biofilm = np.zeros_like(biofilm_bool)
            below_is_biofilm[:, 1:, :] = biofilm_bool[:, :-1, :]
 
            attach_candidates = occupied & ~biofilm_bool & below_is_biofilm
 
            cand_coords = np.nonzero(attach_candidates)
            num_cand = len(cand_coords[0])
 
            if num_cand > 0:
                if self.attachment_probability >= 1.0:
                    attach_mask = np.ones(num_cand, dtype=bool)
                else:
                    attach_mask = (
                        np.random.random(num_cand) < self.attachment_probability
                    )
 
                attached_coords = tuple(c[attach_mask] for c in cand_coords)
                biofilm_grid[attached_coords] = True
                floating_bact_grid[attached_coords] = False

        
        state["biofilm"] = biofilm_grid
        state["floating_bacteria"] = floating_bact_grid

        return state
