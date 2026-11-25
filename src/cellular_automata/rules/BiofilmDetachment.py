import numpy as np

from cellular_automata.registry import register_rule
from cellular_automata.Rule import Rule, StepContext, require_3d
from cellular_automata.State import State


@register_rule
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
        require_3d(type(self).__name__, geometry)

        self.detachment_probability = detachment_rate * scaling
        self.target_keys = ["biofilm", "floating_bacteria"]
        self.required_keys.extend(self.target_keys)

        self.species_keys = species_keys or []
        self.attachment_probability = attachment_probability
        self.required_keys.extend(self.species_keys)

        self.y_axis = geometry.axes.index("y")
        y_length = geometry.size[self.y_axis]
        y_distances = np.arange(y_length, dtype=np.float32) / max(1, y_length - 1)
        self.y_probabilities = np.clip(self.detachment_probability * (y_distances**2), 0.0, 1.0)

    def apply_state(self, state: State, ctx: StepContext | None = None) -> State:
        biofilm_grid = state["biofilm"]
        floating_bact_grid = state["floating_bacteria"]
        biofilm_at_start = biofilm_grid.astype(bool)

        detached = self._detach(biofilm_at_start)
        biofilm_grid[detached] = False
        floating_bact_grid[detached] = True

        if self.species_keys and self.attachment_probability > 0.0:
            attached = self._attach(state, biofilm_at_start, detached)
            biofilm_grid[attached] = True
            floating_bact_grid[attached] = False

        state["biofilm"] = biofilm_grid
        state["floating_bacteria"] = floating_bact_grid

        return state

    def _detach(self, biofilm: np.ndarray) -> np.ndarray:
        detached = np.zeros_like(biofilm)
        active_coords = np.nonzero(biofilm)
        active_probs = self.y_probabilities[active_coords[self.y_axis]]
        detach_mask = self.rng.random(len(active_probs)) < active_probs
        detached[tuple(c[detach_mask] for c in active_coords)] = True
        return detached

    def _attach(
        self, state: State, biofilm_at_start: np.ndarray, detached: np.ndarray
    ) -> np.ndarray:
        # attachment sees the biofilm as it was at the start of the step, so a cell that
        # detached in this call can't re-attach until the next one
        occupied = np.zeros(self.geometry.size, dtype=bool)
        for sp in self.species_keys:
            occupied |= self._cell_mask(state[sp])

        below_is_biofilm = np.zeros_like(biofilm_at_start)
        below_is_biofilm[:, 1:, :] = biofilm_at_start[:, :-1, :]

        candidates = occupied & ~biofilm_at_start & ~detached & below_is_biofilm
        if self.attachment_probability < 1.0:
            cand_coords = np.nonzero(candidates)
            rejected = self.rng.random(len(cand_coords[0])) >= self.attachment_probability
            candidates[tuple(c[rejected] for c in cand_coords)] = False
        return candidates
