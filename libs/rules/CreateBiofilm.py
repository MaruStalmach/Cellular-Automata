import numpy as np

from libs.Rule import Rule
from libs.State import State


class CreateBiofilm(Rule):
	def __init__(
		self,
		geometry,
		time_step: float,
		biofilm_key: str,
		bacteria_keys: list[str],
	):
		super().__init__(geometry, time_step)
		self.biofilm_key = biofilm_key
		self.bacteria_keys = list(bacteria_keys)

		self.required_keys.extend([biofilm_key, *self.bacteria_keys])

	def _bacteria_presence(self, state: State) -> np.ndarray:
		presence = np.zeros(state.shape, dtype=bool)
		for key in self.bacteria_keys:
			presence |= self._cell_mask(state[key])
		return presence

	def apply_state(self, state: State) -> State:
		biofilm_grid = state[self.biofilm_key].copy()
		bacteria_presence = self._bacteria_presence(state)

		if biofilm_grid.ndim == self.geometry.ndim + 1:
			updated_biofilm = np.logical_or(biofilm_grid, bacteria_presence[..., None])
		else:
			updated_biofilm = np.logical_or(biofilm_grid, bacteria_presence)

		state[self.biofilm_key] = updated_biofilm.astype(biofilm_grid.dtype)
		return state
