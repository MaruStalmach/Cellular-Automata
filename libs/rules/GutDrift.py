from libs.Rule import Rule
from libs.State import State
import numpy as np

class GutDrift(Rule):
    """periodically moves floating bacteria down the z-axis and flushes some of the biofilm down the z-axis"""

    def __init__(
        self, geometry, drift_speed: int, target_key: str = "floating_bacteria"
    ):
        super().__init__(geometry=geometry)
        self.drift_speed = drift_speed

        #
        # target key for the layer to be drifted
        self.target_key = target_key
        self.required_keys.append(target_key)

        assert self.geometry.ndim == 3  # only works for 3D

    def apply_state(self, state: State) -> State:
        if self.drift_speed == 0:
            return state

        grid = state[self.target_key]
        z_axis = self.geometry.ndim - 1

        if z_axis in self.geometry.periodic_dims:
            shifted = np.roll(
                grid, shift=self.drift_speed, axis=z_axis
            )  # rolls element along axis z
        else:  # if bacteria are being flushed out (nonperiodic)
            pwidth = [(0, 0)] * self.geometry.ndim

            pwidth[z_axis] = (self.drift_speed, 0)
            padded = np.pad(grid, pwidth, mode="constant", constant_values=0)

            indexer = [slice(None)] * grid.ndim
            indexer[z_axis] = slice(0, grid.shape[z_axis])
            shifted = padded[tuple(indexer)]

        state[self.target_key] = shifted
        return state