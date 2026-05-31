from __future__ import annotations

from random import random

import numpy as np

from libs.Geometry import Geometry
from libs2.State import State


class Rule:
    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.required_keys: list[str] = []

    def apply(self, neighbors, cell, coords=None):
        return cell
    
    def apply_state(self, state: State) -> State:
        '''if state is applayed to a whole array, not per cell'''
        raise NotImplementedError()





class GameOfLife3D(Rule):
    """vectorized 3DGoL based on arrays
    
    args:
    eb
    eh
    fb
    fh
    
    """

    def __init__(self, geometry, eb=5, eh=7, fb=6, fh=6):
        super().__init__(geometry)
        self.required_keys.extend(["alive"])
        self.eb = eb
        self.eh = eh
        self.fb = fb
        self.fh = fh

    def _shift_for_offset(self, grid: np.ndarray, offset: tuple[int, ...]) -> np.ndarray:
        shifted = grid

        for axis, shift in enumerate(offset):
            if shift != 0 and axis in self.geometry.periodic_dims:
                shifted = np.roll(shifted, shift=shift, axis=axis)

        non_periodic_axes = [axis for axis in range(self.geometry.ndim) if axis not in self.geometry.periodic_dims]
        if non_periodic_axes:
            pad_width = [(0, 0)] * shifted.ndim
            for axis in non_periodic_axes:
                pad_width[axis] = (1, 1)
            shifted = np.pad(shifted, pad_width, mode="constant")

            slices = []
            for axis, shift in enumerate(offset):
                if axis in self.geometry.periodic_dims:
                    slices.append(slice(None))
                else:
                    start = 1 + shift
                    stop = start + grid.shape[axis]
                    slices.append(slice(start, stop))
            shifted = shifted[tuple(slices)]

        return shifted

    def apply_state(self, state: State) -> State:
        # Use per-key accessors so reads/writes affect the underlying arrays
        alive_grid = state['alive'].astype(np.int8, copy=False)
        alive_counts = np.zeros_like(alive_grid, dtype=np.int16)

        for offset in self.geometry._offsets:
            alive_counts += self._shift_for_offset(alive_grid, tuple(int(value) for value in offset))

        current_alive = state['alive'] == 1
        survives = current_alive & (alive_counts >= self.eb) & (alive_counts <= self.eh)
        born = (~current_alive) & (alive_counts >= self.fb) & (alive_counts <= self.fh)

        updated_alive = np.zeros_like(state['alive'])
        updated_alive[survives | born] = 1
        state['alive'] = updated_alive
        return state


class BiofilmDetachment(Rule):
    ''''''
    def __init__(self, geometry, detachment_rate:float, scaling:float, target_key:str = 'biofilm'):
        super().__init__(geometry=geometry)
        self.detachment_probability = detachment_rate * scaling

        self.target_key = target_key
        self.required_keys.append(target_key)

    def apply_state(self, state: State) -> State:
        grid = state[self.target_key].copy() 
        chances = np.random.random(grid.shape)
        total_layers = grid.shape[-1]

        assert self.geometry.ndim == 3 #only works for 3D
        assert grid.dtype == bool #True/False layer defining the existence of biofilm
        
        for z in range(total_layers):
            dist_from_wall = z / max(1, total_layers - 1)
            
            chance_detachment = self.detachment_probability * (dist_from_wall ** 2)
            chance_detachment = min(chance_detachment, 1.0)

            detached = chances[..., z] < chance_detachment
            grid[..., z][detached] = False

        state[self.target_key] = grid

        return state


class GutDrift(Rule):
    '''periodically moves floating bacteria down the z-axis and flushes some of the biofilm down the z-axis
    the closer the biofilm is to the wall of the gut, the harder it is for it to get detached'''
    def __init__(self, geometry, drift_speed: int, target_key:str = 'floating_bacteria'):
        super().__init__(geometry=geometry)
        self.drift_speed = drift_speed
    
        self.target_key = self.target_key
        self.required_keys.append(target_key)

        assert self.geometry.ndim == 3 #only works for 3D

    def apply_state(self, state: State) -> State:
        grid = state[self.target_key].copy()

        pwidth = [(0,0)] * self.geometry.ndim

        z_axis = 2

        if z_axis in self.geometry.periodic_dims:    
            shifted = np.roll(grid, shift=self.drift_speed, axis=z_axis) #rolls element along axis z
        else: #if bacteria are being flushed out (nonperiodic)
            pwidth[z_axis] = (self.drift_speed, 0)
            padded = np.pad(grid, pwidth, mode='constant', constant_values=False)
            shifted = padded[..., :-self.drift_speed]


        state[self.target_key] = shifted
        return state
 