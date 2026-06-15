from __future__ import annotations

from random import random

import numpy as np

from libs.Geometry import Geometry
from libs.State import State


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

        assert self.geometry.ndim == 3  #only works for 3D

        original_dtype = grid.dtype
        mask = grid.astype(bool)

        #probability ~ detachment_probability * (z ** 2)
        for z in range(total_layers):
            chance_detachment = self.detachment_probability * (z ** 2)
            chance_detachment = min(chance_detachment, 1.0)

            detached = chances[..., z] < chance_detachment
            mask[..., z][detached] = False

        #write back preserving original dtype
        if original_dtype == bool:
            grid = mask
        else:
            grid[...] = mask.astype(original_dtype)

        state[self.target_key] = grid

        return state


class GutDrift(Rule):
    '''periodically moves floating bacteria down the z-axis and flushes some of the biofilm down the z-axis
    the closer the biofilm is to the wall of the gut, the harder it is for it to get detached'''
    def __init__(self, geometry, drift_speed: int, target_key: str = 'floating_bacteria', detachment_rule: BiofilmDetachment | None = None):
        super().__init__(geometry=geometry)
        self.drift_speed = drift_speed

        # 
        #target key for the layer to be drifted
        self.target_key = target_key
        self.required_keys.append(target_key)

        #optional detachment rule for parity with object-based implementation
        self.detachment_rule = detachment_rule

        assert self.geometry.ndim == 3  #only works for 3D

    def apply_state(self, state: State) -> State:
        grid = state[self.target_key].copy()

        pwidth = [(0, 0)] * self.geometry.ndim

        z_axis = self.geometry.ndim - 1

        if z_axis in self.geometry.periodic_dims:
            shifted = np.roll(grid, shift=self.drift_speed, axis=z_axis)  #rolls element along axis z
        else:  #if bacteria are being flushed out (nonperiodic)
            pwidth[z_axis] = (self.drift_speed, 0)
            padded = np.pad(grid, pwidth, mode='constant', constant_values=0)

            indexer = [slice(None)] * grid.ndim
            indexer[z_axis] = slice(0, grid.shape[z_axis])
            shifted = padded[tuple(indexer)]

        state[self.target_key] = shifted
        return state
    
    
class Diffusion(Rule):
    '''Random walk diffusion based on "Quantitative Cellular Automaton Model For Biofilms" by Pizarro G.'''
    
    def __init__(self, geometry, a, p, target_key='substrate'):
        super().__init__(geometry)
        
        self.required_keys = [f'{target_key}{x}' for x in range(6)]
        p3 = (1-4*p)/(a+1)
        self.p = [a*p3,
                  p,
                  p,
                  p3,
                  p,
                  p]
        
    def apply_state(self, state) -> State:
    
        old_layers = np.empty(shape=state.shape+(6,))
        for i,key in enumerate(self.required_keys):
            old_layers[...,i] = state[key]
        new_layers = np.zeros_like(old_layers)
        
        #each layer corresponds to the dircetion a particle is moving, list p contains probablities of the next direction
        # p0 - no change, p3 - 180 deg turn, p1,p2,p4,p5 - 90 degree turns
        
        # layer 0 - +z
        # l1      - +y
        # l2      - +x
        # l3      - -z
        # l4      - -y
        # l5      - -x
        
        rolls = np.random.choice([0,1,2,3,4,5], size = old_layers.shape, p=self.p)
        
        h = np.stack([np.full(shape=state.shape, fill_value=x,dtype=np.uint8) for x in range(6)], axis=-1)
        
        rolls = ((rolls + h) %6)
        #set empty cells as invalid value
        rolls[old_layers==0]=6
        
        #does check for collisions
        for i in range(6):
            r2 = np.zeros_like(rolls)
            r2[rolls==i]=1
            new_layers[...,i]=np.add.reduce(r2,axis=-1)
            #TODO:deal with collisions
            
            
        #doesnt check for collisions
        for i in range(6):
            new_layers[np.any(rolls==i,axis=-1),i]=1
        
        
        
        
        
        
        
 