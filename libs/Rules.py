from __future__ import annotations

import numpy as np
from json import load, JSONDecodeError

from libs.Geometry import Geometry
from libs.State import State

from typing import Dict
from itertools import permutations


class Rule:
    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.required_keys: list[str] = []

    def apply(self, neighbors, cell, coords=None):
        return cell
    
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
    '''the closer the biofilm is to the wall of the gut, the harder it is for it to get detached
    moves detached cells from biofilm layer to detached bacteria layer'''
    def __init__(self, geometry, detachment_rate:float, scaling:float):
        super().__init__(geometry=geometry)
        self.detachment_probability = detachment_rate * scaling

        self.target_keys = ['biofilm', 'floating_bacteria']
        self.required_keys.extend(self.target_keys)

    def apply_state(self, state: State) -> State:

        assert self.geometry.ndim == 3  #only works for 3D

        biofilm_grid, floating_bact_grid = state['biofilm'], state['floating_bacteria']
        total_layers = biofilm_grid.shape[-1]

        z_indices = np.arange(total_layers)
        #chooses the chance of detachment based on the distance from the wall of the gut
        #TODO: check the distance to the wall at both sides
        chance_detachment = np.clip(self.detachment_probability * (z_indices ** 2), 0.0, 1.0)
     
        chances = np.random.random(biofilm_grid.shape)

        #checks for biofilm on square and checks the prob of detachment
        detached_mask = biofilm_grid & (chances < chance_detachment)
        biofilm_grid[detached_mask] = False

        floating_bact_grid |= detached_mask #bitwise or adds detached cells to floating bact layer

        state['biofilm'] = biofilm_grid
        state['floating_bacteria'] = floating_bact_grid

        return state


class GutDrift(Rule):
    '''periodically moves floating bacteria down the z-axis and flushes some of the biofilm down the z-axis'''
    def __init__(self, geometry, drift_speed: int, target_key: str = 'floating_bacteria'):
        super().__init__(geometry=geometry)
        self.drift_speed = drift_speed

        # 
        #target key for the layer to be drifted
        self.target_key = target_key
        self.required_keys.append(target_key)


        assert self.geometry.ndim == 3  #only works for 3D

    def apply_state(self, state: State) -> State:
        if self.drift_speed == 0:
            return state
        

        grid = state[self.target_key]
        z_axis = self.geometry.ndim - 1

        if z_axis in self.geometry.periodic_dims:
            shifted = np.roll(grid, shift=self.drift_speed, axis=z_axis)  #rolls element along axis z
        else:  #if bacteria are being flushed out (nonperiodic)
            pwidth = [(0, 0)] * self.geometry.ndim

            pwidth[z_axis] = (self.drift_speed, 0)
            padded = np.pad(grid, pwidth, mode='constant', constant_values=0)

            indexer = [slice(None)] * grid.ndim
            indexer[z_axis] = slice(0, grid.shape[z_axis])
            shifted = padded[tuple(indexer)]

        state[self.target_key] = shifted
        return state
    
    
class Diffusion(Rule):
    '''Random walk diffusion based on "Quantitative Cellular Automaton Model For Biofilms" by Pizarro G.
    
    arguments:
    a - defined as p0/p3, where p0 is probability that a particle wont change direction and p3 is probability that a particle will make a 180 deg turn
    p - probability for a particle to do a 90 deg turn. p=p1=p2=p4=p5 because of symmetry. 
    
    '''
    
    def __init__(self, geometry : Geometry, a : float, p : float, target_key : str ='substrate'):
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
    
        layers = np.empty(shape=state.shape+(6,))
        for i,key in enumerate(self.required_keys):
            layers[...,i] = state[key]
        # layers = np.zeros_like(layers)
        
        #each layer corresponds to the dircetion a particle is moving, list p contains probablities of the next direction
        # p0 - no change, p3 - 180 deg turn, p1,p2,p4,p5 - 90 degree turns
        
        # layer 0 - +z
        # l1      - +y
        # l2      - +x
        # l3      - -z
        # l4      - -y
        # l5      - -x
        
        rolls = np.random.choice([0,1,2,3,4,5], size = layers.shape, p=self.p)
        
        h = np.stack([np.full(shape=state.shape, fill_value=x,dtype=np.uint8) for x in range(6)], axis=-1)
        
        
        rolls = ((rolls + h) %6)
        #set empty cells as invalid value
        rolls[layers==0]=6
        
        #reset layers to zeros
        layers[...]=0
        
        #does check for collisions
        for i in range(6):
            r2 = np.zeros_like(rolls)
            r2[rolls==i]=1
            layers[...,i]=np.add.reduce(r2,axis=-1)
            
            
        # #doesnt check for collisions, some particles disappear
        # for i in range(6):
        #     layers[np.any(rolls==i,axis=-1),i]=1


        #move particles according to their movement direction (corresponding layer)
        # layer 0 - +z
        # l1      - +y
        # l2      - +x
        # l3      - -z
        # l4      - -y
        # l5      - -x
        for i, ax_dir in enumerate([(2,1),
                                    (1,1),
                                    (0,1)]):
            ax, dir = ax_dir
            if ax not in state.geometry.periodic_dims:
                sel = [slice(None)]*3
                #save the edge next to wall and zero it in the array so that when it rolls later zeros come out on the other side
                sel[ax] = -1
                edge = layers[tuple(sel)+(i,)].copy()
                layers[tuple(sel)+(i,)] = 0
                # put the edge values on -2 as if they bounced
                sel[ax] = -2
                layers[tuple(sel)+(i,)] += edge
                #similarly for the opposite direction
                sel[ax] = 0
                edge = layers[tuple(sel)+(i+3,)].copy()
                layers[tuple(sel)+(i+3,)] = 0
                # put the edge values on 1 as if they bounced
                sel[ax] = 1
                layers[tuple(sel)+(i+3,)] += edge



            layers[...,i]=np.roll(layers[...,i],dir,ax)
            layers[...,i+3]=np.roll(layers[...,i+3],dir*-1,ax)
            #TODO: deal with collsions >1 values in arrays need to be spread out or some shit idk
            
            
        # naive collision resolution -> delete colliding particles
        layers[layers>1]=1
        
        for i,key in enumerate(self.required_keys):
            state[key] = layers[...,i]
        return state
                        

InteractionConfig = Dict[str, Dict[str, float]]        
        
class SpeciesInteraction(Rule):
    def __init__(self, geometry, interactions: InteractionConfig):
        super().__init__(geometry=geometry)
        self._interactions = interactions

        self.required_keys.extend(interactions.keys())

    @classmethod
    def load_json_config(cls, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                interaction_rules = load(file)
            return interaction_rules
        except FileNotFoundError:
            raise RuntimeError('config file not found')
        except JSONDecodeError as e:
            raise ValueError(e) 
    
    def get_interaction(self, species_a: str, species_b: str) -> float:
        '''gets interaction coeff, returns 0.0 if there's no interaction'''
        return self._interactions.get(species_a, {}).get(species_b, 0.0)
    
    def _get_neighbour_counts(self, grid:np.ndarray) -> np.ndarray:
        neighbour_counts = np.zeros_like(grid, dtype=np.int32)

        for offset in getattr(self.geometry, "_offsets", []):
            shifted = self._shift_for_offset(grid, tuple(int(v) for v in offset))
            neighbour_counts += shifted
        
        return neighbour_counts


    
    def apply_state(self, state: State) -> State:
        
        bacteria_types = list(self._interactions.keys())

        presence = {}
        occupied = np.zeros(state.shape, dtype=bool)
        neighbour_counts = {}
        spawns = {}
        deaths = {}

        for sp in bacteria_types:
            presence[sp] = state[sp].astype(bool) #checks presence of each species on the grid
            occupied |= presence[sp] #mark all occupied grid spots for all species
            neighbour_counts[sp] = self._get_neighbour_counts(presence[sp].astype(np.uint8))

            spawns[sp] = np.zeros(state.shape, dtype=np.float32) #how much a species sp wnats to grow on a gridspot
            deaths[sp] = np.zeros(state.shape, dtype=bool) #defines cells to die
        
        empty = ~occupied
        

        for bact_a, bact_b in permutations(bacteria_types, 2):
            coeff = self.get_interaction(bact_a, bact_b)

            if coeff == 0.0: #no influence on each other between species
                continue

            if coeff < 0: #species are destructive
                kill_prob = np.clip(abs(coeff) * neighbour_counts[bact_a], 0.0, 1.0) 
                killed = presence[bact_b] & (np.random.random(state.shape) < kill_prob)
                deaths[bact_b] |= killed

            elif coeff > 0: #species are in symbiosis
                spawns[bact_b] += empty * coeff * neighbour_counts[bact_a]
        
        # resolving deaths from previous stes
        for sp in bacteria_types:
            state[sp] = (presence[sp] & ~deaths[sp]).astype(state[sp].dtype)

        
        claim_grid_spot = np.stack(
            [spawns[sp] + np.random.random(state.shape) * 1e-6 for sp in bacteria_types], axis=-1
        )
        winner_idx = np.argmax(claim_grid_spot, axis=-1)
        any_claim = np.stack([spawns[sp] > 0 for sp in bacteria_types], axis=-1).any(axis=-1)
        spawn_here = empty & any_claim



        for i, sp in enumerate(bacteria_types):
            wins = spawn_here & (winner_idx == i)
            if wins.any():
                state[sp] = np.where(wins, 1, state[sp])

        return state


        
        
 