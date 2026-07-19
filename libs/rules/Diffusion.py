from libs.Rule import Rule
from libs.State import State
from libs.Geometry import Geometry
import numpy as np

class Diffusion(Rule):
    '''Random walk diffusion based on "Quantitative Cellular Automaton Model For Biofilms" by Pizarro G.
    
    arguments:
    a - defined as p0/p3, where p0 is probability that a particle wont change direction and p3 is probability that a particle will make a 180 deg turn
    p - probability for a particle to do a 90 deg turn. p=p1=p2=p4=p5 because of symmetry. 
    
    '''
    
    def __init__(self, geometry : Geometry, time_step : float, a : float, p : float, target_key : str ='substrate'):
        super().__init__(geometry, time_step)
        
        self.target_key = target_key
        p3 = (1-4*p)/(a+1)
        self.p = [a*p3,
                  p,
                  p,
                  p3,
                  p,
                  p]
        self.rolls = np.zeros(shape=geometry.size+(6,))
        self.layers = np.zeros(shape=geometry.size+(6,))
        self.shift_buffer = np.zeros(shape=geometry.size)
        self.lost_particles : int = 0
        
    def apply_state(self, state) -> State:
    
        source = state[self.target_key]
        
        #each layer corresponds to the dircetion a particle is moving, list p contains probablities of the next direction
        # p0 - no change, p3 - 180 deg turn, p1,p2,p4,p5 - 90 degree turns
        
        # layer 0 - +z
        # l1      - +y
        # l2      - +x
        # l3      - -z
        # l4      - -y
        # l5      - -x
        
        self.rolls = np.random.choice([0,1,2,3,4,5], size=state.shape + (6,), p=self.p).astype(np.uint8)
        self.rolls = (self.rolls + np.arange(6, dtype=self.rolls.dtype)) % 6
        #set empty cells as invalid value
        np.copyto(self.rolls, 6, where=(source == 0))



        # change direction 180 deg for particles that would collide with wall
        for i, ax_dir in enumerate([(2,1),
                                    (1,1),
                                    (0,1)]):
            ax, dir = ax_dir
            if ax not in state.geometry.periodic_dims:
                l=-1*ax+2
                l2=(l+3)%6
                sel = [slice(None)]*3
                sel[ax]=-1
                self.rolls[tuple(sel)+(l,)][self.rolls[tuple(sel)+(l,)]==l]=l2
                sel[ax]=0
                self.rolls[tuple(sel)+(l2,)][self.rolls[tuple(sel)+(l2,)]==l2]=l
        
        #reset layers to zeros
        self.layers[...]=0
        
        #does check for collisions
        for i in range(6):
            self.layers[..., i] = np.count_nonzero(self.rolls == i, axis=-1)
            
            
        #doesnt check for collisions, some particles disappear
        # for i in range(6):
        #     self.layers[np.any(self.rolls==i,axis=-1),i]=1


        #move particles according to their movement direction (corresponding layer)
        # layer 0 - +z
        # l1      - +y
        # l2      - +x
        # l3      - -z
        # l4      - -y
        # l5      - -x

        def shift_roll(layer: np.ndarray, axis: int, shift: int, periodic=False) -> None:
            buffer = self.shift_buffer
            buffer[...] = layer
            layer.fill(0)

            source_slices = [slice(None)] * layer.ndim
            target_slices = [slice(None)] * layer.ndim
            if periodic:
                edge_target = [slice(None)] * layer.ndim
                edge_source = [slice(None)] * layer.ndim
            if shift > 0:
                source_slices[axis] = slice(None, -shift)
                target_slices[axis] = slice(shift, None)
                if periodic:
                    edge_source[axis] = -1
                    edge_target[axis] = 0
            else:
                source_slices[axis] = slice(-shift, None)
                target_slices[axis] = slice(None, shift)
                if periodic:
                    edge_source[axis] = 0
                    edge_target[axis] = -1

            layer[tuple(target_slices)] = buffer[tuple(source_slices)]
            if periodic:
                layer[tuple(edge_target)] = buffer[tuple(edge_source)]

        for i, ax_dir in enumerate([(2,1),
                                    (1,1),
                                    (0,1)]):
            ax, dir = ax_dir
            # if ax not in state.geometry.periodic_dims:
            #     sel = [slice(None)]*3
            #     #save the edge next to wall and zero it in the array so that when it rolls later zeros come out on the other side
            #     sel[ax] = -1
            #     edge = self.layers[tuple(sel)+(i,)].copy()
            #     self.layers[tuple(sel)+(i,)] = 0
            #     # put the edge values on -2 as if they bounced
            #     sel[ax] = -2
            #     self.layers[tuple(sel)+(i,)] += edge
            #     #similarly for the opposite direction
            #     sel[ax] = 0
            #     edge = self.layers[tuple(sel)+(i+3,)].copy()
            #     self.layers[tuple(sel)+(i+3,)] = 0
            #     # put the edge values on 1 as if they bounced
            #     sel[ax] = 1
            #     self.layers[tuple(sel)+(i+3,)] += edge

            if ax in state.geometry.periodic_dims:
                shift_roll(self.layers[..., i], ax, dir, periodic=True)
                shift_roll(self.layers[..., i + 3], ax, -dir, periodic=True)
            else:
                shift_roll(self.layers[..., i], ax, dir)
                shift_roll(self.layers[..., i + 3], ax, -dir)
            #TODO: deal with collsions >1 values in arrays need to be spread out or some shit idk
            
            
        self.lost_particles += self.layers[self.layers>1].sum()-np.prod(self.layers[self.layers>1].shape)
        # naive collision resolution -> delete colliding particles
        self.layers[self.layers>1]=1
        
        
        state[self.target_key]=self.layers[...]
        return state