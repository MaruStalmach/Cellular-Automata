from libs.Rule import Rule
from libs.State import State
from libs.Geometry import Geometry
import numpy as np

DIFFUSION_TIME_STEP=0.01


class AntibioticEffect(Rule):
    def __init__(
        self, geometry: Geometry, time_step: float, bacteria_keys : list[str], biofilm_key : str,  effectiveness : list[float], base_prob : float = 0.01, inoculation_start:int=0, inoculation_end:int=None
    ):
        super().__init__(geometry, time_step)
        self.bacteria_keys = bacteria_keys
        self.biofilm_key = biofilm_key
        self.effectiveness = effectiveness
        self.base_prob = base_prob
        self.scale = max(time_step//DIFFUSION_TIME_STEP,0.1)
        assert len(bacteria_keys)==len(effectiveness)
        self.buff = np.zeros(geometry.size, dtype=np.float32)
        self.start=inoculation_start
        self.end=inoculation_end

    def apply_state(self, state: State):
        # breakpoint()
        if self.end is None:
            self.end = state.sim.max_steps
        step_no = state.sim.step_no
        if step_no<self.start or step_no>=self.end:
            return state
        biofilm_neighbors = self.neighborhood_count(state[self.biofilm_key])
        biofilm_neighbors[:,:,0] += 9 # lets count the bioreactor wall as protection as well
        probs=np.full(state[self.bacteria_keys[0]].shape,self.base_prob,dtype=np.float16)
        biofilm_neighbors = biofilm_neighbors/27 + 1
        probs = probs/(biofilm_neighbors) #prob to die goes lower with more biofilm neighbors
        probs = probs * self.scale
        for i,b_k in enumerate(self.bacteria_keys):
            bacteria_source = state[b_k].copy()
            bacteria_mask = self._cell_mask(state[b_k])
            rolls = np.random.random(bacteria_source.shape).astype(np.float16)
            rolls = rolls/(1+self.effectiveness[i])
            died = bacteria_mask & (rolls<probs)
            bacteria_source[died]=0
            state[b_k]=bacteria_source
        return state
            