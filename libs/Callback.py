from libs.Sim import CellularAutomaton
import numpy as np


class Callback():
    '''A class to generalize update_callback functions used for rendering'''
    def __init__(self, key_to_display : str, sim_obj_reference : CellularAutomaton, step_stride : int = 1):
        self.key = key_to_display
        self.sim = sim_obj_reference
        self.stride = step_stride
        assert isinstance(sim_obj_reference, CellularAutomaton)
    
    
    def __call__(self, step:bool=True):
        
        if step:
            for _ in range(self.stride):
                self.sim.step()
        
        try:
            return self.sim.state[self.key]
        except:
            print(f"RENDERER ERROR: NO SUCH KEY TO RENDER AS {self.key}")
            return 0
        
class ReduceCallback(Callback):
    def __init__(self, key_to_display, sim_obj_reference, step_stride = 1, reduce_op : np.ufunc = np.maximum):
        super().__init__(key_to_display, sim_obj_reference, step_stride)
        self.op = reduce_op
        assert isinstance(reduce_op,np.ufunc)
        # this is intended only for layered state arrays
        assert len(self.sim.state[self.key].shape)==4
        
    def __call__(self, step = True):
        
        if step:
            for _ in range(self.stride):
                self.sim.step()
        
        try:
            return self.op.reduce(self.sim.state[self.key],axis=-1)
        except:
            return 0
        
class MultiKeyCallback(Callback):
    def __init__(self, keys_to_display, sim_obj_reference, step_stride=1):
        super().__init__(keys_to_display,sim_obj_reference,step_stride)
        self.callbacks = []
        for k in self.key:
            if len(self.sim.state[k].shape)==4:
                self.callbacks.append(ReduceCallback(k,self.sim,step_stride))
            else:
                self.callbacks.append(Callback(k,self.sim,step_stride))
    
    def __call__(self, step = True):
        arr_list = []

        if step:
            for _ in range(self.stride):
                self.sim.step()

        for cb in self.callbacks:
            try:
                arr_list.append(cb(step=False))
            except:
                print('ERROR IN MULTICALLBACK')
                arr_list.append(0)
        # stack on axis 0 so it gives the arrays in order when flattened
        return np.stack(arr_list,axis=0)