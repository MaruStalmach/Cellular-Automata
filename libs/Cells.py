from libs.Geometry import Geometry
import numpy as np






#TODO implement
class Cell():
    
    def __init__(self, keys, random=True, random_func=None):
        #TODO: allow configurable initialization
        self.keys = keys
        if random_func is None:
            random_func = np.random.random
        if random:
            self._data = dict([(key,random_func()) for key in keys])
        else:
            self._data = dict([(key,0) for key in keys])
            
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data: dict):
        if self._data.keys != new_data.keys:
            raise(TypeError)
        self._data = new_data
        
    @staticmethod
    def zero_cell(keys):
        return Cell(keys, random=False)
    
    @staticmethod
    def zero_cell_like(cell : 'Cell'):
        return Cell(cell.keys, random=False)
        
        
#TODO implement       
class State():
    
    def __init__(self, geometry: Geometry):
        #TODO: actually fill this with Cells based on Geometry
        
        self.geometry = geometry
        
        self._data = np.full(geometry.size)
        