from libs.Geometry import Geometry
import numpy as np
from copy import copy





#TODO implement
class Cell():
    
    def __init__(self, keys, random=True, random_func=None):
        #TODO: allow configurable initialization
        
        if not keys:
            self.is_zero_cell = True
        else:
            self.is_zero_cell = False
        
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
        if self._data.keys() != new_data.keys():
            raise(TypeError)
        self = Cell(self.keys,random=False)
        self._data = new_data
        
    def __copy__(self):
        a = Cell(self.keys,random=False)
        a.data = self.data
        return a
    
    def __add__(self, other):
        """for cell+cell, returns reference to other (which may or may not be a good idea)

        Args:
            other (_type_): _description_
        """
        assert isinstance(other, Cell)
        if other.is_zero_cell:
            return self
        if self.is_zero_cell:
            return other
        
        for key,value in self._data.items():
            other._data[key] += value
            
        
        return other
            

        
        
ZERO_CELL = Cell(keys=[])
        
#TODO implement       
class State():
    
    def __init__(self, geometry: Geometry):
        #TODO: actually fill this with Cells based on Geometry
        
        self.geometry = geometry
        
        self._data = np.full(geometry.size,copy(ZERO_CELL))
        