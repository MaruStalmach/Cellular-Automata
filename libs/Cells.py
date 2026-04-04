from libs.Geometry import Geometry
import numpy as np
from copy import copy





#TODO implement
class Cell():
    """represents a singular cell
    """
    
    def __init__(self, keys, random : bool = True, random_func : callable = None, random_args : dict = None):
        """
    creates a cell\n

    args:\
    keys - elements inside cell, e.g. bacteria counts or CO2 volume/mass
    random - whether to fill cells with randomly generated values. if False all values will be zero
    random_func - if ranom=True, thisfunction is used for filling cells with values if None, np.random.random is used
    random_args - optional dict of arguments for random_func 
        """
        #TODO: allow configurable initialization
        
        if not keys:
            self.is_zero_cell = True
        else:
            self.is_zero_cell = False
        
        self.keys = keys
        if random_func is None:
            random_func = np.random.random
        if random:
            if random_args is not None:
                self._data = dict([(key,random_func(**random_args)) for key in keys])
            else:
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
        if self or other is a zero cell then returns reference to the nonzero cell

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
    
    def __getitem__(self, key):
        if self.is_zero_cell:
            return 0
        if key not in self.keys:
            raise(TypeError)
        
        return self._data[key]
    
    def __setitem__(self, key, value):

        #TODO: how to handle zero cell?
        if self.is_zero_cell:
            raise(Exception)
        if key not in self.keys:
            raise(TypeError)
        
        self._data[key] = value


        
        
ZERO_CELL = Cell(keys=[])
        
#TODO implement       
class State():
    """
collection of cells, arranged in a 2D or 3D matrix
    """
    
    def __init__(self, geometry: Geometry, random=True, cell_keys = []):
        """
    creates a state of cells

    args:

    geometry - Geometry object
    random - whether or not to fill cells with random values
    cell_keys - keys to use for cell clreation
        """
        
        self.geometry = geometry
        
        if not random:
            self._data = np.full(geometry.size,ZERO_CELL)
        else:
            self._data = np.full(geometry.size, Cell(cell_keys,random_func=np.random.randint, random_args = {'low':2}))
    @property
    def data(self):
        return self._data
    
    @property
    def shape(self):
        return self._data.shape
        

