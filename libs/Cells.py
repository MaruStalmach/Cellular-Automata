from libs.Geometry import Geometry
import numpy as np
from copy import copy



#TODO implement
class Cell():
    """represents a singular cell
    """
    
    def __init__(self, keys, random : bool = True, random_func : callable = None, random_args : dict = None, coords = None):
        """
    creates a cell\n

    args:\
    keys - elements inside cell, e.g. bacteria counts or CO2 volume/mass
    random - whether to fill cells with randomly generated values. if False all values will be zero
    random_func - if ranom=True, thisfunction is used for filling cells with values if None, np.random.random is used
    random_args - optional dict of arguments for random_func 
        """
        #TODO: allow configurable initialization
        
        self.random = random
        self.random_args = random_args
        self.random_func = random_func
        self.coords = tuple(coords) if coords is not None else None
        
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
        
        self._data = new_data.copy()
        
    def __copy__(self):
        a = Cell(self.keys,random=False)
        a.data = self.data
        a.coords = self.coords
        return a
    
    def __add__(self, other):
        """for cell+cell, returns a new cell 
        if self or other is a zero cell then returns reference to the nonzero cell

        Args:
            other (_type_): _description_
        """
        assert isinstance(other, Cell)

        if other.is_zero_cell:
            return self.__copy__()
        if self.is_zero_cell:
            return other.__copy__()
        
        new_cell = self.__copy__()
        for key,value in other._data.items():
            new_cell._data[key] += value

        return new_cell
            
        
        return other
    
    def __mul__(self, other):
        """adds multiplication for Cell * int. This is used during neighbor selection via mask

        """
        assert(isinstance(other,int))
        
        if other==0:
            return ZERO_CELL
        else:
            return self
    
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
        
    def __repr__(self):
        out = f'Cells.Cell(keys={self.keys}, random={self.random}, func={self.random_func}, args = {self.random_args})'
        if not self.is_zero_cell:
            out += f'\ndata={self.data}'
        return out

    def __str__(self):
        if self.is_zero_cell:
            out = 'Zero Cell'
        else:
            out = f'Cell with data: {self.data}'
        
    @staticmethod
    def from_dict(dict : dict):
        cell = Cell(keys=list(dict.keys()),random=False)
        for key,val in dict.items():
            cell[key] = val
        return cell


        
        
ZERO_CELL = Cell(keys=[])


        
#TODO implement       
class State():
    """Collection of cells, arranged in a 2D or 3D matrix"""
    
    def __init__(self, geometry: Geometry, random=True, cell_keys = []):
        """
    creates a state of cells

    args:

    geometry - Geometry object
    random - whether or not to fill cells with random values
    cell_keys - keys to use for cell clreation
        """
        
        self.geometry = geometry
        cell_num = np.prod(geometry.size)
        
        self._data = np.empty(cell_num, dtype=object)

        if not random:
            for i in range(cell_num):
                self._data[i] = Cell(keys=cell_keys,random=False) 
        else:
            for i in range(cell_num):
                self._data[i] = Cell(keys=cell_keys, random=True, random_func=np.random.randint, random_args={'low': 2})
        
        
        self._data = self._data.reshape(geometry.size)
            
        # add padding
        # self._data = np.pad(self._data,((1,1),(1,1),(1,1)),constant_values=ZERO_CELL)
        self._data = self._pad_data(self._data)
        
    def _pad_data(self, unpadded_data:np.ndarray) -> np.ndarray:
        padded = unpadded_data

        for axis, axis_name in enumerate(self.geometry.axes):
            pwidth = [(0,0)] * self.geometry.ndim
            pwidth[axis] = (1,1)

            if axis_name in self.geometry.periodicity:
                padded = np.pad(padded, pad_width=pwidth, mode='wrap')
            else:
                padded = np.pad(padded, pad_width=pwidth, mode='constant', constant_values=ZERO_CELL)


        return padded
    

    @property
    def data(self):
        #remove padding
        # return self._data[1:-1,1:-1,1:-1]
        unpad = []
        for _ in range(self.geometry.ndim):
            unpad.append(slice(1,-1))
        
        unpad = tuple(unpad)

        return self._data[unpad]
    
    @data.setter
    def data(self,val : np.ndarray):
        self._data = self._pad_data(val)
    
    @property
    def shape(self):
        #shape of unpadded data
        return self.data.shape
        

