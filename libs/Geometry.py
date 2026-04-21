import numpy as np
from scipy.sparse import csr_matrix
from itertools import product

class Geometry():
    '''A class to represent the geometry of a CA grid. 
    Contains limits and periodicity information'''
    
    def __init__(
        self, 
        size: tuple[int,...], 
        axes: str, 
        periodicity: str
    ):
        
        self.periodicity = periodicity.lower()
        self.axes = axes.lower()
        self.size = size
        self.num_cells = np.prod(self.size)
        self.ndim = len(size)

        self.periodic_dims = {
            dim_idx for dim_idx, axis in enumerate(self.axes) 
            if axis in self.periodicity
        }

        #TODO: hardcoded moores, make this configurable
        offsets = []
        for offset in product((-1, 0, 1), repeat=self.ndim):
            if any(component != 0 for component in offset):
                offsets.append(offset)
           
        self._offsets = np.array(offsets, dtype=np.int64)


    def _apply_offset(
        self,
        flat_coords: np.ndarray,  
        shifted: np.ndarray,        
        offset: np.ndarray,         
    ) -> tuple[np.ndarray, np.ndarray]:
        
        shifted = flat_coords + offset
        is_valid = np.array([True] * self.num_cells) 

        for dim in range(self.ndim):
            if dim in self.periodic_dims:
                shifted[:, dim] %= self.size[dim] #wrap if periodic axis
            else:
                within_bounds = (
                    (shifted[:, dim] >= 0) &
                    (shifted[:, dim] < self.size[dim])
                )
                is_valid &= within_bounds
        
        origin_idx = np.ravel_multi_index(tuple(flat_coords[is_valid].T), self.size)
        neighbour_idx = np.ravel_multi_index(tuple(shifted[is_valid].T), self.size)

        return origin_idx, neighbour_idx #type: ignore


    def generate_neighbourhood_matrix(
        self
    ) -> csr_matrix:
        
        flat_coords = np.indices(self.size).reshape(self.ndim, -1).T
        shifted = np.empty_like(flat_coords)   

        row_idx, col_idx = [], []

        for offset in self._offsets:
            origin_idx, neighbour_idx = self._apply_offset(flat_coords, shifted, offset)
            row_idx.append(origin_idx)
            col_idx.append(neighbour_idx)

        rows = np.concatenate(row_idx)
        cols = np.concatenate(col_idx)
        data = np.ones(rows.shape[0], dtype=np.int8)

        return csr_matrix((data, (rows, cols)), shape=(self.num_cells, self.num_cells))