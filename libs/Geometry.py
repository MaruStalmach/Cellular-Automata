import numpy as np
from scipy.sparse import csr_matrix
from itertools import product

class Geometry():
    
    def __init__(
        self, 
        size: tuple[int,...], 
        axes: str, 
        periodicity: str = ''
    ):
        
        self.periodicity = periodicity.lower()
        self.axes = axes.lower()
        self.size = size
        self.num_cells = np.prod(self.size)
        self.ndim = len(size)
