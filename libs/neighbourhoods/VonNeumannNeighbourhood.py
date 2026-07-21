import numpy as np

class VonNeumannNeighbourhood:
    def generate_offsets(self, ndim: int, radius:int=1) -> np.ndarray:
        assert ndim > 0, "dimensionality must be greater than 0"
        assert radius > 0, "radius must be greater than 0"
            
        offsets = []
        for dim in range(ndim):
            for direction in (-radius, radius):
                offset = [0] * ndim
                offset[dim] = direction
                offsets.append(tuple(offset))
        return np.array(offsets, dtype=np.int64)