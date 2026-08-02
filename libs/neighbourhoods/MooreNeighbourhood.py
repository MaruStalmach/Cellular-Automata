import numpy as np
from itertools import product


class MooreNeighbourhood:
    def generate_offsets(self, ndim: int, radius: int = 1) -> np.ndarray:
        assert ndim > 0, "dimensionality must be greater than 0"
        assert radius > 0, "radius must be greater than 0"

        offsets = [
            offset
            for offset in product((-radius, 0, radius), repeat=ndim)
            if any(component != 0 for component in offset)
        ]
        return np.array(offsets, dtype=np.int64)
