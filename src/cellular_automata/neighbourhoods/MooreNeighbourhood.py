from itertools import product

import numpy as np

from cellular_automata.registry import register_neighbourhood


@register_neighbourhood
class MooreNeighbourhood:
    def generate_offsets(self, ndim: int, radius: int = 1) -> np.ndarray:
        if ndim < 1:
            raise ValueError(f"dimensionality must be at least 1, got {ndim}")
        if radius < 1:
            raise ValueError(f"radius must be at least 1, got {radius}")

        offsets = [
            offset
            for offset in product((-radius, 0, radius), repeat=ndim)
            if any(component != 0 for component in offset)
        ]
        return np.array(offsets, dtype=np.int64)
