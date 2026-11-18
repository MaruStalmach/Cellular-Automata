import numpy as np

from cellular_automata.registry import register_neighbourhood


@register_neighbourhood
class VonNeumannNeighbourhood:
    def generate_offsets(self, ndim: int, radius: int = 1) -> np.ndarray:
        if ndim < 1:
            raise ValueError(f"dimensionality must be at least 1, got {ndim}")
        if radius < 1:
            raise ValueError(f"radius must be at least 1, got {radius}")

        offsets = []
        for dim in range(ndim):
            for direction in (-radius, radius):
                offset = [0] * ndim
                offset[dim] = direction
                offsets.append(tuple(offset))
        return np.array(offsets, dtype=np.int64)
