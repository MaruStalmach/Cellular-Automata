import numpy as np
from scipy.sparse import csr_matrix

from cellular_automata.Neighbourhood import Neighbourhood
from cellular_automata.neighbourhoods.MooreNeighbourhood import MooreNeighbourhood


class Geometry:
    """Represents the spatial geometry, limits, and boundary conditions of a cellular automaton grid

    Args:
        size (tuple[int, ...]): Spatial dimensions of the grid (e.g., (10, 10) or (10, 10, 10))
        axes (str): String identifier for each axis (e.g., "xy" or "xyz")
        periodicity (str): String indicating which axes have periodic (wrapping) boundaries (e.g., "xy" means both x and y wrap)
    """

    def __init__(
        self,
        size: tuple[int, ...],
        axes: str,
        periodicity: str,
        neighbourhood: Neighbourhood | None = None,
    ):
        self.periodicity = periodicity.lower()
        self.axes = axes.lower()
        self.size = size
        self.num_cells = np.prod(self.size)
        self.ndim = len(size)

        self.periodic_dims = {
            dim_idx for dim_idx, axis in enumerate(self.axes) if axis in self.periodicity
        }

        self.neighbourhood = neighbourhood or MooreNeighbourhood()
        self._offsets = self.neighbourhood.generate_offsets(self.ndim)

    def neighbours_of(self, coords: tuple[int, ...]) -> list[tuple[int, ...]]:
        """The cells next to `coords`, in neighbourhood offset order.

        Periodic axes wrap around; offsets that leave the grid on a non-periodic
        axis are skipped.
        """
        neighbours: list[tuple[int, ...]] = []
        for offset in self._offsets:
            target = []
            for axis, (coord, delta) in enumerate(zip(coords, offset, strict=True)):
                value = coord + int(delta)
                if axis in self.periodic_dims:
                    value %= self.size[axis]
                elif value < 0 or value >= self.size[axis]:
                    break
                target.append(value)
            else:
                neighbours.append(tuple(target))
        return neighbours

    def _apply_offset(
        self, flat_coords: np.ndarray, offset: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        shifted = flat_coords + offset
        is_valid = np.ones(self.num_cells, dtype=bool)

        for dim in range(self.ndim):
            if dim in self.periodic_dims:
                shifted[:, dim] %= self.size[dim]  # wrap if periodic axis
            else:
                within_bounds = (shifted[:, dim] >= 0) & (shifted[:, dim] < self.size[dim])
                is_valid &= within_bounds

        origin_idx = np.ravel_multi_index(tuple(flat_coords[is_valid].T), self.size)
        neighbour_idx = np.ravel_multi_index(tuple(shifted[is_valid].T), self.size)

        return origin_idx, neighbour_idx

    def generate_neighbourhood_matrix(self) -> csr_matrix:

        flat_coords = np.indices(self.size).reshape(self.ndim, -1).T

        row_idx, col_idx = [], []

        for offset in self._offsets:
            origin_idx, neighbour_idx = self._apply_offset(flat_coords, offset)
            row_idx.append(origin_idx)
            col_idx.append(neighbour_idx)

        rows = np.concatenate(row_idx)
        cols = np.concatenate(col_idx)
        data = np.ones(rows.shape[0], dtype=np.int8)

        return csr_matrix((data, (rows, cols)), shape=(self.num_cells, self.num_cells))
