from __future__ import annotations
from typing import Callable, Iterable, Optional

import numpy as np

from libs.Geometry import Geometry


class State:
    """Array-backed cellular automaton state -> swapped creating Cell obj on array to working on multiple layers of arrays"""

    def __init__(
        self,
        geometry: Geometry,
        random: bool = True,
        cell_keys: Iterable[str] = (),
        random_func: Optional[Callable[..., object]] = None,
        random_args: Optional[dict] = None,
        dtype: np.dtype | type = np.uint8,
    ) -> None:
        

        self.geometry = geometry
        self.keys = tuple(cell_keys)
        self.key_to_index = {key: idx for idx, key in enumerate(self.keys)}
        self.dtype = dtype


        expected_shape = tuple(self.geometry.size) + (len(self.keys),)

        self._data = np.zeros(expected_shape, dtype=self.dtype)

        if not random or self._data.size == 0:
            return



        if random_func is None:
            self._data[...] = np.random.randint(0, 2, size=expected_shape, dtype=self.dtype)
            return

        rand_args = random_args or {}
        try:
            generated = random_func(size=expected_shape, **rand_args)
        except TypeError:
            generated = random_func(**rand_args)

        generated = np.asarray(generated)
        if generated.shape != expected_shape:
            generated = np.broadcast_to(generated, expected_shape)




        self._data[...] = np.asarray(generated, dtype=self.dtype)




    # -- Properties -------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """return a view of the internal ndarray"""
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        """replace the internal array with 'new_data' after validation
        """
        arr = np.asarray(new_data, dtype=self.dtype)
        required_shape = tuple(self.geometry.size) + (len(self.keys),)

        if arr.shape != required_shape:
            raise ValueError(f"expected data shape {required_shape}, got {arr.shape}")

        # keep an internal copy 
        self._data = arr.copy()



    @property
    def shape(self) -> tuple[int, ...]:
        """return the geometric (unpadded) shape of the grid (no keys).
        xample: a 3D grid returns (nx, ny, nz)
        """
        return self.geometry.size

    def copy(self) -> "State":
        """return a deep copy"""
        new_state = State(self.geometry, random=False, cell_keys=self.keys, dtype=self.dtype)
        new_state.data = self.data
        return new_state




    def __getitem__(self, key: str) -> np.ndarray:
        """return the 2D/3D/... grid corresponding to `key`
        example: state['alive'] returns an array shaped like
        geometry.size
        """
        if key not in self.key_to_index:
            raise KeyError(f"unknown key: {key}")

        index = self.key_to_index[key]
        return self._data[..., index]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """set the grid values for a given key
        """
        if key not in self.key_to_index:
            raise KeyError(f"unknown key: {key}")

        index = self.key_to_index[key]
        self._data[..., index] = np.asarray(value, dtype=self.dtype)




    def key_grid(self, key: str, as_bool: bool = False) -> np.ndarray:
        """return the grid for key, optionally as a boolean mask
        when as_bool=True, non-zero values are treated as True
        """
        grid = self[key]
        if as_bool:
            return grid.astype(bool)
        return grid

