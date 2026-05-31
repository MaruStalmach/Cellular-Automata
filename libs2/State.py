from typing import Callable, Iterable, Optional

import numpy as np

from libs.Geometry import Geometry


class State:
    """array-based cellular automaton state -> swapped creating Cell obj on array to working on multiple layers of arrays
    
    args:
    geometry - Geometry object definind the size of the array
    random - whether or not to fill cells with random values
    cell_keys - keys to use for creation of arrays
    random_func (optional) - func generating initial state of array
    random_args (optional) - args for random_func
    dtype - type of the data stored in array
    key_dtypes (optional) - overwrites data types for specified keys
    """

    def __init__(
        self,
        geometry: Geometry,
        random: bool = True,
        cell_keys: Iterable[str] = (),
        random_func: Optional[Callable[..., object]] = None,
        random_args: Optional[dict] = None,
        dtype: np.dtype | type = np.uint8,
        key_dtypes: Optional[dict[str, np.dtype | type]] = None,
    ) -> None:
        

        self.geometry = geometry
        self.keys = tuple(cell_keys)
        self.key_to_index = {key: idx for idx, key in enumerate(self.keys)}
        self.dtype = dtype
        
        # per-key dtypes 
        if key_dtypes is not None:
            self.key_dtypes:dict[str, np.dtype | type] = key_dtypes.copy()
        else:
            self.key_dtypes:dict[str, np.dtype | type] = {}

        #shape of the spatial grid and number of keys
        grid_dims = tuple(self.geometry.size)
        num_of_keys = len(self.keys)

        # store per-key arrays in a dict so each key can have its own dtype
        self._data: dict[str, np.ndarray] = {}
        for key in self.keys:
            kd = self.key_dtypes.get(key, self.dtype)
            self._data[key] = np.zeros(grid_dims, dtype=kd)




        if not random or num_of_keys == 0:
            return


        #fill array with initial values
        expected_shape = grid_dims + (num_of_keys,)

        if random_func is None:
            for key in self.keys:
                kd = self.key_dtypes.get(key, self.dtype)
                self._data[key][...] = np.random.randint(0, 2, size=grid_dims).astype(kd)
            return

        rand_args = random_args or {}
        try:
            generated = random_func(size=expected_shape, **rand_args)
        except TypeError:
            generated = random_func(**rand_args)

        generated = np.asarray(generated)
        if generated.shape != expected_shape:
            generated = np.broadcast_to(generated, expected_shape)

        #distribute to per-key arrays 
        for idx, key in enumerate(self.keys):
            kd = self.key_dtypes.get(key, self.dtype)
            self._data[key][...] = np.asarray(generated[..., idx]).astype(kd)

    



    # -- Properties -------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """return a view of the internal ndarray"""
        if len(self.keys) == 0:
            return np.empty(tuple(self.geometry.size) + (0,), dtype=self.dtype)
        return np.stack([self._data[k] for k in self.keys], axis=-1)

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        '''replace internal data with array'''
        arr = np.asarray(new_data)
        required_shape = tuple(self.geometry.size) + (len(self.keys),)

        for idx, key in enumerate(self.keys):
            kd = self.key_dtypes.get(key, self.dtype)
            self._data[key] = arr[..., idx].astype(kd).copy()



    @property
    def shape(self) -> tuple[int, ...]:
        """return the geometric (unpadded) shape of the grid (no keys)
        xample: a 3D grid returns (nx, ny, nz)
        """
        return self.geometry.size

    def copy(self) -> "State":
        new_state = State(self.geometry, random=False, cell_keys=self.keys, dtype=self.dtype)
        for key in self.keys:
            new_state._data[key] = self._data[key].copy()
        return new_state


    def _pad_data(self, unpadded_data: np.ndarray, constant_value: int = 0) -> np.ndarray:
        """pad spatial axes only and preserve the final keys axis
        - for axes listed in self.geometry.periodicity use wrap padding
        - for non-periodic axes use constant padding with constant_value
        """
        padded = unpadded_data

        for axis, axis_name in enumerate(self.geometry.axes):
            pwidth = [(0, 0)] * padded.ndim
            pwidth[axis] = (1, 1)

            if axis_name in getattr(self.geometry, "periodicity", ""):
                padded = np.pad(padded, pad_width=pwidth, mode="wrap")
            else:
                padded = np.pad(padded, pad_width=pwidth, mode="constant", constant_values=constant_value)

        return padded



    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        '''set val for a key'''
        # Respect per-key dtype if provided, otherwise fall back to global dtype
        kd = self.key_dtypes.get(key, self.dtype)
        self._data[key] = np.asarray(value, dtype=kd).copy()


    def key_grid(self, key: str, as_bool: bool = False) -> np.ndarray:
        grid = self[key]
        if as_bool:
            return grid.astype(bool)
        return grid

