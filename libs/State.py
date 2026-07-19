from typing import Callable, Iterable, Optional

import numpy as np

from libs.Geometry import Geometry


class State:
    """Manages multi-layered, array-based cellular automaton states

    Args:
        geometry (Geometry): Grid spatial dimensions and boundaries
        cell_keys (Iterable[str], optional): Names of simulation layers. Defaults to ()
        key_dtypes (dict[str, np.dtype | type], optional): Per-key data type overrides. Defaults to None
        key_layers (Iterable[int], optional): Depth/number of layers per key. Defaults to None
        random (bool, optional): If True, initializes with random values; otherwise 0s. Defaults to False
        random_func (Iterable[Callable], optional): Per-key random generation functions. Defaults to None
        random_args (Iterable[dict], optional): Kwargs mapped to `random_func`. Defaults to None
        dtype (np.dtype | type, optional): Global default array data type. Defaults to np.uint8
    """

    def __init__(
        self,
        geometry: Geometry,
        cell_keys: Iterable[str] = (),
        key_dtypes: Optional[dict[str, np.dtype | type]] = None,
        key_layers: Optional[Iterable] = None,
        random: bool = False,
        random_func: Optional[Iterable[Callable[..., object]]] = None,
        random_args: Optional[Iterable[dict]] = None,
        dtype: np.dtype | type = np.uint8,
    ) -> None:

        self.geometry = geometry

        self.keys = tuple(cell_keys)
        self.key_to_index = {key: idx for idx, key in enumerate(self.keys)}
        self.dtype = dtype

        # per-key dtypes
        if key_dtypes is not None:
            self.key_dtypes: dict[str, np.dtype | type] = key_dtypes.copy()
        else:
            self.key_dtypes: dict[str, np.dtype | type] = {}

        if key_layers is not None:
            self.key_layers = key_layers.copy()
        else:
            self.key_layers = [1 for _ in range(len(cell_keys))]

        # shape of the spatial grid and number of keys
        grid_dims = self.geometry.size
    
        # store per-key arrays in a dict so each key can have its own dtype
        self._data: dict[str, np.ndarray] = {}
        for i, key in enumerate(self.keys):
            kd = self.key_dtypes.get(key, self.dtype)
            ### for keys with multiple layers create a bigger array
            self._data[key] = np.zeros(grid_dims + (self.key_layers[i],), dtype=kd)

        if random:
            # fill array with initial values

            if random_func is None:
                for i, key in enumerate(self.keys):
                    kd = self.key_dtypes.get(key, self.dtype)

                    self._data[key][...] = np.random.randint(
                        0, 2, size=grid_dims + (self.key_layers[i],)
                    ).astype(kd)
            else:
                if callable(random_func):
                    random_func = [random_func] * len(self.keys)

                rand_args = random_args or [{} for _ in range(len(self.keys))]
                for i, key in enumerate(self.keys):
                    kd = self.key_dtypes.get(key, self.dtype)
                    expected_shape = grid_dims + (self.key_layers[i],)
                    try:
                        generated = random_func[i](
                            size=expected_shape, dtype=kd, **rand_args[i]
                        )
                    except TypeError:
                        try:
                            generated = random_func[i](
                                size=expected_shape, **rand_args[i]
                            )
                        except TypeError:
                            generated = random_func[i](**rand_args[i])

                    generated = np.asarray(generated)
                    if generated.shape != expected_shape:
                        generated = np.broadcast_to(generated, expected_shape)

                    self._data[key][...] = generated.astype(kd)

        ### squeeze last dim
        for key in self.keys:
            try:
                ### if there was only one layer, remove the last dim/axis
                self._data[key] = self._data[key].squeeze(axis=-1)
            except ValueError:
                ### squeeeze raises a ValueError if selected dim is not 1
                pass

    # -- Properties -------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """return a view of the internal ndarray"""
        if not self.keys:
            return np.empty(self.geometry.size + (0,), dtype=self.dtype)
        # had to change this since introducing the concept of layers per key - it should still return similiar output
        list_to_stack = []
        for k in self.keys:
            arr = self._data[k].copy()
            # keys with multiple layers will have dim + 1 dimensions
            if len(arr.shape) == len(self.geometry.size) + 1:
                # has to be unstacked to become a tuple of grid_dim shaped arrays
                list_to_stack.extend(np.unstack(arr, axis=-1))
            else:
                list_to_stack.append(arr)
        return np.stack(list_to_stack, axis=-1)

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        """replace internal data with array"""
        arr = np.asarray(new_data)
        # required_shape = tuple(self.geometry.size) + (len(self.keys),)

        arr_cnt = 0
        for key_cnt, key in enumerate(self.keys):
            kd = self.key_dtypes.get(key, self.dtype)
            layers = self.key_layers[key_cnt]
            # stack multiple layer keys
            if layers > 1:
                self._data[key] = np.stack(
                    arr[..., arr_cnt : arr_cnt + layers], axis=-1
                ).astype(kd)
            else:
                self._data[key] = arr[..., arr_cnt].astype(kd).copy()
            arr_cnt += layers

    @property
    def shape(self) -> tuple[int, ...]:
        """return the geometric (unpadded) shape of the grid (no keys)
        xample: a 3D grid returns (nx, ny, nz)
        """
        return self.geometry.size

    def copy(self) -> "State":
        ##TODO:FIX THIS
        new_state = State(
            self.geometry, random=False, cell_keys=self.keys, dtype=self.dtype
        )
        for key in self.keys:
            new_state._data[key] = self._data[key].copy()
        return new_state

    def _pad_data(
        self, unpadded_data: np.ndarray, constant_value: int = 0
    ) -> np.ndarray:
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
                padded = np.pad(
                    padded,
                    pad_width=pwidth,
                    mode="constant",
                    constant_values=constant_value,
                )

        return padded

        # kd = self.key_dtypes.get(key, self.dtype)
        # self._data[key] = np.asarray(value, dtype=kd).copy()

    def key_grid(self, key: str, as_bool: bool = False) -> np.ndarray:
        grid = self[key]
        if as_bool:
            return grid.astype(bool)
        return grid

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """set val for a key"""
        # respect per-key dtype if provided, otherwise fall back to global dtype
        if key not in self._data:
            raise KeyError("key not in set")

        target = self._data[key]
        arr = np.asarray(value)

        if target.ndim == arr.ndim + 1 and target.shape[-1] == 1:
            arr = np.expand_dims(arr, axis=-1)

        target[...] = arr
