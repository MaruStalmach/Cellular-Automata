from typing import Callable, Iterable, Optional

import numpy as np

from cellular_automata.Geometry import Geometry


def _check_fits(key: str, values: np.ndarray, dtype: np.dtype) -> None:
    """Raise ValueError if integer `values` would wrap around when cast to `dtype`."""
    if values.size == 0 or not np.issubdtype(dtype, np.integer):
        return
    if not (np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_)):
        return
    info = np.iinfo(dtype)
    low, high = values.min(), values.max()
    if low < info.min or high > info.max:
        raise ValueError(
            f"random_func for key '{key}' gave values {low}..{high}, which don't fit "
            f"its dtype {dtype} ({info.min}..{info.max})"
        )


class State:
    """Manages multi-layered, array-based cellular automaton states

    Args:
        geometry (Geometry): Grid spatial dimensions and boundaries
        cell_keys (Iterable[str], optional): Names of simulation layers. Defaults to ()
        key_dtypes (dict[str, np.dtype | type], optional): Per-key data type overrides. Defaults to None
        key_layers (Iterable[int], optional): Depth/number of layers per key. Defaults to None
        random (bool, optional): If True, initializes with random values; otherwise 0s. Defaults to False
        random_func (Callable | Iterable[Callable | None], optional): Per-key initialisers,
            each called as ``func(size=shape, **random_args)`` where shape is the key's
            grid shape plus its layer axis; the result is broadcast to that shape and
            cast to the key's dtype. None leaves a key at 0. Defaults to None
        random_args (Iterable[dict], optional): Kwargs mapped to `random_func`. Defaults to None
        dtype (np.dtype | type, optional): Global default array data type. Defaults to np.uint8
        rng (np.random.Generator, optional): Generator for the default random fill. Defaults to an unseeded one
    """

    def __init__(
        self,
        geometry: Geometry,
        cell_keys: Iterable[str] = (),
        key_dtypes: Optional[dict[str, np.dtype | type]] = None,
        key_layers: Optional[Iterable] = None,
        random: bool = False,
        random_func: Optional[Callable[..., object] | Iterable[Callable[..., object] | None]] = None,
        random_args: Optional[Iterable[dict | None]] = None,
        dtype: np.dtype | type = np.uint8,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()

        self.geometry = geometry

        self.keys = tuple(cell_keys)
        self.key_to_index = {key: idx for idx, key in enumerate(self.keys)}
        self.dtype = dtype

        # per-key dtypes
        self.key_dtypes: dict[str, np.dtype | type] = (
            dict(key_dtypes) if key_dtypes is not None else {}
        )

        if key_layers is not None:
            self.key_layers = list(key_layers)
        else:
            self.key_layers = [1 for _ in self.keys]
        if len(self.key_layers) != len(self.keys):
            raise ValueError(
                f"{len(self.keys)} cell_keys but {len(self.key_layers)} key_layers"
            )

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

                    self._data[key][...] = rng.integers(
                        0, 2, size=grid_dims + (self.key_layers[i],)
                    ).astype(kd)
            else:
                funcs: list[Callable[..., object] | None] = (
                    [random_func] * len(self.keys) if callable(random_func) else list(random_func)
                )
                rand_args = list(random_args or [{} for _ in range(len(self.keys))])
                for i, key in enumerate(self.keys):
                    kd = self.key_dtypes.get(key, self.dtype)
                    expected_shape = grid_dims + (self.key_layers[i],)
                    func = funcs[i]
                    if func is None:
                        continue
                    generated = np.asarray(func(size=expected_shape, **(rand_args[i] or {})))
                    if generated.shape != expected_shape:
                        generated = np.broadcast_to(generated, expected_shape)
                    _check_fits(key, generated, np.dtype(kd))
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
        list_to_stack: list[np.ndarray] = []
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
                # the layers are already the last axis; np.stack would iterate the first
                self._data[key] = arr[..., arr_cnt : arr_cnt + layers].astype(kd)
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
        """Independent copy of the arrays."""
        new_state = State(
            self.geometry, 
            random=False, 
            cell_keys=self.keys, 
            dtype=self.dtype, 
            key_dtypes=self.key_dtypes, 
            key_layers=self.key_layers
        )
        for key in self.keys:
            new_state._data[key] = self._data[key].copy()
        return new_state

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
