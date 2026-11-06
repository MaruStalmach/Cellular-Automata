from datetime import datetime
import json
from pathlib import Path

import numpy as np

from libs.Geometry import Geometry
from libs.State import State


class StateTracker:
    def __init__(self, keys: list[str]) -> None:
        self.history: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        self.geometry: Geometry | None = None
        self.key_layers: dict[str, int] = {}

    @staticmethod
    def _default_axes(ndim: int) -> str:
        axis_names = "xyzabcdefghijklmnopqrstuvw"
        if ndim > len(axis_names):
            raise ValueError(f"cannot infer axes for {ndim} dimensions")
        return axis_names[:ndim]

    @staticmethod
    def _normalize_file_path(filename: str) -> Path:
        file_path = Path(filename)

        if file_path.suffix == ".npz":
            return file_path

        base_dir = Path("output_files/tracked_simulations")
        if file_path.is_absolute() or len(file_path.parts) > 1:
            return file_path.with_suffix(".npz")

        return base_dir / f"{file_path.name}.npz"

    @staticmethod
    def _infer_spatial_shape(frames: list[np.ndarray]) -> tuple[int, ...]:
        if not frames:
            raise ValueError("cannot infer geometry from an empty tracker")

        spatial_shape = list(frames[0].shape)
        for frame in frames[1:]:
            candidate_shape = frame.shape
            prefix_length = 0
            for left_dim, right_dim in zip(spatial_shape, candidate_shape):
                if left_dim != right_dim:
                    break
                prefix_length += 1
            spatial_shape = spatial_shape[:prefix_length]

        if not spatial_shape:
            raise ValueError("tracked arrays do not share a common spatial shape")

        return tuple(spatial_shape)

    def record_step(self, state: State, step_no: int) -> None:
        if self.geometry is None:
            self.geometry = state.geometry

        for key in state.keys:
            self.history[key].append(state[key].copy())
            self.key_layers[key] = state.key_layers[state.key_to_index[key]]

    def save(self, filename: str = "tracked_simulation") -> None:
        base_dir = Path("output_files/tracked_simulations")

        if filename == "tracked_simulation":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = base_dir / f"{filename}_{timestamp}.npz"
        else:
            file_path = base_dir / f"{filename}.npz"

        stacked_history = {
            key: np.stack(arrays, axis=0) for key, arrays in self.history.items()
        }

        metadata = {
            "geometry_size": list(self.geometry.size) if self.geometry else None,
            "geometry_axes": self.geometry.axes if self.geometry else None,
            "geometry_periodicity": self.geometry.periodicity if self.geometry else "",
            "keys": list(self.history.keys()),
            "key_layers": [self.key_layers.get(key, 1) for key in self.history],
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            file_path,
            __state_tracker_metadata__=np.array(json.dumps(metadata)),
            **stacked_history,
        )

    def get_history(self) -> dict[str, list[np.ndarray]]:
        return self.history

    @classmethod
    def load(cls, filename: str = "tracked_simulation") -> "StateTracker":
        file_path = cls._normalize_file_path(filename)

        with np.load(file_path) as loaded:
            metadata = None
            if "__state_tracker_metadata__" in loaded.files:
                metadata = json.loads(str(loaded["__state_tracker_metadata__"].item()))

            history = {
                key: [frame.copy() for frame in loaded[key]]
                for key in loaded.files
                if key != "__state_tracker_metadata__"
            }

        tracker = cls(list(history.keys()))
        tracker.history = history

        if metadata and metadata.get("geometry_size"):
            tracker.geometry = Geometry(
                size=tuple(metadata["geometry_size"]),
                axes=metadata.get("geometry_axes") or cls._default_axes(
                    len(metadata["geometry_size"])
                ),
                periodicity=metadata.get("geometry_periodicity", ""),
            )

            tracker.key_layers = {
                key: int(layer_count)
                for key, layer_count in zip(metadata.get("keys", []), metadata.get("key_layers", []))
            }

        return tracker

    def frame(self, step_no: int = -1) -> dict[str, np.ndarray]:
        return {key: values[step_no].copy() for key, values in self.history.items()}

    def to_state(
        self,
        step_no: int = -1,
        geometry: Geometry | None = None,
        axes: str | None = None,
        periodicity: str = "",
    ) -> State:
        frame = self.frame(step_no)

        if geometry is None:
            if self.geometry is not None:
                geometry = self.geometry
            else:
                spatial_shape = self._infer_spatial_shape(list(frame.values()))
                geometry = Geometry(
                    size=spatial_shape,
                    axes=axes or self._default_axes(len(spatial_shape)),
                    periodicity=periodicity,
                )
        elif self.geometry is not None and geometry.size != self.geometry.size:
            raise ValueError(
                "provided geometry does not match the tracked spatial shape"
            )

        key_layers = []
        key_dtypes: dict[str, np.dtype | type] = {}
        spatial_rank = len(geometry.size)
        for key, values in frame.items():
            tracked_layers = self.key_layers.get(key)

            if tracked_layers is not None:
                key_layers.append(tracked_layers)
            elif values.ndim == spatial_rank:
                key_layers.append(1)
            elif values.ndim == spatial_rank + 1:
                key_layers.append(values.shape[-1])
            else:
                raise ValueError(
                    f"tracked key '{key}' has unsupported shape {values.shape}"
                )
            key_dtypes[key] = values.dtype

        state = State(
            geometry=geometry,
            cell_keys=list(frame.keys()),
            key_layers=key_layers,
            key_dtypes=key_dtypes,
            random=False,
        )

        for key, values in frame.items():
            state[key] = values

        return state

    def to_renderer_frame(self, step_no: int = -1) -> np.ndarray:
        frame = self.frame(step_no)

        if len(frame) == 1:
            return next(iter(frame.values())).copy()

        rendered_keys = []
        if self.geometry is not None:
            spatial_rank = len(self.geometry.size)
        else:
            spatial_rank = len(self._infer_spatial_shape(list(frame.values())))

        for values in frame.values():
            if values.ndim == spatial_rank + 1:
                rendered_keys.append(values.max(axis=-1))
            else:
                rendered_keys.append(values)

        return np.stack(rendered_keys, axis=0)


class TrackedStatePlayback:
    def __init__(
        self,
        tracker: StateTracker,
        step_no: int = 0,
        geometry: Geometry | None = None,
        axes: str | None = None,
        periodicity: str = "",
        stride : int = 1
    ) -> None:
        self.tracker = tracker
        self.step_no = step_no
        self.geometry = geometry
        self.axes = axes
        self.periodicity = periodicity
        self.state = tracker.to_state(step_no, geometry, axes, periodicity)
        self.stride = stride

    def __call__(self, step: bool = True) -> np.ndarray:
        if step:
            self.step_no = min(
                self.step_no + self.stride, len(next(iter(self.tracker.history.values()))) - 1
            )

        self.state = self.tracker.to_state(
            self.step_no, self.geometry, self.axes, self.periodicity
        )
        return self.tracker.to_renderer_frame(self.step_no)
