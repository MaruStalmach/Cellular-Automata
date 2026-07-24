from datetime import datetime
from pathlib import Path

import numpy as np

from libs.State import State


class StateTracker:
    def __init__(self, keys: list[str]) -> None:
        self.history: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    def record_step(self, state: State, step_no: int) -> None:
        for key in state.keys:
            self.history[key].append(state[key].copy())

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
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file_path, **stacked_history)
            print(f"Simulation saved to: {file_path}")
        except OSError as e:
            raise RuntimeError(f"Failed to save history to {file_path}: {e}") from e