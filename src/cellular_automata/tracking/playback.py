"""Helpers for replaying tracked runs, shared by examples/Replay_sim.py and generate_gif.py."""

from collections.abc import Iterable

import numpy as np

from cellular_automata.State import State


def parse_keys(raw_keys: list[str] | None, available_keys: Iterable[str]) -> list[str]:
    """Split `-k a b,c` style arguments into key names; no arguments means every key."""
    if not raw_keys:
        return list(available_keys)

    parsed_keys: list[str] = []
    for item in raw_keys:
        parsed_keys.extend([key for key in item.split(",") if key])

    return parsed_keys


def frame_for_keys(state: State, keys: list[str], spatial_rank: int) -> np.ndarray:
    """One renderer frame: each key collapsed over its layers, stacked if there are several."""
    frames = []
    for key in keys:
        values = state[key]
        if values.ndim == spatial_rank + 1:
            values = values.max(axis=-1)
        frames.append(values)

    if len(frames) == 1:
        return frames[0]

    return np.stack(frames, axis=0)
