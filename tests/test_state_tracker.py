from libs.Geometry import Geometry
from libs.State import State
from libs.tracking.StateTracker import StateTracker, TrackedStatePlayback

import numpy as np


def test_state_tracker_round_trip_and_playback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    geometry = Geometry((2, 2, 2), axes="xyz", periodicity="")
    state = State(
        geometry=geometry,
        random=False,
        cell_keys=["alive", "substrate"],
        key_layers=[1, 2],
    )

    state["alive"][...] = np.array(
        [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.uint8
    )
    state["substrate"][...] = np.array(
        [
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            [[[0, 1], [1, 0]], [[1, 0], [0, 1]]],
        ],
        dtype=np.uint8,
    )

    tracker = StateTracker(list(state.keys))
    tracker.record_step(state, 0)

    state["alive"][...] = 1 - state["alive"]
    state["substrate"][...] = 1 - state["substrate"]
    tracker.record_step(state, 1)

    tracker.save("demo")

    loaded = StateTracker.load("demo")

    restored = loaded.to_state(1, geometry=geometry)
    assert np.array_equal(restored["alive"], state["alive"])
    assert np.array_equal(restored["substrate"], state["substrate"])

    playback = TrackedStatePlayback(loaded, step_no=0, geometry=geometry)

    first_frame = playback(step=False)
    second_frame = playback()

    assert first_frame.shape == (2, 2, 2, 2)
    assert second_frame.shape == (2, 2, 2, 2)
    assert np.array_equal(first_frame[0], tracker.history["alive"][0])
    assert np.array_equal(second_frame[0], tracker.history["alive"][1])
    assert np.array_equal(second_frame[1], tracker.history["substrate"][1].max(axis=-1))