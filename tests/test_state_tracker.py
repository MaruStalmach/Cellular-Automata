from cellular_automata.Geometry import Geometry
from cellular_automata.State import State
from cellular_automata.tracking.StateTracker import StateTracker, TrackedStatePlayback

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

def _two_key_state():
    geometry = Geometry((2, 2, 2), axes="xyz", periodicity="")
    return State(geometry=geometry, random=False, cell_keys=["alive", "substrate"])


def test_tracker_records_only_its_keys():
    state = _two_key_state()
    tracker = StateTracker(["alive"])

    tracker.record_step(state, 0)

    assert list(tracker.history) == ["alive"]
    assert len(tracker.history["alive"]) == 1


def test_save_and_load_resolve_paths_the_same_way(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = _two_key_state()
    tracker = StateTracker(["alive"])
    tracker.record_step(state, 0)

    for name in [str(tmp_path / "absolute.npz"), "nested/run", "plain"]:
        tracker.save(name)
        assert len(StateTracker.load(name).history["alive"]) == 1


def test_sim_saves_partial_history_on_demand(tmp_path):
    from cellular_automata.rules.GameOfLife3D import GameOfLife3D
    from cellular_automata.Sim import CellularAutomaton

    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    output = tmp_path / "partial.npz"
    ca = CellularAutomaton(
        geometry,
        [GameOfLife3D(geometry, time_step=1)],
        max_steps=10,
        tracker=StateTracker(["alive"]),
        output_filename=str(output),
    )
    ca.step()

    ca.save_tracker()

    assert len(StateTracker.load(str(output)).history["alive"]) == 2
