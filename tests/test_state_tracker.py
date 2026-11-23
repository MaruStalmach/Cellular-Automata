import numpy as np
import pytest

from cellular_automata.Geometry import Geometry
from cellular_automata.State import State
from cellular_automata.tracking.StateTracker import StateTracker, TrackedStatePlayback


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


# --- record_every (BUG-10) ---------------------------------------------------------


def _gol_sim(tracker, max_steps):
    from cellular_automata.rules import GameOfLife3D
    from cellular_automata.Sim import CellularAutomaton

    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    return CellularAutomaton(
        geometry, [GameOfLife3D(geometry, time_step=1)], tracker=tracker,
        max_steps=max_steps, seed=0, output_filename="run",
    )


def test_record_every_keeps_every_nth_step_and_the_last():
    tracker = StateTracker(["alive"], record_every=3)
    sim = _gol_sim(tracker, max_steps=7)

    while sim.step() == 0:
        pass

    assert tracker.steps == [0, 3, 6, 7]
    assert len(tracker.history["alive"]) == 4


def test_record_every_one_records_every_step():
    tracker = StateTracker(["alive"])
    sim = _gol_sim(tracker, max_steps=4)

    while sim.step() == 0:
        pass

    assert tracker.steps == [0, 1, 2, 3, 4]


def test_saving_an_interrupted_run_records_the_current_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = StateTracker(["alive"], record_every=10)
    sim = _gol_sim(tracker, max_steps=100)
    for _ in range(4):
        sim.step()

    sim.save_tracker()

    assert tracker.steps == [0, 4]
    np.testing.assert_array_equal(tracker.history["alive"][-1], sim.state["alive"])


def test_recorded_steps_survive_a_save_and_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = StateTracker(["alive"], record_every=2)
    sim = _gol_sim(tracker, max_steps=5)
    while sim.step() == 0:
        pass

    loaded = StateTracker.load("run")

    assert loaded.steps == [0, 2, 4, 5]
    assert loaded.record_every == 2


def test_files_without_recorded_steps_load_as_every_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.savez_compressed(tmp_path / "old.npz", alive=np.zeros((3, 2, 2, 2), dtype=np.uint8))

    loaded = StateTracker.load(str(tmp_path / "old.npz"))

    assert loaded.steps == [0, 1, 2]


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_record_every_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="record_every"):
        StateTracker(["alive"], record_every=value)
