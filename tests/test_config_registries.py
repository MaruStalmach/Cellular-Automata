"""Trackers, callbacks, initialisers and dtypes are resolved explicitly (ARCH-1)."""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from cellular_automata import Callback as callback_module
from cellular_automata.config import Parser
from cellular_automata.registry import CALLBACKS, INITIALISERS, TRACKERS
from cellular_automata.tracking.StateTracker import StateTracker
from cellular_automata.util import helper_functions


class FakeRenderer:
    def __init__(self, init_state, *window_size, update_callback=None, keys_to_render=1):
        self.update_callback = update_callback


@pytest.fixture(autouse=True)
def no_opengl(monkeypatch):
    monkeypatch.setattr(Parser, "CARenderer", FakeRenderer)


def _write(tmp_path: Path, state_key=None, tracker=None, callback=None) -> Path:
    config = {
        "sim": {"size": [3, 3, 3], "periodicity": "", "max_steps": 1,
                "render": callback is not None, "seed": 1},
        "state_keys": [state_key or {"name": "alive"}],
        "rules": [{"name": "GameOfLife3D", "args": {"time_step": 1}}],
    }
    if tracker is not None:
        config["tracker"] = tracker
    if callback is not None:
        config["renderer"] = {"key": "alive", "callback": callback}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def _defined_in(module, kind):
    return {
        name: obj
        for name, obj in vars(module).items()
        if kind(obj) and getattr(obj, "__module__", None) == module.__name__
    }


def test_built_in_trackers_callbacks_and_initialisers_are_registered():
    assert TRACKERS.get("StateTracker") is StateTracker
    callbacks = _defined_in(callback_module, inspect.isclass)
    initialisers = _defined_in(helper_functions, inspect.isfunction)

    assert {"Callback", "MultiKeyCallback", "ReduceCallback"} <= set(callbacks)
    assert {name: CALLBACKS.get(name) for name in callbacks} == callbacks
    assert {"init_block", "random_seeds"} <= set(initialisers)
    assert {name: INITIALISERS.get(name) for name in initialisers} == initialisers


@pytest.mark.parametrize("name", ["NoSuchTracker", "Path", "State"])
def test_unknown_tracker_is_rejected(tmp_path: Path, name):
    with pytest.raises(ValueError, match=f"Unknown tracker '{name}'"):
        Parser.parse_json(_write(tmp_path, tracker={"name": name, "args": {}}))


@pytest.mark.parametrize("name", ["NoSuchCallback", "CARenderer", "parse_json"])
def test_unknown_callback_is_rejected(tmp_path: Path, name):
    with pytest.raises(ValueError, match=f"Unknown callback '{name}'"):
        Parser.parse_json(_write(tmp_path, callback={"name": name}))


def test_registered_callback_is_built(tmp_path: Path):
    # ReduceCallback reduces over layers, so the key needs more than one
    layered = {"name": "alive", "layers": 2}
    _, renderer = Parser.parse_json(
        _write(tmp_path, state_key=layered, callback={"name": "ReduceCallback", "args": {}})
    )

    assert type(renderer.update_callback) is callback_module.ReduceCallback


@pytest.mark.parametrize("name", ["getnestedattr", "parse_json", "Path", "json.load"])
def test_unknown_random_func_is_rejected(tmp_path: Path, name):
    key = {"name": "alive", "random": True, "random_func": name}
    with pytest.raises(ValueError, match=f"Unknown initialiser '{name}'"):
        Parser.parse_json(_write(tmp_path, state_key=key))


@pytest.mark.parametrize("dtype", ["np.float32", "float32"])
def test_dtype_names_with_or_without_np_prefix(tmp_path: Path, dtype):
    sim, _ = Parser.parse_json(_write(tmp_path, state_key={"name": "alive", "dtype": dtype}))

    assert sim.state["alive"].dtype == np.float32


def test_bare_dtype_name_is_not_silently_float(tmp_path: Path):
    """"uint8" without the np. prefix used to fall back to float64 arrays."""
    sim, _ = Parser.parse_json(_write(tmp_path, state_key={"name": "alive", "dtype": "uint8"}))

    assert sim.state["alive"].dtype == np.uint8


def test_sim_dtype_is_the_default_for_keys(tmp_path: Path):
    path = _write(tmp_path)
    config = json.loads(path.read_text())
    # not the parser's own default (np.uint8), so the test shows sim.dtype is read
    config["sim"]["dtype"] = "float32"
    path.write_text(json.dumps(config))

    sim, _ = Parser.parse_json(path)

    assert sim.state["alive"].dtype == np.float32


@pytest.mark.parametrize("dtype", ["np.random", "Path", "np.notatype", "object", "U10", 8, None])
def test_unknown_dtype_is_rejected(tmp_path: Path, dtype):
    with pytest.raises(ValueError, match=f"Unknown dtype '?{dtype}'?"):
        Parser.parse_json(_write(tmp_path, state_key={"name": "alive", "dtype": dtype}))
