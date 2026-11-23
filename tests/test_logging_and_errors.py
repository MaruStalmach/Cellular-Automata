"""Library errors propagate instead of being printed, and messages go through logging (HYG-4)."""

import ast
import logging
from pathlib import Path

import numpy as np
import pytest

from cellular_automata.Callback import Callback, MultiKeyCallback, ReduceCallback
from cellular_automata.config.Parser import build_simulation
from cellular_automata.Geometry import Geometry
from cellular_automata.rules import GameOfLife3D
from cellular_automata.Sim import CellularAutomaton
from cellular_automata.State import State

SRC = Path(__file__).resolve().parents[1] / "src" / "cellular_automata"


def _sim(max_steps=1):
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    state = State(geometry, cell_keys=["alive", "s"], key_layers=[1, 4])
    return CellularAutomaton(
        geometry, [GameOfLife3D(geometry, time_step=1)], state=state, seed=1,
        max_steps=max_steps,
    )


def test_library_does_not_print():
    offenders = [
        f"{path.relative_to(SRC)}:{node.lineno}"
        for path in SRC.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
    ]

    assert offenders == []


@pytest.mark.parametrize(
    "make",
    [
        lambda sim: Callback("missing", sim),
        lambda sim: ReduceCallback("missing", sim),
        lambda sim: MultiKeyCallback(["alive", "missing"], sim),
    ],
    ids=["Callback", "ReduceCallback", "MultiKeyCallback"],
)
def test_callback_with_an_unknown_key_raises_when_built(make):
    with pytest.raises(ValueError, match="'missing'.*Known keys"):
        make(_sim())


def test_callback_errors_propagate():
    callback = MultiKeyCallback(["alive", "s"], _sim())

    def broken(step=True):
        raise RuntimeError("broken callback")

    callback.callbacks[1] = broken

    with pytest.raises(RuntimeError, match="broken callback"):
        callback(step=False)


def test_multi_key_callback_stacks_each_key():
    frame = MultiKeyCallback(["alive", "s"], _sim())(step=False)

    assert frame.shape == (2, 3, 3, 3)


def test_end_of_run_is_logged_not_printed(caplog, capsys):
    sim = _sim(max_steps=1)
    sim.step()

    with caplog.at_level(logging.INFO, logger="cellular_automata.Sim"):
        assert sim.step() == -1

    assert "simulation ended" in caplog.text
    assert capsys.readouterr().out == ""


def test_config_without_tracker_is_logged(caplog, capsys):
    config = {
        "sim": {"size": [3, 3, 3], "periodicity": "", "max_steps": 1, "seed": 0},
        "state_keys": [{"name": "alive"}],
        "rules": [{"name": "GameOfLife3D", "args": {"time_step": 1}}],
    }

    with caplog.at_level(logging.INFO, logger="cellular_automata.config.Parser"):
        build_simulation(config)

    assert "no 'tracker' block" in caplog.text
    assert capsys.readouterr().out == ""


def test_multi_key_callback_keeps_the_state_dtype():
    assert MultiKeyCallback(["alive"], _sim())(step=False).dtype == np.uint8
