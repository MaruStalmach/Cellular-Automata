"""Seeding: a seed fixes every random draw in a simulation (REPRO-1)."""

import json
from pathlib import Path

import numpy as np
import pytest

from cellular_automata.config.Parser import parse_json
from cellular_automata.Geometry import Geometry
from cellular_automata.Rule import Rule
from cellular_automata.Sim import CellularAutomaton
from cellular_automata.State import State
from cellular_automata.util.helper_functions import random_seeds


class _DrawingRule(Rule):
    """Records one draw from its generator per application."""

    def __init__(self, geometry, time_step=1):
        super().__init__(geometry, time_step)
        self.draws: list[float] = []

    def apply_state(self, state):
        self.draws.append(float(self.rng.random()))
        return state


def _geometry():
    return Geometry((4, 4, 4), axes="xyz", periodicity="")


def _empty_state(geometry):
    return State(geometry, cell_keys=["alive"])


def _sim(seed, n_rules=2, steps=3):
    geometry = _geometry()
    rules = [_DrawingRule(geometry) for _ in range(n_rules)]
    sim = CellularAutomaton(
        geometry, rules, state=_empty_state(geometry), max_steps=steps, seed=seed
    )
    for _ in range(steps):
        sim.step()
    return rules


def test_rule_has_a_generator_without_a_simulation():
    assert isinstance(_DrawingRule(_geometry()).rng, np.random.Generator)


def test_same_seed_gives_same_draws():
    first = _sim(seed=7)
    second = _sim(seed=7)

    assert [r.draws for r in first] == [r.draws for r in second]


def test_different_seeds_give_different_draws():
    assert _sim(seed=7)[0].draws != _sim(seed=8)[0].draws


def test_each_rule_gets_its_own_stream():
    first, second = _sim(seed=7)

    assert first.draws != second.draws


def test_adding_a_rule_does_not_change_earlier_rules_streams():
    assert _sim(seed=7, n_rules=1)[0].draws == _sim(seed=7, n_rules=2)[0].draws


def test_unseeded_simulation_records_a_seed_that_reproduces_it():
    geometry = _geometry()
    rules = [_DrawingRule(geometry) for _ in range(2)]
    sim = CellularAutomaton(
        geometry, rules, state=_empty_state(geometry), max_steps=3, seed=None
    )
    for _ in range(3):
        sim.step()

    replayed = _sim(seed=sim.seed)

    assert isinstance(sim.seed, int)
    assert [r.draws for r in replayed] == [r.draws for r in rules]


def test_auto_created_state_is_seeded():
    geometry = _geometry()

    def build(seed):
        rule = _DrawingRule(geometry)
        rule.required_keys = ["alive"]
        return CellularAutomaton(geometry, [rule], seed=seed).state["alive"]

    np.testing.assert_array_equal(build(3), build(3))
    assert not np.array_equal(build(3), build(4))


def test_state_default_fill_uses_given_generator():
    geometry = _geometry()

    def build(seed):
        rng = np.random.default_rng(seed)
        return State(geometry, cell_keys=["alive"], random=True, rng=rng)["alive"]

    np.testing.assert_array_equal(build(1), build(1))
    assert not np.array_equal(build(1), build(2))


def test_random_seeds_uses_given_generator():
    def build(seed):
        return random_seeds((5, 5, 5), n=4, value=1, rng=np.random.default_rng(seed))

    np.testing.assert_array_equal(build(1), build(1))
    assert not np.array_equal(build(1), build(2))


def _write_config(tmp_path: Path, seed=None, random_func="np.random.choice") -> Path:
    sim = {"size": [6, 6, 6], "periodicity": "", "max_steps": 1, "render": False}
    if seed is not None:
        sim["seed"] = seed
    random_args = (
        {"a": [0, 1], "p": [0.5, 0.5]}
        if random_func == "np.random.choice"
        else {"n": 5, "value": 1}
    )
    config = {
        "sim": sim,
        "state_keys": [
            {
                "name": "alive",
                "random": True,
                "random_func": random_func,
                "random_args": random_args,
            }
        ],
        "rules": [{"name": "GameOfLife3D", "args": {"time_step": 1}}],
    }
    path = tmp_path / f"config_{seed}_{random_func}.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("random_func", ["np.random.choice", "random_seeds"])
def test_config_seed_fixes_initial_state(tmp_path: Path, random_func):
    def build(seed):
        sim, _ = parse_json(_write_config(tmp_path, seed, random_func))
        return sim.state["alive"]

    np.testing.assert_array_equal(build(11), build(11))
    assert not np.array_equal(build(11), build(12))


def test_config_seed_fixes_rule_streams(tmp_path: Path):
    def draws(seed):
        sim, _ = parse_json(_write_config(tmp_path, seed))
        return [float(rule.rng.random()) for rule in sim.rules]

    assert draws(11) == draws(11)
    assert draws(11) != draws(12)


def test_unseeded_config_reports_the_seed_it_used(tmp_path: Path, capsys):
    sim, _ = parse_json(_write_config(tmp_path, seed=None))

    replay, _ = parse_json(_write_config(tmp_path, seed=sim.seed))

    assert str(sim.seed) in capsys.readouterr().out
    np.testing.assert_array_equal(sim.state["alive"], replay.state["alive"])


def test_legacy_only_numpy_random_function_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="np.random.randint"):
        parse_json(_write_config(tmp_path, seed=1, random_func="np.random.randint"))


def test_tracker_output_records_the_seed(tmp_path: Path):
    from cellular_automata.tracking.StateTracker import StateTracker

    geometry = _geometry()
    tracker = StateTracker(keys=["alive"])
    sim = CellularAutomaton(
        geometry, [_DrawingRule(geometry)], state=_empty_state(geometry), tracker=tracker,
        seed=5,
    )
    out = tmp_path / "run.npz"
    sim.tracker.save(str(out))

    with np.load(out) as saved:
        metadata = json.loads(str(saved["__state_tracker_metadata__"].item()))

    assert metadata["seed"] == 5
