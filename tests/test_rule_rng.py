"""Every built-in rule draws from its own seeded generator, never from np.random (REPRO-1)."""

import json
from pathlib import Path

import numpy as np
import pytest

from cellular_automata.config.Parser import parse_json
from cellular_automata.rules import Diffusion
from tests.rule_fixtures import RULE_IDS, SIZE, make_geometry, make_state, species_rules

LEGACY_FUNCTIONS = ["random", "rand", "randint", "choice", "shuffle", "permutation", "uniform"]


@pytest.fixture
def no_global_rng(monkeypatch):
    """Make any draw from the global legacy RNG fail the test."""

    def forbidden(name):
        def call(*args, **kwargs):
            raise AssertionError(f"rule called np.random.{name}; use self.rng")

        return call

    for name in LEGACY_FUNCTIONS:
        monkeypatch.setattr(np.random, name, forbidden(name))


@pytest.mark.parametrize("index", range(len(RULE_IDS)), ids=RULE_IDS)
def test_rule_does_not_draw_from_global_rng(index, no_global_rng):
    geometry = make_geometry()
    rule, keys, layers = species_rules(geometry)[index]
    state = make_state(geometry, keys, layers)

    rule.apply_state(state)


@pytest.mark.parametrize("index", range(len(RULE_IDS)), ids=RULE_IDS)
def test_rule_output_is_fixed_by_its_generator(index):
    def run(seed):
        geometry = make_geometry()
        rule, keys, layers = species_rules(geometry)[index]
        rule.rng = np.random.default_rng(seed)
        state = rule.apply_state(make_state(geometry, keys, layers))
        return {key: state[key].copy() for key in keys}

    first, second, other = run(3), run(3), run(4)

    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    # a fixture whose outcome doesn't depend on the draws would prove nothing
    assert any(not np.array_equal(first[key], other[key]) for key in first)


def test_diffusion_refill_does_not_draw_from_global_rng(no_global_rng):
    rule = Diffusion(make_geometry(), time_step=1, a=7.4, p=0.07)
    rule.lost_particles = rule.RESTORE_THRESHOLD

    rule._restore_random()

    assert int(rule.layers.sum()) == rule.RESTORE_THRESHOLD


def test_seeded_config_gives_identical_tracked_runs(tmp_path: Path):
    """Two runs of a tiny multi-rule config with the same seed record the same history."""
    species = ["A", "B"]
    config = {
        "sim": {"size": list(SIZE), "periodicity": "xy", "max_steps": 3, "render": False,
                "seed": 21},
        "state_keys": [
            {"name": "substrate", "layers": 6, "random": True,
             "random_func": "np.random.choice", "random_args": {"a": [0, 1], "p": [0.7, 0.3]}},
            {"name": "biofilm", "random": True, "random_func": "np.random.choice",
             "random_args": {"a": [0, 1], "p": [0.5, 0.5]}},
            *[
                {"name": sp, "random": True, "random_func": "random_seeds",
                 "random_args": {"n": 6, "value": 1}}
                for sp in species
            ],
        ],
        "rules": [
            {"name": "Diffusion", "args": {"time_step": 1, "a": 7.4, "p": 0.07,
                                           "biofilm_key": "biofilm"}},
            {"name": "Utilization", "args": {"time_step": 1, "bacteria_keys": species,
                                             "substrate_key": "substrate"}},
            {"name": "BacteriaGrowth", "args": {"time_step": 1, "bacteria_keys": species,
                                                "substrate_key": "substrate"}},
            {"name": "SpeciesInteraction", "args": {
                "time_step": 1, "interactions": {"A": {"B": -0.5}, "B": {"A": 0.5}}}},
            {"name": "BioreactorBiofilmDetachment", "args": {
                "time_step": 1, "biofilm_key": "biofilm", "detachment_rate": 0.3,
                "scaling": 1.0}},
            {"name": "BacteriaDecay", "args": {"time_step": 1, "target_keys": species,
                                               "decay_probability": 0.1}},
        ],
        "tracker": {"name": "StateTracker", "args": {}},
    }
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(config))

    def history(seed):
        config["sim"]["seed"] = seed
        path.write_text(json.dumps(config))
        sim, _ = parse_json(path)
        while sim.step() == 0:
            pass
        return sim.tracker.get_history()

    first, second = history(21), history(21)

    for key in first:
        np.testing.assert_array_equal(np.stack(first[key]), np.stack(second[key]))
