"""The gut Monte Carlo search (research/gut/scripts/mc_search.py) on tiny grids."""

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cellular_automata.Geometry import Geometry
from cellular_automata.rules import BiofilmDetachment
from cellular_automata.State import State

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "research" / "gut" / "scripts"
GUT_CONFIG = REPO_ROOT / "research" / "gut" / "configs" / "mc_search.json"


@pytest.fixture
def mc(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    sys.modules.pop("mc_search", None)
    return importlib.import_module("mc_search")


@pytest.fixture
def tiny_config(mc):
    """The real gut config on a 4x4x4 grid with 2 steps."""
    config = mc.load_config(GUT_CONFIG)
    config["sim"]["size"] = [4, 4, 4]
    config["sim"]["max_steps"] = 2
    config["mc_search"]["n_iterations"] = 3
    return config


def test_the_gut_config_is_valid_for_the_search(mc):
    config = mc.load_config(GUT_CONFIG)

    species = mc.validate_config(config)

    assert species == list(config["abundances"]["before"])


@pytest.mark.parametrize(
    "break_config, message",
    [
        (lambda c: c["abundances"]["after"].pop("other"), "different species"),
        (lambda c: c["abundances"]["before"].update(other=0.5), "sum to 1"),
        (lambda c: c["state_keys"].pop(), "missing"),
        (lambda c: c["rules"][0]["args"]["interactions"].pop("other"), "interactions name"),
        (lambda c: c["rules"].pop(1), "BiofilmDetachment"),
        (lambda c: c["mc_search"].update(density=0), "density"),
        (lambda c: c["mc_search"].update(interaction_bounds=[1, -1]), "interaction_bounds"),
        (lambda c: c["rules"][1]["args"]["species_keys"].pop(), "BiofilmDetachment species_keys"),
        (lambda c: c["rules"][2]["args"]["species_keys"].pop(), "GutDrift species_keys"),
    ],
)
def test_validate_config_rejects_inconsistent_configs(mc, tiny_config, break_config, message):
    break_config(tiny_config)

    with pytest.raises(ValueError, match=message):
        mc.validate_config(tiny_config)


def test_species_grid_has_the_requested_counts(mc):
    abundances = {"a": 0.5, "b": 0.3, "c": 0.2}

    labels = mc.build_species_grid((10, 10, 10), abundances, 0.7, np.random.default_rng(0))

    # 700 occupied: floor(350, 210, 140) leaves no remainder
    assert np.bincount(labels.ravel(), minlength=4).tolist() == [300, 350, 210, 140]


def test_species_grid_gives_the_rounding_remainder_to_the_largest_species(mc):
    abundances = {"a": 0.2, "b": 0.45, "c": 0.35}

    labels = mc.build_species_grid((3, 3, 3), abundances, 1.0, np.random.default_rng(0))

    # 27 cells: floor(5.4, 12.15, 9.45) = 5, 12, 9 and the 1 left over goes to b
    assert np.bincount(labels.ravel(), minlength=4).tolist() == [0, 5, 13, 9]


def test_species_grid_is_seeded(mc):
    abundances = {"a": 0.6, "b": 0.4}

    def grid(seed):
        return mc.build_species_grid((4, 4, 4), abundances, 0.5, np.random.default_rng(seed))

    assert np.array_equal(grid(1), grid(1))
    assert not np.array_equal(grid(1), grid(2))


def _detachment(size, detachment_rate):
    geometry = Geometry(size, axes="xyz", periodicity="")
    return BiofilmDetachment(geometry, 1.0, detachment_rate=detachment_rate, scaling=1.0)


def test_seed_biofilm_follows_the_rule_wall_axis(mc):
    # rate 1: the rule detaches nothing at y=0 and everything at the top of y
    detachment = _detachment((3, 4, 5), detachment_rate=1.0)
    occupied = np.ones((3, 4, 5), dtype=bool)
    occupied[0] = False

    biofilm = mc.seed_biofilm(occupied, detachment, np.random.default_rng(0))

    assert biofilm.dtype == np.uint8
    assert not biofilm[~occupied].any()
    assert biofilm[1:, 0, :].all()
    assert not biofilm[:, -1, :].any()


def test_seed_biofilm_attaches_every_occupied_cell_without_detachment(mc):
    detachment = _detachment((3, 3, 3), detachment_rate=0.0)
    occupied = np.random.default_rng(0).random((3, 3, 3)) < 0.5

    biofilm = mc.seed_biofilm(occupied, detachment, np.random.default_rng(0))

    assert np.array_equal(biofilm.astype(bool), occupied)


def test_relative_abundances_are_shares_of_the_occupied_cells(mc):
    state = State(Geometry((2, 2, 1), axes="xyz", periodicity=""), cell_keys=["a", "b"])
    state["a"] = np.array([[[1], [1]], [[1], [0]]], dtype=np.uint8)
    state["b"] = np.array([[[0], [0]], [[0], [1]]], dtype=np.uint8)

    assert mc.relative_abundances(state, ["a", "b"]) == {"a": 0.75, "b": 0.25}


def test_relative_abundances_of_an_empty_grid_are_zero(mc):
    state = State(Geometry((2, 2, 1), axes="xyz", periodicity=""), cell_keys=["a"])

    assert mc.relative_abundances(state, ["a"]) == {"a": 0.0}


def test_abundance_error_is_the_sum_of_squared_differences(mc):
    target = {"a": 0.5, "b": 0.5}

    assert mc.abundance_error(target, target) == 0.0
    assert mc.abundance_error({"a": 0.8, "b": 0.2}, target) == pytest.approx(0.18)


def test_unchanged_abundances_score_zero_whatever_the_density(mc, tiny_config):
    # every coefficient 0, no detachment and no drift: nothing changes, so the
    # final shares equal the initial ones (the old score compared shares of all
    # cells, which only matched at density 1)
    config = copy.deepcopy(tiny_config)
    abundances = {sp: 0.0 for sp in config["abundances"]["before"]}
    abundances.update(BBBBB=0.75, other=0.25)
    config["abundances"] = {"before": abundances, "after": abundances}
    config["sim"]["size"] = [4, 4, 4]
    config["mc_search"]["density"] = 0.5
    for rule in config["rules"]:
        if rule["name"] == "BiofilmDetachment":
            rule["args"]["detachment_rate"] = 0.0
        if rule["name"] == "GutDrift":
            rule["args"]["drift_speed"] = 0
    species = mc.validate_config(config)
    initial = mc.build_initial_state(config, species, np.random.default_rng(0))
    zeros = {a: {b: 0.0 for b in species if b != a} for a in species}

    outcome = mc.evaluate_guess(config, initial, zeros, seed=1)

    assert outcome["error"] == 0.0
    assert outcome["occupied_cells"] == 32
    assert outcome["final_abundances"]["BBBBB"] == 0.75


def test_a_guess_runs_with_its_own_coefficients_and_seed(mc, tiny_config, monkeypatch):
    species = mc.validate_config(tiny_config)
    initial = mc.build_initial_state(tiny_config, species, np.random.default_rng(0))
    interactions = mc.random_interactions(species, (-1.0, 1.0), np.random.default_rng(1))
    built = []
    real_build = mc.build_simulation

    def record_build(data):
        built.append(data)
        return real_build(data)

    monkeypatch.setattr(mc, "build_simulation", record_build)
    sim = mc.run_guess(tiny_config, initial, interactions, seed=7)

    (data,) = built
    rule = next(r for r in data["rules"] if r["name"] == "SpeciesInteraction")
    assert rule["args"]["interactions"] == interactions
    assert data["sim"]["seed"] == 7 and sim.seed == 7
    assert "tracker" not in data and data["sim"]["render"] is False
    assert sim.step_no == tiny_config["sim"]["max_steps"]
    # the caller's config is untouched
    assert tiny_config["rules"][0]["args"]["interactions"] != interactions


def test_apply_initial_state_writes_species_biofilm_and_clears_floating(mc, tiny_config):
    species = mc.validate_config(tiny_config)
    initial = mc.build_initial_state(tiny_config, species, np.random.default_rng(0))
    sim = mc.build_simulation(mc.config_for_guess(tiny_config, {}, seed=0))
    sim.state["floating_bacteria"] = np.ones((4, 4, 4), dtype=np.uint8)

    mc.apply_initial_state(sim.state, initial)

    for label, sp in enumerate(species, start=1):
        assert np.array_equal(sim.state[sp], initial.labels == label)
    assert np.array_equal(sim.state["biofilm"], initial.biofilm)
    assert not sim.state["floating_bacteria"].any()


def test_guesses_get_distinct_seeds_and_repeat_with_the_same_seed(mc):
    species = ["a", "b", "c"]

    _, first = mc.draw_guesses(species, seed=5, n_iterations=4, bounds=(-1.0, 1.0))
    _, again = mc.draw_guesses(species, seed=5, n_iterations=4, bounds=(-1.0, 1.0))
    _, other = mc.draw_guesses(species, seed=6, n_iterations=4, bounds=(-1.0, 1.0))

    assert first == again
    assert first != other
    seeds = [seed for _, seed in first]
    assert len(set(seeds)) == 4 and all(0 <= s < 2**63 for s in seeds)
    coefficients = [
        c for interactions, _ in first for row in interactions.values() for c in row.values()
    ]
    assert all(-1.0 <= c <= 1.0 for c in coefficients)


def test_search_scores_every_guess_and_repeats_with_the_same_seed(mc, tiny_config, tmp_path):
    output = tmp_path / "results" / "mc.json"

    results = mc.run_search(tiny_config, output, workers=2, seed=11)
    repeat = mc.run_search(tiny_config, tmp_path / "repeat.json", workers=1, seed=11)

    assert json.loads(output.read_text()) == results
    assert results["complete"] and results["n_scored"] == 3 and results["seed"] == 11
    assert [row["index"] for row in results["guesses"]] == [0, 1, 2]
    assert results["best"] == min(results["guesses"], key=lambda row: row["error"])
    assert results["config"]["sim"]["seed"] == 11
    assert [r["error"] for r in repeat["guesses"]] == [r["error"] for r in results["guesses"]]


def test_a_failing_guess_still_writes_the_partial_results(mc, tiny_config, tmp_path):
    output = tmp_path / "mc.json"
    # builds fine, then raises in the worker on the first step
    tiny_config["state_keys"].append({"name": "marker", "layers": 2})
    tiny_config["rules"].append(
        {
            "name": "Inoculation",
            "args": {
                "time_step": 1.0,
                "target_key": "marker",
                "location": [0, 0, 0],
                "size": [1, 1, 1],
                "layer": 5,
            },
        }
    )

    with pytest.raises(ValueError, match="layer 5"):
        mc.run_search(tiny_config, output, workers=1, seed=3)

    written = json.loads(output.read_text())
    assert written["complete"] is False


def test_the_iterations_override_counts_before_validation(mc, tiny_config, tmp_path):
    del tiny_config["mc_search"]["n_iterations"]

    results = mc.run_search(tiny_config, tmp_path / "mc.json", n_iterations=1, workers=1, seed=2)

    assert results["n_scored"] == 1


def test_initial_state_has_the_configured_density(mc, tiny_config):
    tiny_config["sim"]["size"] = [5, 5, 4]
    tiny_config["sim"]["seed"] = 4
    tiny_config["mc_search"]["density"] = 0.3

    initial = mc.initial_state_for(tiny_config)

    assert np.count_nonzero(initial.labels) == int(100 * 0.3)
    assert not initial.biofilm[initial.labels == 0].any()


def test_a_saved_guess_replays_to_the_same_error(mc, tiny_config, tmp_path):
    output = tmp_path / "mc.json"
    mc.run_search(tiny_config, output, workers=2, seed=9)
    results = json.loads(output.read_text())
    best = results["best"]

    sim = mc.replay_guess(results, best["index"])

    species = list(results["config"]["abundances"]["before"])
    final = mc.relative_abundances(sim.state, species)
    assert final == best["final_abundances"]
    assert mc.abundance_error(final, results["config"]["abundances"]["after"]) == best["error"]


def test_the_search_checkpoints_while_it_runs(mc, tiny_config, tmp_path, monkeypatch):
    tiny_config["mc_search"]["checkpoint_every_s"] = -1  # after every result
    written = []
    real_write = mc._write_results

    def record_write(path, results):
        written.append((results["n_scored"], results["complete"]))
        real_write(path, results)

    monkeypatch.setattr(mc, "_write_results", record_write)
    mc.run_search(tiny_config, tmp_path / "mc.json", workers=1, seed=5)

    checkpoints, final = written[:-1], written[-1]
    assert checkpoints and all(complete is False for _, complete in checkpoints)
    assert [n for n, _ in checkpoints] == sorted(n for n, _ in checkpoints)
    assert final == (3, True)


def test_an_interrupted_search_writes_what_it_scored(mc, tiny_config, tmp_path, monkeypatch):
    output = tmp_path / "mc.json"
    real_info = mc.logger.info

    def interrupt_after_first_result(message, *args):
        real_info(message, *args)
        if message.startswith("[%d/%d]"):  # logged right after a guess is scored
            raise KeyboardInterrupt

    monkeypatch.setattr(mc.logger, "info", interrupt_after_first_result)
    try:
        results = mc.run_search(tiny_config, output, workers=1, seed=6)
    except KeyboardInterrupt:
        pytest.fail("run_search let the KeyboardInterrupt escape")

    written = json.loads(output.read_text())
    assert written == results
    assert written["complete"] is False and written["n_scored"] == 1
