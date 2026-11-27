"""The single-run gut script (research/gut/scripts/run_gut.py) on tiny grids."""

import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

from cellular_automata.Geometry import Geometry
from cellular_automata.rules import BiofilmDetachment
from cellular_automata.tracking.StateTracker import StateTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "research" / "gut" / "scripts"
GUT_CONFIG = REPO_ROOT / "research" / "gut" / "configs" / "gut.json"


@pytest.fixture
def gut(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    sys.modules.pop("run_gut", None)
    return importlib.import_module("run_gut")


@pytest.fixture
def tiny_config(gut):
    """The real gut config on an 8x8x4 grid with 2 steps and a thin mat."""
    config = gut.load_config(GUT_CONFIG)
    config["sim"]["size"] = [8, 8, 4]
    config["sim"]["max_steps"] = 2
    config["initial_state"].update(mean_thickness=2, thickness_spread=1, patch_size=2)
    config["initial_state"]["lumen_density"] = 0.2
    return config


def _args(config, rule_name):
    return next(rule["args"] for rule in config["rules"] if rule["name"] == rule_name)


def _detachment(size=(6, 8, 5), walls=None, periodicity=""):
    geometry = Geometry(size, axes="xyz", periodicity=periodicity)
    return BiofilmDetachment(geometry, 1, detachment_rate=0, scaling=1, walls=walls)


# --- config -------------------------------------------------------------------


def test_the_gut_config_is_valid(gut):
    config = gut.load_config(GUT_CONFIG)

    assert gut.validate_config(config) == list(config["abundances"]["before"])


@pytest.mark.parametrize(
    "break_config, message",
    [
        (lambda c: c["abundances"]["after"].pop("other"), "different species"),
        (lambda c: c["abundances"]["before"].update(other=0.5), "sum to 1"),
        (lambda c: c["state_keys"].pop(), "missing"),
        (lambda c: c["abundances"]["before"].update(BBBBB=-0.1), "negative"),
        (lambda c: _args(c, "SpeciesInteraction")["interactions"].pop("other"), "interactions"),
        (
            lambda c: _args(c, "BiofilmDetachment")["species_keys"].pop(),
            "BiofilmDetachment species_keys",
        ),
        (lambda c: _args(c, "GutDrift")["species_keys"].pop(), "GutDrift species_keys"),
        (lambda c: c["initial_state"].pop("patch_size"), "initial_state.patch_size"),
        (lambda c: c["initial_state"].update(patch_size=0), "patch_size must be > 0"),
        (lambda c: c["initial_state"].update(mean_thickness=-1), ">= 0"),
        (lambda c: c["initial_state"].update(lumen_density=1.5), "lumen_density"),
    ],
)
def test_validate_config_rejects_inconsistent_configs(gut, tiny_config, break_config, message):
    break_config(tiny_config)

    with pytest.raises(ValueError, match=message):
        gut.validate_config(tiny_config)


# --- the mat ---------------------------------------------------------------------


def test_mat_heights_have_the_requested_mean_and_spread(gut):
    heights = gut.mat_heights((60, 60), [False, False], 10, 3, 2, np.random.default_rng(0))

    assert heights.min() >= 0
    assert heights.mean() == pytest.approx(10, abs=0.5)
    assert heights.std() == pytest.approx(3, abs=0.5)


def test_mat_heights_are_patchy_not_independent(gut):
    rng = np.random.default_rng(0)
    smooth = gut.mat_heights((60, 60), [False, False], 10, 3, 16, rng).astype(float)

    # neighbouring columns have similar heights when patch_size is large
    neighbour_correlation = np.corrcoef(smooth[:-1].ravel(), smooth[1:].ravel())[0, 1]
    assert neighbour_correlation > 0.8


def test_without_spread_the_mat_is_flat(gut):
    heights = gut.mat_heights((5, 5), [False, False], 3, 0, 2, np.random.default_rng(0))

    assert (heights == 3).all()


def test_the_mat_grows_from_the_floor_in_solid_columns(gut):
    detachment = _detachment()
    initial = {"mean_thickness": 3, "thickness_spread": 2, "patch_size": 1}

    mat = gut.build_mat(detachment, initial, np.random.default_rng(1))

    heights = mat.sum(axis=1)
    for x, z in np.ndindex(6, 5):
        assert mat[x, : heights[x, z], z].all() and not mat[x, heights[x, z] :, z].any()
    assert np.array_equal(detachment._linked_to_wall(mat), mat)


def test_a_mat_grows_from_every_wall(gut):
    detachment = _detachment(size=(4, 10, 4), walls=["y-", "y+"])
    initial = {"mean_thickness": 2, "thickness_spread": 0, "patch_size": 1}

    mat = gut.build_mat(detachment, initial, np.random.default_rng(0))

    assert mat[:, [0, 1, 8, 9], :].all() and not mat[:, 2:8, :].any()


# --- species and lumen --------------------------------------------------------------


def test_species_have_the_requested_counts_among_the_occupied_cells(gut):
    occupied = np.zeros((10, 10, 10), dtype=bool)
    occupied[:, :7, :] = True  # 700 cells

    labels = gut.assign_species(occupied, {"a": 0.5, "b": 0.3, "c": 0.2}, np.random.default_rng(0))

    assert np.bincount(labels.ravel(), minlength=4).tolist() == [300, 350, 210, 140]
    assert not labels[~occupied].any()


def test_the_rounding_remainder_goes_to_the_largest_species(gut):
    occupied = np.ones((3, 3, 3), dtype=bool)

    labels = gut.assign_species(
        occupied, {"a": 0.2, "b": 0.45, "c": 0.35}, np.random.default_rng(0)
    )

    # 27 cells: floor(5.4, 12.15, 9.45) = 5, 12, 9 and the 1 left over goes to b
    assert np.bincount(labels.ravel(), minlength=4).tolist() == [0, 5, 13, 9]


def test_the_starting_grid_is_a_linked_mat_plus_floating_lumen_cells(gut, tiny_config):
    sim, config, species = gut.prepare(tiny_config, seed=3)
    state = sim.state
    biofilm = state["biofilm"].astype(bool)
    floating = state["floating_bacteria"].astype(bool)
    occupied = np.zeros(biofilm.shape, dtype=bool)
    for sp in species:
        assert not (occupied & state[sp].astype(bool)).any()  # one species per cell
        occupied |= state[sp].astype(bool)

    assert biofilm.any() and floating.any()
    assert not (biofilm & floating).any()
    assert np.array_equal(occupied, biofilm | floating)
    detachment = next(r for r in sim.rules if isinstance(r, BiofilmDetachment))
    assert np.array_equal(detachment._linked_to_wall(biofilm), biofilm)
    assert floating.sum() / (~biofilm).sum() == pytest.approx(0.2, abs=0.1)


def test_the_starting_grid_is_seeded(gut, tiny_config):
    def grid(seed):
        sim, _, _ = gut.prepare(tiny_config, seed=seed)
        return {key: sim.state[key].copy() for key in sim.state.keys}

    first, again, other = grid(5), grid(5), grid(6)

    assert all(np.array_equal(first[key], again[key]) for key in first)
    assert any(not np.array_equal(first[key], other[key]) for key in first)


# --- a run -----------------------------------------------------------------------------


def test_an_unchanged_community_scores_zero(gut, tiny_config):
    # no interactions, detachment or drift: the shares stay the 'before' ones
    abundances = {sp: 0.0 for sp in tiny_config["abundances"]["before"]}
    abundances.update(BBBBB=0.75, other=0.25)
    tiny_config["abundances"] = {"before": abundances, "after": abundances}
    _args(tiny_config, "BiofilmDetachment")["detachment_rate"] = 0.0
    _args(tiny_config, "GutDrift")["drift_speed"] = 0

    summary = gut.run(tiny_config, seed=1)

    # the starting shares round to whole cells, so compare with them, not with 0.75
    # (lumen cells resting on the mat attach, so only the biofilm/floating split changes)
    assert summary["final"]["abundances"] == summary["start"]["abundances"]
    assert summary["error_vs_after"] == pytest.approx(0.0, abs=1e-3)


def test_a_run_reports_and_records_the_filled_starting_grid(gut, tiny_config, tmp_path):
    tracker_path = tmp_path / "run.npz"

    summary = gut.run(tiny_config, seed=4, tracker_path=tracker_path)

    assert summary["seed"] == 4 and summary["steps"] == 2
    assert summary["config"]["sim"]["seed"] == 4
    assert sum(summary["start"]["abundances"].values()) == pytest.approx(1.0)
    tracker = StateTracker.load(str(tracker_path))
    # the config records every 5th step, and always the last one
    assert tracker.steps == [0, 2]
    start_biofilm = tracker.history["biofilm"][0]
    assert start_biofilm.sum() == summary["start"]["biofilm_cells"] > 0


def test_main_writes_the_history_and_a_summary(gut, tiny_config, tmp_path, monkeypatch):
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(tiny_config))
    monkeypatch.setattr(gut, "RESULTS_DIR", tmp_path / "results")

    gut.main(["--config", str(config_path), "--seed", "2", "--name", "first"])

    summary = json.loads((tmp_path / "results" / "first.json").read_text())
    assert summary["seed"] == 2
    assert (tmp_path / "results" / "first.npz").exists()


def test_main_does_not_overwrite_earlier_results(gut, tiny_config, tmp_path, monkeypatch):
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(tiny_config))
    monkeypatch.setattr(gut, "RESULTS_DIR", tmp_path / "results")
    gut.main(["--config", str(config_path), "--seed", "2", "--name", "run"])
    first = (tmp_path / "results" / "run.json").read_text()

    with pytest.raises(SystemExit, match="already exists"):
        gut.main(["--config", str(config_path), "--seed", "3", "--name", "run"])
    assert (tmp_path / "results" / "run.json").read_text() == first

    gut.main(["--config", str(config_path), "--seed", "3", "--name", "run", "--overwrite"])
    assert json.loads((tmp_path / "results" / "run.json").read_text())["seed"] == 3


def test_default_names_include_the_time(gut, tiny_config, tmp_path, monkeypatch):
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(tiny_config))
    monkeypatch.setattr(gut, "RESULTS_DIR", tmp_path / "results")

    gut.main(["--config", str(config_path), "--seed", "2"])

    (summary,) = (tmp_path / "results").glob("tiny_2_*.json")
    assert summary.with_suffix(".npz").exists()


def test_abundance_error_is_the_sum_of_squared_differences(gut):
    target = {"a": 0.5, "b": 0.5}

    assert gut.abundance_error(target, target) == 0.0
    assert gut.abundance_error({"a": 0.8, "b": 0.2}, target) == pytest.approx(0.18)


def test_errors_are_against_after_and_before(gut, tiny_config):
    before = {sp: 0.0 for sp in tiny_config["abundances"]["before"]}
    before.update(BBBBB=0.75, other=0.25)
    after = {sp: 0.0 for sp in before}
    after.update(BBBBB=0.25, other=0.75)
    tiny_config["abundances"] = {"before": before, "after": after}
    _args(tiny_config, "BiofilmDetachment")["detachment_rate"] = 0.0
    _args(tiny_config, "GutDrift")["drift_speed"] = 0

    summary = gut.run(tiny_config, seed=1)

    # nothing changes, so the final shares are the starting ('before') ones
    final = summary["final"]["abundances"]
    # (lumen cells resting on the mat attach, so only the biofilm/floating split changes)
    assert summary["final"]["abundances"] == summary["start"]["abundances"]
    assert summary["error_vs_before"] == gut.abundance_error(final, before)
    assert summary["error_vs_after"] == gut.abundance_error(final, after)
    assert summary["error_vs_after"] == pytest.approx(2 * 0.5**2, abs=1e-2)


def test_the_final_composition_comes_from_the_simulation(gut, tiny_config):
    # certain detachment: the mat erodes from the top and released cells float
    _args(tiny_config, "BiofilmDetachment")["detachment_rate"] = 1e6
    _args(tiny_config, "GutDrift")["drift_speed"] = 0

    summary = gut.run(tiny_config, seed=1)

    assert summary["final"]["biofilm_cells"] < summary["start"]["biofilm_cells"]
    assert summary["final"]["floating_cells"] > summary["start"]["floating_cells"]


def test_cells_spawned_by_interactions_are_counted_as_unflagged(gut, tiny_config):
    # every species strongly favours every other, so empty cells get spawned
    interactions = _args(tiny_config, "SpeciesInteraction")["interactions"]
    for row in interactions.values():
        for other in row:
            row[other] = 1.0
    _args(tiny_config, "BiofilmDetachment")["detachment_rate"] = 0.0
    _args(tiny_config, "GutDrift")["drift_speed"] = 0

    summary = gut.run(tiny_config, seed=1)

    assert summary["start"]["unflagged_cells"] == 0
    final = summary["final"]
    assert final["unflagged_cells"] > 0
    assert (
        final["occupied_cells"]
        == final["biofilm_cells"] + final["floating_cells"] + final["unflagged_cells"]
    )


def test_a_whole_run_repeats_with_the_same_seed(gut, tiny_config, tmp_path):
    _args(tiny_config, "BiofilmDetachment")["detachment_rate"] = 10.0
    interactions = _args(tiny_config, "SpeciesInteraction")["interactions"]
    rng = np.random.default_rng(0)
    for row in interactions.values():
        for other in row:
            row[other] = float(rng.uniform(-1, 1))

    first = gut.run(tiny_config, seed=8, tracker_path=tmp_path / "a.npz")
    again = gut.run(tiny_config, seed=8, tracker_path=tmp_path / "b.npz")

    assert first["final"] == again["final"]
    a, b = StateTracker.load(str(tmp_path / "a.npz")), StateTracker.load(str(tmp_path / "b.npz"))
    assert all(
        np.array_equal(x, y)
        for key in a.history
        for x, y in zip(a.history[key], b.history[key], strict=True)
    )


def test_a_run_without_steps_still_saves_its_history(gut, tiny_config, tmp_path):
    tiny_config["sim"]["max_steps"] = 0

    gut.run(tiny_config, seed=1, tracker_path=tmp_path / "none.npz")

    assert StateTracker.load(str(tmp_path / "none.npz")).steps == [0]


def test_periodic_wall_axes_wrap_when_smoothing(gut, monkeypatch):
    modes = []
    real_filter = gut.gaussian_filter

    def record(noise, sigma, mode):
        modes.append(mode)
        return real_filter(noise, sigma=sigma, mode=mode)

    monkeypatch.setattr(gut, "gaussian_filter", record)
    detachment = _detachment(periodicity="z")
    initial = {"mean_thickness": 2, "thickness_spread": 1, "patch_size": 4}

    gut.build_mat(detachment, initial, np.random.default_rng(0))

    assert modes == [["reflect", "wrap"]]  # the floor's axes are x and z


def test_species_are_mixed_in_the_mat_and_the_lumen(gut, tiny_config):
    tiny_config["sim"]["size"] = [20, 20, 20]
    tiny_config["initial_state"].update(mean_thickness=6, thickness_spread=0, lumen_density=0.3)
    abundances = {sp: 0.0 for sp in tiny_config["abundances"]["before"]}
    abundances.update(BBBBB=0.5, other=0.5)
    tiny_config["abundances"] = {"before": abundances, "after": abundances}

    sim, _, _ = gut.prepare(tiny_config, seed=2)

    bbbbb = sim.state["BBBBB"].astype(bool)
    for region in (sim.state["biofilm"].astype(bool), sim.state["floating_bacteria"].astype(bool)):
        assert bbbbb[region].mean() == pytest.approx(0.5, abs=0.05)
    # spread over the grid, not filled in blocks
    per_x_slab = bbbbb.sum(axis=(1, 2))
    assert per_x_slab.min() > 0.5 * per_x_slab.mean()


def test_patch_size_is_roughly_the_width_of_a_mound(gut):
    heights = gut.mat_heights((400, 400), [True, True], 10, 3, 20, np.random.default_rng(0))
    field = heights.astype(float) - heights.mean()

    def correlation(lag):
        return float((field * np.roll(field, lag, axis=0)).mean() / field.var())

    # Gaussian sigma = patch_size / 4: correlation exp(-1) at half a patch, ~0 at a whole one
    assert correlation(10) == pytest.approx(np.exp(-1), abs=0.1)
    assert abs(correlation(20)) < 0.1


def test_preparing_a_run_does_not_log_a_missing_tracker(gut, tiny_config, caplog, monkeypatch):
    parser_logger = logging.getLogger(gut.PARSER_LOGGER)
    monkeypatch.setattr(parser_logger, "level", 5)  # a level nothing else sets

    with caplog.at_level("INFO"):
        gut.prepare(tiny_config, seed=1, tracker_path=None)

    assert "no 'tracker' block" not in caplog.text
    assert parser_logger.level == 5
