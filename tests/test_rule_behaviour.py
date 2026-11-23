"""Behaviour and invariant tests for the rules that only had key and RNG checks (TEST-2).

Every test uses a tiny grid and a seeded generator, and checks a property that must
hold for any random draw (particles conserved, nothing created from nothing, no cell
held by two species) or an outcome forced by a probability of 0 or 1.
"""

import numpy as np
import pytest

from cellular_automata.Geometry import Geometry
from cellular_automata.rules import (
    BacteriaDecay,
    BacteriaGrowth,
    BiofilmDetachment,
    Diffusion,
    GutDrift,
    Utilization,
)
from cellular_automata.State import State

SEEDS = range(5)


def _geometry(size=(5, 5, 5), periodicity=""):
    return Geometry(size, axes="xyz", periodicity=periodicity)


def _state(geometry, keys, layers=None):
    return State(geometry, cell_keys=keys, key_layers=layers or [1] * len(keys))


def _seeded(rule, seed):
    rule.rng = np.random.default_rng(seed)
    return rule


def _neighbours(coords, size):
    """Moore neighbours of `coords` on a non-periodic grid."""
    found = set()
    for offset in np.ndindex(3, 3, 3):
        delta = np.array(offset) - 1
        if not delta.any():
            continue
        target = np.array(coords) + delta
        if ((target >= 0) & (target < size)).all():
            found.add(tuple(int(v) for v in target))
    return found


# --- BacteriaGrowth ---------------------------------------------------------------


def _growth(geometry, **kwargs):
    return BacteriaGrowth(
        geometry, time_step=1, bacteria_keys=["A", "B"], substrate_key="substrate", **kwargs
    )


def _growth_state(geometry, rng):
    state = _state(geometry, ["A", "B", "substrate"], [1, 1, 6])
    cells = rng.random(geometry.size)
    state["A"][...] = cells < 0.15
    state["B"][...] = (cells >= 0.15) & (cells < 0.3)
    state["substrate"][...] = rng.random(state["substrate"].shape) < 0.5
    return state


def test_growth_without_substrate_creates_nothing():
    geometry = _geometry()
    state = _growth_state(geometry, np.random.default_rng(0))
    state["substrate"][...] = 0
    before = {key: state[key].copy() for key in ("A", "B")}

    state = _seeded(_growth(geometry), 0).apply_state(state)

    for key in ("A", "B"):
        np.testing.assert_array_equal(state[key], before[key])


@pytest.mark.parametrize("seed", SEEDS)
def test_growth_keeps_every_cell_and_never_double_occupies(seed):
    geometry = _geometry()
    state = _growth_state(geometry, np.random.default_rng(seed))
    before = {key: state[key].copy() for key in ("A", "B", "substrate")}

    state = _seeded(_growth(geometry, base_prob=1.0), seed).apply_state(state)

    for key in ("A", "B"):
        assert (state[key] >= before[key]).all(), f"{key} lost a cell"
    assert not (state["A"].astype(bool) & state["B"].astype(bool)).any()
    np.testing.assert_array_equal(state["substrate"], before["substrate"])


@pytest.mark.parametrize("seed", SEEDS)
def test_growth_places_each_new_cell_next_to_a_parent_of_the_same_species(seed):
    geometry = _geometry()
    state = _growth_state(geometry, np.random.default_rng(seed))
    before = {key: state[key].copy() for key in ("A", "B")}

    state = _seeded(_growth(geometry, base_prob=1.0), seed).apply_state(state)

    for key in ("A", "B"):
        new_cells = np.argwhere(state[key].astype(bool) & ~before[key].astype(bool))
        parents = {tuple(int(v) for v in c) for c in np.argwhere(before[key])}
        for cell in map(tuple, new_cells):
            assert _neighbours(cell, geometry.size) & parents, f"{key} grew at {cell}"


@pytest.mark.parametrize("seed", SEEDS)
def test_growth_gives_each_parent_at_most_one_daughter(seed):
    geometry = _geometry()
    state = _state(geometry, ["A", "B", "substrate"], [1, 1, 6])
    state["A"][2, 2, 2] = 1
    state["substrate"][...] = 1

    state = _seeded(_growth(geometry, base_prob=1.0), seed).apply_state(state)

    # growth probability is clipped to 1 with this much substrate
    assert int(state["A"].sum()) == 2


# --- Utilization -------------------------------------------------------------------


def _utilization(geometry, **kwargs):
    return Utilization(
        geometry, time_step=1, bacteria_keys=["A", "B"], substrate_key="substrate", **kwargs
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_utilization_never_adds_substrate_or_changes_bacteria(seed):
    geometry = _geometry()
    state = _growth_state(geometry, np.random.default_rng(seed))
    before = {key: state[key].copy() for key in ("A", "B", "substrate")}

    state = _seeded(_utilization(geometry), seed).apply_state(state)

    assert (state["substrate"] <= before["substrate"]).all()
    for key in ("A", "B"):
        np.testing.assert_array_equal(state[key], before[key])


@pytest.mark.parametrize("seed", SEEDS)
def test_utilization_eats_at_most_one_particle_per_bacterium(seed):
    geometry = _geometry()
    state = _growth_state(geometry, np.random.default_rng(seed))
    bacteria = int(state["A"].sum() + state["B"].sum())
    substrate_before = int(state["substrate"].sum())

    state = _seeded(_utilization(geometry, substitute_constant=1.0), seed).apply_state(state)

    eaten = substrate_before - int(state["substrate"].sum())
    assert 0 <= eaten <= bacteria


@pytest.mark.parametrize("seed", SEEDS)
def test_utilization_only_eats_next_to_bacteria(seed):
    geometry = _geometry()
    state = _state(geometry, ["A", "B", "substrate"], [1, 1, 6])
    state["A"][0, 0, 0] = 1
    state["substrate"][...] = 1

    state = _seeded(_utilization(geometry, substitute_constant=1.0), seed).apply_state(state)

    eaten_cells = {tuple(int(v) for v in c) for c in np.argwhere((state["substrate"] == 0).any(-1))}
    assert eaten_cells <= _neighbours((0, 0, 0), geometry.size)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_particle_two_bacteria_pick_is_eaten_once(seed):
    geometry = _geometry()
    state = _state(geometry, ["A", "B", "substrate"], [1, 1, 6])
    state["A"][1, 2, 2] = 1
    state["B"][3, 2, 2] = 1
    state["substrate"][2, 2, 2, 0] = 1  # the only particle, next to both
    rule = _utilization(geometry, substitute_constant=1.0, Sb=1e9)  # both always eat

    state = _seeded(rule, seed).apply_state(state)

    assert not state["substrate"].any()


def test_utilization_without_bacteria_leaves_substrate_alone():
    geometry = _geometry()
    state = _state(geometry, ["A", "B", "substrate"], [1, 1, 6])
    state["substrate"][...] = 1

    state = _seeded(_utilization(geometry, substitute_constant=1.0), 0).apply_state(state)

    assert state["substrate"].all()


# --- Diffusion ---------------------------------------------------------------------


def _diffusion_state(geometry, rng, density=0.1):
    state = _state(geometry, ["substrate", "biofilm"], [6, 1])
    state["substrate"][...] = rng.random(state["substrate"].shape) < density
    state["biofilm"][...] = rng.random(geometry.size) < 0.3
    return state


@pytest.mark.parametrize("periodicity", ["", "xy", "xyz"])
@pytest.mark.parametrize("biofilm_key", [None, "biofilm"])
@pytest.mark.parametrize("seed", SEEDS)
def test_diffusion_accounts_for_every_particle(seed, biofilm_key, periodicity):
    """Particles on the grid plus those waiting to be refilled stay constant."""
    geometry = _geometry(periodicity=periodicity)
    state = _diffusion_state(geometry, np.random.default_rng(seed))
    rule = _seeded(
        Diffusion(geometry, time_step=1, a=7.4, p=0.07, biofilm_key=biofilm_key), seed
    )
    total = int(state["substrate"].sum())

    for _ in range(10):
        state = rule.apply_state(state)
        assert set(np.unique(state["substrate"])) <= {0, 1}
        assert int(state["substrate"].sum()) + rule.lost_particles == total


@pytest.mark.parametrize("seed", SEEDS)
def test_diffusion_moves_a_lone_particle_to_an_adjacent_cell(seed):
    geometry = _geometry()
    state = _state(geometry, ["substrate"], [6])
    state["substrate"][2, 2, 2, int(np.random.default_rng(seed).integers(6))] = 1
    rule = _seeded(Diffusion(geometry, time_step=1, a=7.4, p=0.07), seed)

    state = rule.apply_state(state)

    (cell,) = np.argwhere(state["substrate"].any(-1))
    assert np.abs(cell - 2).sum() == 1
    assert int(state["substrate"].sum()) == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_diffusion_reflects_particles_at_closed_walls(seed):
    # a = 1e9: particles almost never turn, so they run into the walls
    geometry = _geometry(size=(3, 3, 3))
    state = _state(geometry, ["substrate"], [6])
    state["substrate"][0, 0, 0, :] = 1
    rule = _seeded(Diffusion(geometry, time_step=1, a=1e9, p=1e-12), seed)

    for _ in range(20):
        state = rule.apply_state(state)
        assert int(state["substrate"].sum()) + rule.lost_particles == 6


# --- GutDrift ----------------------------------------------------------------------


def _drift_state(geometry):
    return _state(geometry, ["floating_bacteria", "A", "B"])


def test_drift_moves_floating_cells_along_z():
    geometry = _geometry()
    state = _drift_state(geometry)
    state["A"][1, 1, 1] = 1
    state["floating_bacteria"][1, 1, 1] = 1

    state = GutDrift(geometry, time_step=1, drift_speed=2, species_keys=["A", "B"]).apply_state(state)

    assert state["A"][1, 1, 3] == 1 and int(state["A"].sum()) == 1
    assert state["floating_bacteria"][1, 1, 3] == 1 and int(state["floating_bacteria"].sum()) == 1


def test_drift_leaves_attached_cells_in_place():
    geometry = _geometry()
    state = _drift_state(geometry)
    state["B"][2, 2, 0] = 1  # not floating

    state = GutDrift(geometry, time_step=1, drift_speed=1, species_keys=["A", "B"]).apply_state(state)

    assert state["B"][2, 2, 0] == 1 and int(state["B"].sum()) == 1


def test_drift_off_a_closed_end_removes_the_cell():
    geometry = _geometry()
    state = _drift_state(geometry)
    state["A"][0, 0, 4] = 1
    state["floating_bacteria"][0, 0, 4] = 1

    state = GutDrift(geometry, time_step=1, drift_speed=1, species_keys=["A", "B"]).apply_state(state)

    assert not state["A"].any() and not state["floating_bacteria"].any()


def test_drift_wraps_on_a_periodic_axis():
    geometry = _geometry(periodicity="z")
    state = _drift_state(geometry)
    state["A"][0, 0, 4] = 1
    state["floating_bacteria"][0, 0, 4] = 1

    state = GutDrift(geometry, time_step=1, drift_speed=1, species_keys=["A", "B"]).apply_state(state)

    assert state["A"][0, 0, 0] == 1 and state["floating_bacteria"][0, 0, 0] == 1


def test_drift_into_an_attached_cell_removes_the_floating_cell():
    geometry = _geometry()
    state = _drift_state(geometry)
    state["A"][1, 1, 1] = 1
    state["floating_bacteria"][1, 1, 1] = 1
    state["B"][1, 1, 2] = 1  # attached, in the way

    state = GutDrift(geometry, time_step=1, drift_speed=1, species_keys=["A", "B"]).apply_state(state)

    assert not state["A"].any()
    assert state["B"][1, 1, 2] == 1
    assert not state["floating_bacteria"].any()


@pytest.mark.parametrize("seed", SEEDS)
def test_drift_never_double_occupies(seed):
    geometry = _geometry()
    rng = np.random.default_rng(seed)
    state = _drift_state(geometry)
    cells = rng.random(geometry.size)
    state["A"][...] = cells < 0.2
    state["B"][...] = (cells >= 0.2) & (cells < 0.4)
    state["floating_bacteria"][...] = (cells < 0.4) & (rng.random(geometry.size) < 0.5)

    state = GutDrift(geometry, time_step=1, drift_speed=1, species_keys=["A", "B"]).apply_state(state)

    assert not (state["A"].astype(bool) & state["B"].astype(bool)).any()
    occupied = state["A"].astype(bool) | state["B"].astype(bool)
    assert not (state["floating_bacteria"].astype(bool) & ~occupied).any()


# --- BiofilmDetachment -------------------------------------------------------------


def _detachment_state(geometry):
    return _state(geometry, ["biofilm", "floating_bacteria", "A", "B"])


@pytest.mark.parametrize("seed", SEEDS)
def test_detachment_never_creates_biofilm_without_attachment(seed):
    geometry = _geometry()
    state = _detachment_state(geometry)
    state["biofilm"][...] = np.random.default_rng(seed).random(geometry.size) < 0.5
    before = state["biofilm"].copy()
    rule = _seeded(BiofilmDetachment(geometry, time_step=1, detachment_rate=0.5, scaling=1.0), seed)

    state = rule.apply_state(state)

    assert (state["biofilm"] <= before).all()
    # every cell that detached is now floating
    detached = before.astype(bool) & ~state["biofilm"].astype(bool)
    assert state["floating_bacteria"][detached].all()


def test_certain_detachment_clears_all_but_the_wall_layer():
    # probability grows with y**2 and is 0 at the wall (y = 0)
    geometry = _geometry()
    state = _detachment_state(geometry)
    state["biofilm"][...] = 1
    rule = _seeded(BiofilmDetachment(geometry, time_step=1, detachment_rate=1.0, scaling=1.0), 0)

    state = rule.apply_state(state)

    assert state["biofilm"][:, 0, :].all()
    assert not state["biofilm"][:, -1, :].any()
    assert not state["floating_bacteria"][:, 0, :].any()


def test_zero_detachment_changes_nothing():
    geometry = _geometry()
    state = _detachment_state(geometry)
    state["biofilm"][...] = 1
    rule = _seeded(BiofilmDetachment(geometry, time_step=1, detachment_rate=0.0, scaling=1.0), 0)

    state = rule.apply_state(state)

    assert state["biofilm"].all() and not state["floating_bacteria"].any()


def test_certain_attachment_attaches_a_bacterium_on_top_of_biofilm():
    geometry = _geometry()
    state = _detachment_state(geometry)
    state["biofilm"][2, 0, 2] = 1
    state["A"][2, 1, 2] = 1  # one step further from the wall, on top of the biofilm
    state["floating_bacteria"][2, 1, 2] = 1
    state["B"][2, 3, 2] = 1  # nothing below it
    state["floating_bacteria"][2, 3, 2] = 1
    rule = BiofilmDetachment(
        geometry, time_step=1, detachment_rate=0.0, scaling=1.0,
        species_keys=["A", "B"], attachment_probability=1.0,
    )

    state = rule.apply_state(state)

    assert state["biofilm"][2, 1, 2] == 1 and state["floating_bacteria"][2, 1, 2] == 0
    assert state["biofilm"][2, 3, 2] == 0 and state["floating_bacteria"][2, 3, 2] == 1


# --- BacteriaDecay -----------------------------------------------------------------


@pytest.mark.parametrize("probability, survivors", [(0.0, "all"), (1.0, "none")])
def test_decay_at_the_extremes(probability, survivors):
    geometry = _geometry()
    state = _state(geometry, ["A", "B"])
    state["A"][...] = np.random.default_rng(0).random(geometry.size) < 0.5
    before = state["A"].copy()
    rule = BacteriaDecay(geometry, time_step=1, target_keys=["A", "B"], decay_probability=probability)

    state = _seeded(rule, 0).apply_state(state)

    if survivors == "all":
        np.testing.assert_array_equal(state["A"], before)
    else:
        assert not state["A"].any()


@pytest.mark.parametrize("seed", SEEDS)
def test_decay_never_creates_cells(seed):
    geometry = _geometry()
    state = _state(geometry, ["A", "B"])
    state["A"][...] = np.random.default_rng(seed).random(geometry.size) < 0.5
    before = state["A"].copy()
    rule = BacteriaDecay(geometry, time_step=1, target_keys=["A", "B"], decay_probability=0.5)

    state = _seeded(rule, seed).apply_state(state)

    assert (state["A"] <= before).all() and not state["B"].any()


# --- fairness of the vectorised choices (PERF-1) -------------------------------------


def test_growth_picks_each_empty_neighbour_equally_often():
    geometry = _geometry()
    counts = np.zeros(geometry.size, dtype=int)
    for seed in range(2600):
        state = _state(geometry, ["A", "B", "substrate"], [1, 1, 6])
        state["A"][2, 2, 2] = 1
        state["substrate"][...] = 1
        state = _seeded(_growth(geometry, base_prob=1.0), seed).apply_state(state)
        state["A"][2, 2, 2] = 0
        counts += state["A"]

    picks = counts[1:4, 1:4, 1:4].ravel()
    picks = np.delete(picks, 13)  # the parent itself
    # 2600 draws over 26 cells: 100 each, sd ~10
    assert picks.sum() == 2600
    assert picks.min() > 65 and picks.max() < 135


def test_two_species_competing_for_one_cell_win_equally_often():
    geometry = _geometry(size=(1, 1, 3))
    wins = {"A": 0, "B": 0}
    for seed in range(2000):
        state = _state(geometry, ["A", "B", "substrate"], [1, 1, 6])
        state["A"][0, 0, 0] = 1
        state["B"][0, 0, 2] = 1
        state["substrate"][...] = 1
        state = _seeded(_growth(geometry, base_prob=1.0), seed).apply_state(state)
        for key in wins:
            wins[key] += int(state[key][0, 0, 1])

    assert wins["A"] + wins["B"] == 2000
    assert 900 < wins["A"] < 1100  # sd ~22


def test_utilization_picks_particles_not_cells():
    # one neighbour holds 3 particles, the other 1: the first should be picked 3x as often
    geometry = _geometry(size=(1, 1, 3))
    picked = np.zeros(2, dtype=int)
    for seed in range(2000):
        state = _state(geometry, ["A", "B", "substrate"], [1, 1, 6])
        state["A"][0, 0, 1] = 1
        state["substrate"][0, 0, 0, :3] = 1
        state["substrate"][0, 0, 2, 0] = 1
        rule = _utilization(geometry, substitute_constant=1.0, Sb=1e9)
        state = _seeded(rule, seed).apply_state(state)
        picked += [3 - int(state["substrate"][0, 0, 0].sum()), 1 - int(state["substrate"][0, 0, 2].sum())]

    assert picked.sum() == 2000
    assert 1440 < picked[0] < 1560  # expected 1500, sd ~19


@pytest.mark.parametrize("seed", SEEDS)
def test_utilization_gives_the_same_result_in_chunks(seed, monkeypatch):
    geometry = _geometry()
    whole = _seeded(_utilization(geometry), seed).apply_state(
        _growth_state(geometry, np.random.default_rng(seed))
    )
    monkeypatch.setattr(Utilization, "CHUNK", 3)
    chunked = _seeded(_utilization(geometry), seed).apply_state(
        _growth_state(geometry, np.random.default_rng(seed))
    )

    np.testing.assert_array_equal(whole["substrate"], chunked["substrate"])
