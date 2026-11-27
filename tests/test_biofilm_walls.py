"""BiofilmDetachment walls, sandwich protection and release of cut-off biofilm."""

import numpy as np
import pytest

from cellular_automata.Geometry import Geometry
from cellular_automata.neighbourhoods.MooreNeighbourhood import MooreNeighbourhood
from cellular_automata.neighbourhoods.VonNeumannNeighbourhood import VonNeumannNeighbourhood
from cellular_automata.rules import BiofilmDetachment
from cellular_automata.State import State

CERTAIN = 1e6  # detachment_rate * scaling far above 1: every exposed cell off the wall detaches


def _geometry(size=(5, 5, 5), periodicity="", neighbourhood=None):
    return Geometry(size, axes="xyz", periodicity=periodicity, neighbourhood=neighbourhood)


def _state(geometry):
    return State(geometry, cell_keys=["biofilm", "floating_bacteria", "A"])


def _rule(geometry, rate=0.0, seed=0, **kwargs):
    rule = BiofilmDetachment(geometry, time_step=1, detachment_rate=rate, scaling=1.0, **kwargs)
    rule.rng = np.random.default_rng(seed)
    return rule


def _biofilm(state, *cells):
    for cell in cells:
        state["biofilm"][cell] = 1


# --- the default wall keeps the old gradient ------------------------------------


def test_default_wall_is_the_low_y_face_with_the_old_gradient():
    geometry = _geometry(size=(3, 6, 4))
    rule = BiofilmDetachment(geometry, time_step=1, detachment_rate=0.2, scaling=0.5)

    y = np.arange(6) / 5
    assert rule.walls == [(1, -1)]
    assert np.allclose(rule.probabilities, (0.1 * y**2)[None, :, None])
    assert rule.touches_wall[:, 0, :].all() and not rule.touches_wall[:, 1:, :].any()


# --- detachment: wall layer, sandwiched and exposed cells ----------------------


def test_only_the_top_of_a_column_detaches_even_when_detachment_is_certain():
    geometry = _geometry()
    state = _state(geometry)
    _biofilm(state, *[(2, y, 2) for y in range(4)])

    state = _rule(geometry, rate=CERTAIN).apply_state(state)

    # y=0 touches the wall and y=1, y=2 are sandwiched; only the exposed top goes
    assert state["biofilm"][2, :3, 2].all()
    assert not state["biofilm"][2, 3, 2] and state["floating_bacteria"][2, 3, 2]


def test_a_column_erodes_one_cell_per_call():
    geometry = _geometry()
    state = _state(geometry)
    _biofilm(state, *[(2, y, 2) for y in range(5)])
    rule = _rule(geometry, rate=CERTAIN)

    heights = []
    for _ in range(5):
        state = rule.apply_state(state)
        heights.append(int(state["biofilm"][2, :, 2].sum()))

    assert heights == [4, 3, 2, 1, 1]  # the wall layer never goes


def test_an_exposed_cell_with_nothing_above_detaches_with_the_gradient_probability():
    # a 1-cell-high mat next to a 3-high mound: the mound's middle is sandwiched
    geometry = _geometry(size=(5, 5, 1))
    state = _state(geometry)
    _biofilm(state, (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 2, 0))
    rule = _rule(geometry, rate=1.0)  # p = (y/4)**2

    detached = np.zeros(geometry.size)
    for seed in range(400):
        fresh = _state(geometry)
        fresh["biofilm"][...] = state["biofilm"]
        rule.rng = np.random.default_rng(seed)
        detached += ~rule.apply_state(fresh)["biofilm"].astype(bool) & state["biofilm"].astype(bool)

    rates = detached / 400
    assert rates[1, 1, 0] == 0  # sandwiched
    assert rates[0, 0, 0] == rates[1, 0, 0] == 0  # wall layer
    assert rates[0, 1, 0] == pytest.approx(1 / 16, abs=0.04)
    assert rates[1, 2, 0] == pytest.approx(4 / 16, abs=0.07)


# --- release: biofilm has to stay linked to a wall ----------------------------------


def test_biofilm_cut_off_from_the_wall_becomes_floating():
    geometry = _geometry()
    state = _state(geometry)
    _biofilm(state, (1, 0, 1), (1, 1, 1))  # linked
    _biofilm(state, (3, 2, 3), (3, 3, 3))  # a hanging cluster

    state = _rule(geometry).apply_state(state)

    assert state["biofilm"][1, 0, 1] and state["biofilm"][1, 1, 1]
    assert not state["biofilm"][3, 2:4, 3].any()
    assert state["floating_bacteria"][3, 2:4, 3].all()
    assert not state["floating_bacteria"][1, :2, 1].any()


def test_a_cluster_held_sideways_by_a_linked_column_stays():
    geometry = _geometry()
    state = _state(geometry)
    _biofilm(state, (1, 0, 1), (1, 1, 1), (1, 2, 1), (2, 2, 1), (3, 2, 1))  # an overhang

    state = _rule(geometry).apply_state(state)

    assert state["biofilm"][(1, 2, 3), (2, 2, 2), (1, 1, 1)].all()
    assert not state["floating_bacteria"].any()


@pytest.mark.parametrize(
    "neighbourhood, stays", [(MooreNeighbourhood(), True), (VonNeumannNeighbourhood(), False)]
)
def test_a_diagonal_link_counts_only_in_the_moore_neighbourhood(neighbourhood, stays):
    geometry = _geometry(neighbourhood=neighbourhood)
    state = _state(geometry)
    _biofilm(state, (1, 0, 1), (2, 1, 2))  # touching only at a corner

    state = _rule(geometry).apply_state(state)

    assert bool(state["biofilm"][2, 1, 2]) is stays
    assert bool(state["floating_bacteria"][2, 1, 2]) is not stays


@pytest.mark.parametrize("periodicity, stays", [("x", True), ("", False)])
def test_links_wrap_around_periodic_axes(periodicity, stays):
    geometry = _geometry(periodicity=periodicity)
    state = _state(geometry)
    _biofilm(state, (0, 0, 2), (4, 1, 2))  # neighbours only across the x edge

    state = _rule(geometry).apply_state(state)

    assert bool(state["biofilm"][4, 1, 2]) is stays


def test_a_bacterium_attaching_to_a_hanging_cluster_floats_off_with_it():
    geometry = _geometry()
    state = _state(geometry)
    _biofilm(state, (2, 2, 2))  # hanging
    state["A"][2, 3, 2] = 1
    state["floating_bacteria"][2, 3, 2] = 1

    state = _rule(geometry, species_keys=["A"]).apply_state(state)

    assert not state["biofilm"].any()
    assert state["floating_bacteria"][2, 2, 2] and state["floating_bacteria"][2, 3, 2]


def test_release_keeps_every_cell_exactly_once():
    geometry = _geometry(size=(6, 6, 6))
    for seed in range(5):
        state = _state(geometry)
        rng = np.random.default_rng(seed)
        state["biofilm"][...] = rng.random(geometry.size) < 0.4
        before = state["biofilm"].astype(bool)

        state = _rule(geometry, rate=0.5, seed=seed).apply_state(state)

        biofilm = state["biofilm"].astype(bool)
        floating = state["floating_bacteria"].astype(bool)
        assert not (biofilm & floating).any()
        assert np.array_equal(biofilm | floating, before)
        assert np.array_equal(biofilm, biofilm & _rule(geometry)._linked_to_wall(biofilm))


# --- several walls ---------------------------------------------------------------


def test_two_walls_measure_distance_to_the_nearest_one():
    geometry = _geometry(size=(2, 7, 2))
    rule = BiofilmDetachment(
        geometry, time_step=1, detachment_rate=1.0, scaling=1.0, walls=["y-", "y+"]
    )

    distance_in_cells = np.array([0, 1, 2, 3, 2, 1, 0])
    assert np.allclose(rule.probabilities[0, :, 0], (distance_in_cells / 3) ** 2)
    assert rule.touches_wall[0, :, 0].tolist() == [1, 0, 0, 0, 0, 0, 1]
    # the middle row is a tie and uses the first wall listed
    assert rule.nearest_wall[0, :, 0].tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_biofilm_grows_down_from_a_ceiling():
    geometry = _geometry()
    state = _state(geometry)
    _biofilm(state, (2, 4, 2), (2, 3, 2))  # hanging from y+
    state["A"][2, 2, 2] = 1
    state["floating_bacteria"][2, 2, 2] = 1  # below it, i.e. away from the ceiling

    state = _rule(geometry, species_keys=["A"], walls=["y+"]).apply_state(state)

    assert state["biofilm"][2, 2:5, 2].all()
    assert not state["floating_bacteria"].any()


def test_a_floor_column_is_sandwiched_against_a_ceiling_column():
    geometry = _geometry(size=(1, 6, 1))
    state = _state(geometry)
    state["biofilm"][...] = 1

    state = _rule(geometry, rate=CERTAIN, walls=["y-", "y+"]).apply_state(state)

    # every cell has biofilm on both sides, or touches a wall
    assert state["biofilm"].all()


def test_a_wall_on_another_axis():
    geometry = _geometry()
    state = _state(geometry)
    _biofilm(state, *[(2, 2, z) for z in range(3)])

    state = _rule(geometry, rate=CERTAIN, walls=["z-"]).apply_state(state)

    assert state["biofilm"][2, 2, :2].all() and not state["biofilm"][2, 2, 2]


@pytest.mark.parametrize(
    "walls, message",
    [
        ([], "at least one wall"),
        (["w-"], "must be an axis"),
        (["y"], "must be an axis"),
        (["y-", "y-"], "listed twice"),
        (["x-"], "periodic axis"),
    ],
)
def test_bad_walls_are_rejected(walls, message):
    with pytest.raises(ValueError, match=message):
        BiofilmDetachment(
            _geometry(periodicity="x"), time_step=1, detachment_rate=0, scaling=1, walls=walls
        )


# --- linking against a plain breadth-first search ----------------------------------


def _reference_linked(rule, biofilm):
    """Cells reachable from wall-touching biofilm, one neighbour step at a time."""
    size = np.array(biofilm.shape)
    periodic = [axis in rule.geometry.periodic_dims for axis in range(biofilm.ndim)]
    queue = [tuple(c) for c in np.argwhere(biofilm & rule.touches_wall)]
    seen = set(queue)
    while queue:
        cell = queue.pop()
        for offset in rule.geometry._offsets:
            nb = np.array(cell) + offset
            for axis in range(len(nb)):
                if periodic[axis]:
                    nb[axis] %= size[axis]
            if (nb < 0).any() or (nb >= size).any():
                continue
            nb_t = tuple(int(v) for v in nb)
            if biofilm[nb_t] and nb_t not in seen:
                seen.add(nb_t)
                queue.append(nb_t)
    linked = np.zeros_like(biofilm)
    for cell in seen:
        linked[cell] = True
    return linked


@pytest.mark.parametrize("neighbourhood", [MooreNeighbourhood(), VonNeumannNeighbourhood()])
@pytest.mark.parametrize("periodicity", ["", "x", "xz"])
@pytest.mark.parametrize("walls", [["y-"], ["y-", "y+"], ["z+"]])
def test_linking_matches_a_breadth_first_search(neighbourhood, periodicity, walls):
    if "z" in periodicity and any(w.startswith("z") for w in walls):
        pytest.skip("a wall can't sit on a periodic axis")
    geometry = _geometry(size=(7, 6, 5), periodicity=periodicity, neighbourhood=neighbourhood)
    rule = _rule(geometry, walls=walls)
    for seed in range(10):
        biofilm = np.random.default_rng(seed).random(geometry.size) < 0.3

        assert np.array_equal(rule._linked_to_wall(biofilm), _reference_linked(rule, biofilm))


def test_linking_a_thin_mat_far_from_the_grid_edges():
    # the search is cropped to the biofilm's bounding box; links must survive that
    geometry = _geometry(size=(9, 9, 9), periodicity="x")
    rule = _rule(geometry)
    biofilm = np.zeros(geometry.size, dtype=bool)
    biofilm[3:6, 0:3, 4:6] = True  # linked block
    biofilm[8, 1, 5] = biofilm[0, 2, 5] = True  # linked only through the x wrap... and not at all
    biofilm[6, 6, 6] = True  # floating far away

    assert np.array_equal(rule._linked_to_wall(biofilm), _reference_linked(rule, biofilm))


# --- review follow-ups ----------------------------------------------------------------


def test_an_overhang_with_biofilm_above_but_not_below_can_detach():
    # x=0 column y=0..3; at x=1 cells at y=3 and y=4 hang off it
    geometry = _geometry(size=(5, 5, 1))
    state = _state(geometry)
    _biofilm(state, (0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 3, 0), (1, 4, 0))

    state = _rule(geometry, rate=CERTAIN).apply_state(state)

    # (1, 3) has biofilm above it but nothing below, so it isn't sandwiched
    assert not state["biofilm"][1, 3, 0] and state["floating_bacteria"][1, 3, 0]


def test_each_cell_attaches_towards_its_own_nearest_wall():
    # biofilm hangs from the ceiling at y=5..7; y=4 is 3 cells from y+ and 4 from y-
    geometry = _geometry(size=(1, 8, 1))
    state = _state(geometry)
    _biofilm(state, (0, 5, 0), (0, 6, 0), (0, 7, 0))
    state["A"][0, 4, 0] = 1
    state["floating_bacteria"][0, 4, 0] = 1

    state = _rule(geometry, species_keys=["A"], walls=["y-", "y+"]).apply_state(state)

    assert state["biofilm"][0, 4, 0] and not state["floating_bacteria"][0, 4, 0]


def test_walls_given_as_a_single_string_are_rejected_clearly():
    with pytest.raises(ValueError, match="must be a list"):
        BiofilmDetachment(_geometry(), time_step=1, detachment_rate=0, scaling=1, walls="y-")


class _RadiusTwoVonNeumann:
    def generate_offsets(self, ndim, radius=1):
        return VonNeumannNeighbourhood().generate_offsets(ndim, radius=2)


def test_wider_neighbourhoods_are_rejected():
    with pytest.raises(ValueError, match="radius-1"):
        _rule(_geometry(neighbourhood=_RadiusTwoVonNeumann()))


def test_the_per_cell_arrays_are_small():
    rule = _rule(_geometry(size=(4, 4, 4)), walls=["y-", "y+"])

    assert rule.nearest_wall.dtype == np.int8
    assert rule.probabilities.dtype == np.float32
    assert rule.touches_wall.dtype == np.bool_
