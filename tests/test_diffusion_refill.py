"""Diffusion refill places only as many particles as there is room for (BUG-7)."""

from cellular_automata.Geometry import Geometry
from cellular_automata.rules import Diffusion


def _rule():
    geometry = Geometry((4, 4, 4), axes="xyz", periodicity="xy")  # 96 slots per z-layer
    return Diffusion(geometry, time_step=0.01, a=1, p=0.1)


def test_refill_with_fewer_empty_slots_than_lost_particles_does_not_raise():
    rule = _rule()
    rule.layers[...] = 1
    rule.layers[0, 0, 3, :] = 0  # 6 empty slots in the top z-layer
    rule.lost_particles = 150

    rule._restore_random()

    assert rule.layers.min() == 1
    assert rule.lost_particles == 144


def test_refill_with_no_empty_slots_keeps_the_count():
    rule = _rule()
    rule.layers[...] = 1
    rule.lost_particles = 150

    rule._restore_random()

    assert rule.lost_particles == 150


def test_refill_uses_only_the_highest_layer_with_room():
    rule = _rule()
    rule.layers[:, :, 3, :] = 1  # top layer full, so particles go to z = 2
    rule.lost_particles = 100

    rule._restore_random()

    assert rule.layers[:, :, 2, :].sum() == 96  # the whole layer: 4 * 4 * 6 slots
    assert rule.layers[:, :, :2, :].sum() == 0
    assert rule.lost_particles == 4  # placed in a later step


def test_refill_places_a_full_batch_when_there_is_room():
    geometry = Geometry((5, 5, 5), axes="xyz", periodicity="xy")
    rule = Diffusion(geometry, time_step=0.01, a=1, p=0.1)  # top layer: 150 slots
    rule.lost_particles = 130

    rule._restore_random()

    assert rule.layers[:, :, 4, :].sum() == 100
    assert rule.layers.sum() == 100
    assert rule.lost_particles == 30
