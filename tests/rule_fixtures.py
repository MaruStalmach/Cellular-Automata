"""Each built-in rule with the keys (and layers) it uses, shared by the rule tests."""

from types import SimpleNamespace

import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.rules import (
    AntibioticEffect,
    BacteriaDecay,
    BacteriaGrowth,
    BiofilmDetachment,
    BioreactorBiofilmDetachment,
    CreateBiofilm,
    Diffusion,
    GameOfLife3D,
    GutDrift,
    Inoculation,
    SpeciesInteraction,
    Utilization,
)
from cellular_automata.State import State

SIZE = (6, 6, 6)


def make_geometry():
    return Geometry(SIZE, axes="xyz", periodicity="")


def make_state(geometry, keys, layers=None):
    """Half-filled random state, drawn before the global RNG is patched."""
    layers = layers or [1] * len(keys)
    rng = np.random.default_rng(0)
    state = State(geometry, cell_keys=keys, key_layers=layers)
    for key in keys:
        state[key][...] = rng.integers(0, 2, size=state[key].shape)
    # AntibioticEffect reads the step number through this back-reference (ARCH-5)
    state.sim = SimpleNamespace(step_no=0, max_steps=10)
    return state


def species_rules(geometry):
    species = ["A", "B"]
    return [
        (
            AntibioticEffect(
                geometry, time_step=1, bacteria_keys=species, biofilm_key="biofilm",
                effectiveness=[0.5, 0.5], base_prob=0.001,
            ),
            ["A", "B", "biofilm"],
            None,
        ),
        (BacteriaDecay(geometry, time_step=1, target_keys=species, decay_probability=0.5),
         species, None),
        (
            BacteriaGrowth(
                geometry, time_step=1, bacteria_keys=species, substrate_key="substrate"
            ),
            ["A", "B", "substrate"],
            [1, 1, 6],
        ),
        (
            Utilization(geometry, time_step=1, bacteria_keys=species, substrate_key="substrate"),
            ["A", "B", "substrate"],
            [1, 1, 6],
        ),
        (
            BiofilmDetachment(
                geometry, time_step=1, detachment_rate=0.5, scaling=1.0,
                species_keys=species, attachment_probability=0.5,
            ),
            ["biofilm", "floating_bacteria", "A", "B"],
            None,
        ),
        (
            BioreactorBiofilmDetachment(
                geometry, time_step=1, biofilm_key="biofilm", detachment_rate=0.5, scaling=1.0
            ),
            ["biofilm"],
            None,
        ),
        (
            SpeciesInteraction(
                geometry, time_step=1,
                interactions={"A": {"B": -1.0}, "B": {"A": 1.0}},
            ),
            ["A", "B"],
            None,
        ),
        (Diffusion(geometry, time_step=1, a=7.4, p=0.07), ["substrate"], [6]),
        (
            Diffusion(geometry, time_step=1, a=7.4, p=0.07, biofilm_key="biofilm"),
            ["substrate", "biofilm"],
            [6, 1],
        ),
    ]


RULE_IDS = [
    "AntibioticEffect", "BacteriaDecay", "BacteriaGrowth", "Utilization",
    "BiofilmDetachment", "BioreactorBiofilmDetachment", "SpeciesInteraction",
    "Diffusion", "Diffusion-biofilm",
]


def other_rules(geometry):
    """Rules that draw no random numbers, so they only belong in key checks."""
    return [
        (GameOfLife3D(geometry, time_step=1, target_key="alive"), ["alive"], None),
        (
            CreateBiofilm(geometry, time_step=1, biofilm_key="biofilm", bacteria_keys=["A", "B"]),
            ["biofilm", "A", "B"],
            None,
        ),
        (
            GutDrift(geometry, time_step=1, drift_speed=1, species_keys=["A", "B"]),
            ["floating_bacteria", "A", "B"],
            None,
        ),
        (
            Inoculation(
                geometry, time_step=1, target_key="substrate", location=[1, 1, 1],
                size=[2, 2, 2],
            ),
            ["substrate"],
            [6],
        ),
    ]


OTHER_RULE_IDS = ["GameOfLife3D", "CreateBiofilm", "GutDrift", "Inoculation"]
