"""Simulation rules.

Each rule lives in its own module and registers itself with ``@register_rule``.
Every module in this package is imported below, so a new rule needs only a new
file to be usable from configs. The built-in classes are also re-exported, so
``from cellular_automata.rules import GameOfLife3D`` gives the class.
"""

import sys

from cellular_automata.registry import import_submodules
from cellular_automata.rules.AntibioticEffect import AntibioticEffect
from cellular_automata.rules.BacteriaDecay import BacteriaDecay
from cellular_automata.rules.BacteriaGrowth import BacteriaGrowth
from cellular_automata.rules.BiofilmDetachment import BiofilmDetachment
from cellular_automata.rules.BioreactorBiofilmDetachment import (
    BioreactorBiofilmDetachment,
)
from cellular_automata.rules.CreateBiofilm import CreateBiofilm
from cellular_automata.rules.Diffusion import Diffusion
from cellular_automata.rules.GameOfLife3D import GameOfLife3D
from cellular_automata.rules.GutDrift import GutDrift
from cellular_automata.rules.Inoculation import Inoculation
from cellular_automata.rules.SpeciesInteraction import SpeciesInteraction
from cellular_automata.rules.Utilization import Utilization

__all__ = [
    "AntibioticEffect",
    "BacteriaDecay",
    "BacteriaGrowth",
    "BiofilmDetachment",
    "BioreactorBiofilmDetachment",
    "CreateBiofilm",
    "Diffusion",
    "GameOfLife3D",
    "GutDrift",
    "Inoculation",
    "SpeciesInteraction",
    "Utilization",
]

import_submodules(sys.modules[__name__])
