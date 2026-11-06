"""Simulation rules.

Each rule lives in its own module; the classes are re-exported here so that
``from cellular_automata.rules import GameOfLife3D`` gives the class.
"""

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
