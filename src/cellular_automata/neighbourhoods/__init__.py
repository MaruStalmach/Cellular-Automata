"""Neighbourhoods: every module here is imported so its class registers itself."""

import sys

from cellular_automata.registry import import_submodules

import_submodules(sys.modules[__name__])
