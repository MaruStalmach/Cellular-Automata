"""Smoke tests for examples/: every script imports, and builds (not runs) its simulation."""

import importlib.util
from pathlib import Path

import pytest

from cellular_automata.Sim import CellularAutomaton

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.py"))
# examples that build a simulation in code expose build_sim(size)
BUILDABLE = ["sim_no_render", "CA_diffusion", "CA_interaction", "CA_drift_sim"]
TINY_SIZE = (4, 4, 4)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_imports(path: Path):
    _load(path)


@pytest.mark.parametrize("name", BUILDABLE)
def test_example_builds_simulation(name: str):
    module = _load(EXAMPLES_DIR / f"{name}.py")

    ca_sim = module.build_sim(TINY_SIZE)

    assert isinstance(ca_sim, CellularAutomaton)
    ca_sim.step()
    assert ca_sim.step_no == 1
