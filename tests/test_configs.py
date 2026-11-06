"""Every JSON config in the repo must build a simulation (rendering forced off)."""

import json
from pathlib import Path

import pytest

from cellular_automata.Sim import CellularAutomaton
from cellular_automata.config.Parser import parse_json

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIRS = [REPO_ROOT / "configs", REPO_ROOT / "research"]
CONFIGS = sorted(
    path
    for config_dir in CONFIG_DIRS
    for path in config_dir.rglob("*.json")
    if "configs" in path.parts
)


def test_configs_were_found():
    assert len(CONFIGS) > 0


@pytest.mark.parametrize(
    "config_path", CONFIGS, ids=[str(p.relative_to(REPO_ROOT)) for p in CONFIGS]
)
def test_config_builds_simulation(config_path: Path, tmp_path: Path):
    data = json.loads(config_path.read_text())
    data["sim"]["render"] = False
    headless_config = tmp_path / config_path.name
    headless_config.write_text(json.dumps(data))

    ca_sim, renderer = parse_json(headless_config)

    assert isinstance(ca_sim, CellularAutomaton)
    assert renderer is None
    assert ca_sim.state.shape[: ca_sim.geometry.ndim] == tuple(data["sim"]["size"])
