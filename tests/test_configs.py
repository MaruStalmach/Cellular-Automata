"""Every JSON config in the repo must build a simulation (rendering forced off)."""

import json
import re
from pathlib import Path

import pytest

from cellular_automata.config.Parser import parse_json
from cellular_automata.Sim import CellularAutomaton

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


def test_readme_example_config_builds_simulation(tmp_path: Path):
    readme = (REPO_ROOT / "README.md").read_text()
    blocks = re.findall(r"```json\n(.*?)```", readme, flags=re.DOTALL)
    assert len(blocks) == 1, "expected exactly one JSON example in README.md"
    config_path = tmp_path / "readme_example.json"
    config_path.write_text(blocks[0])

    ca_sim, renderer = parse_json(config_path)

    assert isinstance(ca_sim, CellularAutomaton)
    assert renderer is None
