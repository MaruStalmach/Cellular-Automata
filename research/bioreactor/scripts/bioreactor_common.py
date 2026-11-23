"""Paths and config helpers shared by the bioreactor scripts."""

import json
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = STUDY_DIR / "configs"


def load_interactions(config_name: str, configs_dir: Path = CONFIGS_DIR) -> dict:
    """The SpeciesInteraction coefficients in `<configs_dir>/<config_name>.json`."""
    config_path = configs_dir / f"{config_name}.json"
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    for rule in config["rules"]:
        if rule.get("name") == "SpeciesInteraction":
            return rule["args"]["interactions"]
    raise ValueError(f"SpeciesInteraction rule was not found in {config_path}.")
