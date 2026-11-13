"""Research scripts pass a seed into the configs they generate (REPRO-1)."""

import importlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "research" / "bioreactor" / "scripts"


def _screening(monkeypatch, tmp_path: Path):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    sys.modules.pop("screening", None)
    screening = importlib.import_module("screening")
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy(REPO_ROOT / "research/bioreactor/configs/screen.json", configs)
    monkeypatch.setattr(screening, "STUDY_DIR", tmp_path)
    return screening


def test_screening_config_records_the_given_seed(monkeypatch, tmp_path: Path):
    screening = _screening(monkeypatch, tmp_path)

    path = screening.save_screening_config({}, run_no=1, seed=123)

    assert json.loads(path.read_text())["sim"]["seed"] == 123


def test_bayesian_optimization_seeds_through_the_config():
    source = (SCRIPTS_DIR / "bayesian_optimization.py").read_text()

    assert "np.random.seed" not in source
    assert "seed=RANDOM_SEED" in source
