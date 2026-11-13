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


class _StopTrial(Exception):
    pass


def test_bayesian_optimization_seeds_every_trial_through_the_config(monkeypatch, tmp_path):
    _screening(monkeypatch, tmp_path)
    sys.modules.pop("bayesian_optimization", None)
    bo = importlib.import_module("bayesian_optimization")
    calls = []

    def record_config(*args, **kwargs):
        calls.append(kwargs)
        return tmp_path / "unused.json"

    def stop(_path):
        raise _StopTrial

    monkeypatch.setattr(bo, "save_screening_config", record_config)
    monkeypatch.setattr(bo, "parse_json", stop)
    monkeypatch.setattr(bo, "interactions_from_vector", lambda base, values: {})

    for run_no in (1, 2):
        try:
            bo.run_trial(run_no, values=None, base_interactions={}, real_data=None)
        except _StopTrial:
            pass

    assert [call.get("seed") for call in calls] == [bo.RANDOM_SEED, bo.RANDOM_SEED]
