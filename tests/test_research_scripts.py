"""Research scripts pass a seed into the configs they generate (REPRO-1)."""

import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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

    def no_global_seed(_seed):
        raise AssertionError("rules draw from self.rng; seed through sim.seed instead")

    monkeypatch.setattr(bo.np.random, "seed", no_global_seed)

    for run_no in (1, 2):
        try:
            bo.run_trial(run_no, values=None, base_interactions={}, real_data=None)
        except _StopTrial:
            pass

    assert [call.get("seed") for call in calls] == [bo.RANDOM_SEED, bo.RANDOM_SEED]


# --- calc_rmse maps measurement times to frames through the step length (RES-2) ---


def _old_calc_rmse(screening, relative_sim, relative_data):
    """calc_rmse before RES-2: assumed 1 step = 30 min and one frame per step."""
    time_idx = np.asarray(relative_data["time"].values * 2, dtype=int)
    sim = relative_sim[:, time_idx]
    total = 0.0
    for i in range(len(relative_data)):
        errors = []
        for j, name in enumerate(screening.BACTERIA_KEYS):
            name = "Rest" if name == "Other_bacteria" else name
            errors.append((relative_data.iloc[i][name] - sim[j, i]) ** 2)
        total += np.mean(errors)
    return float(np.sqrt(total / len(relative_data)))


def _random_sim(screening, n_frames, seed=0):
    sim = np.random.default_rng(seed).random((len(screening.BACTERIA_KEYS), n_frames))
    return sim / sim.sum(axis=0)


def test_calc_rmse_matches_the_old_result_at_30_minute_steps(monkeypatch, tmp_path):
    screening = _screening(monkeypatch, tmp_path)
    monkeypatch.setattr(screening, "STUDY_DIR", REPO_ROOT / "research" / "bioreactor")
    data = screening.get_real_data()
    sim = _random_sim(screening, n_frames=1249)

    new = screening.calc_rmse(sim, data, step_seconds=1800)

    assert new == pytest.approx(_old_calc_rmse(screening, sim, data), rel=1e-12)


def test_calc_rmse_follows_the_step_length_and_recorded_steps(monkeypatch, tmp_path):
    screening = _screening(monkeypatch, tmp_path)
    data = pd.DataFrame({key: [0.5, 0.25] for key in screening.DATA_COLUMNS.values()})
    data["time"] = [1.0, 2.0]  # hours
    sim = _random_sim(screening, n_frames=5)

    # 30-minute steps, every second one recorded: frames are steps 0, 2, 4, 6, 8,
    # so hour 1 (step 2) is frame 1 and hour 2 (step 4) is frame 2
    every_other = screening.calc_rmse(sim, data, step_seconds=1800, recorded_steps=[0, 2, 4, 6, 8])
    # one-hour steps, all recorded: hour 1 is frame 1 and hour 2 is frame 2 as well
    hourly = screening.calc_rmse(sim, data, step_seconds=3600)
    # 30-minute steps, all recorded: hours 1 and 2 are frames 2 and 4
    half_hourly = screening.calc_rmse(sim, data, step_seconds=1800)

    assert every_other == pytest.approx(hourly)
    assert half_hourly == pytest.approx(screening.calc_rmse(sim[:, [0, 2, 4]], data, 3600))


def test_calc_rmse_rejects_a_run_shorter_than_the_data(monkeypatch, tmp_path):
    screening = _screening(monkeypatch, tmp_path)
    data = pd.DataFrame({key: [1.0] for key in screening.DATA_COLUMNS.values()})
    data["time"] = [10.0]

    with pytest.raises(ValueError, match=r"no recorded frame for steps \[20\]"):
        screening.calc_rmse(_random_sim(screening, n_frames=5), data, step_seconds=1800)


def test_calc_rmse_rejects_times_between_steps(monkeypatch, tmp_path):
    screening = _screening(monkeypatch, tmp_path)
    data = pd.DataFrame({key: [1.0] for key in screening.DATA_COLUMNS.values()})
    data["time"] = [0.25]

    with pytest.raises(ValueError, match="not whole steps"):
        screening.calc_rmse(_random_sim(screening, n_frames=5), data, step_seconds=1800)


# --- shared helpers (RES-2) ----------------------------------------------------------


def _common(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    sys.modules.pop("bioreactor_common", None)
    return importlib.import_module("bioreactor_common")


def test_load_interactions_reads_the_species_interaction_rule(monkeypatch):
    common = _common(monkeypatch)
    config = json.loads((REPO_ROOT / "research/bioreactor/configs/screen.json").read_text())
    (expected,) = [r["args"]["interactions"] for r in config["rules"] if r["name"] == "SpeciesInteraction"]

    assert common.load_interactions("screen") == expected


def test_load_interactions_without_the_rule_raises(monkeypatch, tmp_path):
    common = _common(monkeypatch)
    (tmp_path / "plain.json").write_text(json.dumps({"rules": [{"name": "Diffusion", "args": {}}]}))

    with pytest.raises(ValueError, match="SpeciesInteraction rule was not found"):
        common.load_interactions("plain", configs_dir=tmp_path)


def test_scripts_share_one_study_dir(monkeypatch, tmp_path):
    common = _common(monkeypatch)
    screening = _screening(monkeypatch, tmp_path)
    sys.modules.pop("analyze_screening", None)
    analyze = importlib.import_module("analyze_screening")

    assert common.STUDY_DIR == REPO_ROOT / "research" / "bioreactor"
    assert analyze.RESULTS_DIR == common.STUDY_DIR / "results"
    assert screening.CONFIGS_DIR == common.CONFIGS_DIR


# --- command-line options (RES-2) ------------------------------------------------------


def _bo(monkeypatch, tmp_path):
    _screening(monkeypatch, tmp_path)
    sys.modules.pop("bayesian_optimization", None)
    return importlib.import_module("bayesian_optimization")


def test_option_defaults_are_the_previous_constants(monkeypatch, tmp_path):
    bo = _bo(monkeypatch, tmp_path)
    screening = sys.modules["screening"]

    s = screening.parse_args([])
    b = bo.parse_args([])

    assert (s.runs, s.bio_number, s.config, s.active_pairs, s.value_scale, s.seed) == (
        12, 4, "screen", 18, 0.8, None
    )
    assert (b.bio_number, b.config, b.initial_runs, b.bo_runs, b.seed, b.seeds, b.radius) == (
        4, "bo", 5, 10, bo.RANDOM_SEED, 1, 0.5
    )


def test_bo_rejects_zero_seeds(monkeypatch, tmp_path):
    bo = _bo(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        bo.parse_args(["--seeds", "0"])


def test_bo_trial_scores_the_mean_over_its_seeds(monkeypatch, tmp_path):
    from types import SimpleNamespace

    bo = _bo(monkeypatch, tmp_path)
    used_seeds = []
    rmse_by_seed = {7: 0.2, 8: 0.4, 9: 0.6}

    def save_config(interactions, run_no, config_name, seed):
        used_seeds.append(seed)
        return bo.CONFIGS_DIR / f"{config_name}_{run_no}.json"

    monkeypatch.setattr(bo, "save_screening_config", save_config)
    monkeypatch.setattr(bo, "interactions_from_vector", lambda base, values: {})
    monkeypatch.setattr(
        bo, "parse_json",
        lambda path: (SimpleNamespace(max_dt=1800, tracker=SimpleNamespace(steps=None)), None),
    )
    monkeypatch.setattr(bo, "get_relative_sim_data", lambda sim: None)
    monkeypatch.setattr(bo, "calc_rmse", lambda *args: rmse_by_seed[used_seeds[-1]])
    results = tmp_path / "bo.csv"

    score = bo.run_trial(
        3, np.zeros(len(bo.SELECTED_PAIRS)), {}, None, seeds=(7, 8, 9), results_path=results
    )

    assert used_seeds == [7, 8, 9]
    assert score == pytest.approx(0.4)
    row = pd.read_csv(results).iloc[0]
    assert (row["run_no"], row["seeds"]) == (3, 3)
    assert row["rmse_sd"] == pytest.approx(0.2)


def test_screening_main_passes_its_options_through(monkeypatch, tmp_path):
    screening = _screening(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(screening, "get_real_data", lambda bio_number: f"bio{bio_number}")
    monkeypatch.setattr(screening, "load_interactions", lambda name: {"from": name})

    def trial(run_no, data, base, **kwargs):
        calls.append((run_no, data, base, kwargs))
        return {"run_no": run_no, "seed": 0, "rmse": 0.1, "active_pairs": [], "config_path": "x"}

    monkeypatch.setattr(screening, "run_screening_trial", trial)
    output = tmp_path / "out.csv"

    screening.main([
        "--runs", "2", "--bio-number", "3", "--config", "bo", "--active-pairs", "5",
        "--value-scale", "0.3", "--seed", "1", "--output", str(output),
    ])

    assert [c[0] for c in calls] == [0, 1]
    _, data, base, kwargs = calls[0]
    assert (data, base) == ("bio3", {"from": "bo"})
    assert (kwargs["config_name"], kwargs["active_pairs"], kwargs["value_scale"]) == ("bo", 5, 0.3)
    assert len(pd.read_csv(output)) == 2
