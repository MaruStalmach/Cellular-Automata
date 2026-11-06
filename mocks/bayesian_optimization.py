import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import qmc

from libs.config.Parser import parse_json
from mocks.screening import calc_rmse, get_real_data, get_relative_sim_data, save_screening_config


BIO_NUMBER=4
BASE_CONFIG = 'bo'
RESULTS_PATH = Path(__file__).resolve().parent.parent / 'optim_results' / 'bo_results.csv'
INITIAL_RUNS = 5
BO_RUNS = 10
RANDOM_SEED = 20260903
VALUE_RADIUS = 0.5
VALUE_BOUNDS = (-1.0, 1.0)

SELECTED_PAIRS = [
    ('Oscillospiraceae','Other_bacteria'),
    ('Bacteroidaceae','Lachnospiraceae'),
    ('Enterobacteriaceae','Bifidobacteriaceae'),
    ('Tannerellaceae','Bifidobacteriaceae'),
    ('Bacteroidaceae','Oscillospiraceae'),
    ('Sutterellaceae','Oscillospiraceae'),
    ('Prevotellaceae','Enterobacteriaceae'),
    ('Bifidobacteriaceae','Sutterellaceae'),
    ('Prevotellaceae','Bifidobacteriaceae'),
    ('Oscillospiraceae','Tannerellaceae'),
]


def load_interactions(config_name: str) -> dict:
    config_path = Path(__file__).resolve().parent.parent / 'libs' / 'config' / f'{config_name}.json'
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    for rule in config['rules']:
        if rule.get('name') == 'SpeciesInteraction':
            return rule['args']['interactions']
    raise ValueError('SpeciesInteraction rule was not found in the config.')


def vector_from_interactions(interactions: dict) -> np.ndarray:
    return np.array([
        interactions[source][target] for source, target in SELECTED_PAIRS
    ], dtype=float)


def interactions_from_vector(base_interactions: dict, values: np.ndarray) -> dict:
    interactions = {
        source: dict(targets) for source, targets in base_interactions.items()
    }
    for (source, target), value in zip(SELECTED_PAIRS, values):
        interactions[source][target] = float(value)
    return interactions


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    results = pd.read_csv(RESULTS_PATH)
    value_columns = [f'x{i}' for i in range(len(SELECTED_PAIRS))]
    required = {'run_no', 'rmse', 'config_path', *value_columns}
    if not required.issubset(results.columns):
        raise ValueError(f'{RESULTS_PATH} is missing BO columns.')
    return results


def save_result(run_no: int, values: np.ndarray, rmse: float, config_path: str) -> None:
    row = {
        'run_no': run_no,
        'rmse': rmse,
        'config_path': config_path,
    }
    row.update({f'x{i}': value for i, value in enumerate(values)})
    new_row = pd.DataFrame([row])
    existing = load_results()
    updated = pd.concat([existing, new_row], ignore_index=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(RESULTS_PATH, index=False)


def rbf_kernel(left: np.ndarray, right: np.ndarray, length_scale: float = 0.35) -> np.ndarray:
    squared_distance = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * squared_distance / length_scale**2)


def expected_improvement(candidates: np.ndarray, observations: np.ndarray, scores: np.ndarray) -> np.ndarray:
    kernel = rbf_kernel(observations, observations)
    noise = 1e-6
    alpha = np.linalg.solve(kernel + noise * np.eye(len(observations)), scores)
    cross_kernel = rbf_kernel(candidates, observations)
    mean = cross_kernel @ alpha
    variance = 1.0 - np.sum(
        cross_kernel * np.linalg.solve(kernel + noise * np.eye(len(observations)), cross_kernel.T).T,
        axis=1,
    )
    standard_deviation = np.sqrt(np.maximum(variance, 1e-12))
    improvement = scores.min() - mean
    z_score = improvement / standard_deviation
    return improvement * ndtr(z_score) + standard_deviation * np.exp(-0.5 * z_score**2) / np.sqrt(2 * np.pi)


def propose(values: np.ndarray, scores: np.ndarray, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    candidate_count = max(4000, 200 * len(SELECTED_PAIRS))
    candidates = rng.uniform(lower, upper, size=(candidate_count, len(SELECTED_PAIRS)))
    acquisition = expected_improvement(candidates, values, scores)
    return candidates[int(np.argmax(acquisition))]


def run_trial(run_no: int, values: np.ndarray, base_interactions: dict, real_data: pd.DataFrame) -> float:
    interactions = interactions_from_vector(base_interactions, values)
    config_path = save_screening_config(interactions, run_no, config_name=BASE_CONFIG)
    np.random.seed(RANDOM_SEED)
    ca_sim, _ = parse_json(config_path)
    relative_sim = get_relative_sim_data(ca_sim)
    rmse = calc_rmse(relative_sim, real_data)
    save_result(run_no, values, rmse, config_path)
    print(f'run {run_no}: RMSE={rmse:.6f}')
    return rmse


def main() -> None:
    base_interactions = load_interactions(BASE_CONFIG)
    baseline = vector_from_interactions(base_interactions)
    real_data = get_real_data(bio_number=BIO_NUMBER)
    rng = np.random.default_rng(RANDOM_SEED)

    results = load_results()
    if results.empty:
        run_trial(0, baseline, base_interactions, real_data)
        results = load_results()

    value_columns = [f'x{i}' for i in range(len(SELECTED_PAIRS))]
    values = results[value_columns].to_numpy(dtype=float)
    scores = results['rmse'].to_numpy(dtype=float)
    lower = np.maximum(VALUE_BOUNDS[0], baseline - VALUE_RADIUS)
    upper = np.minimum(VALUE_BOUNDS[1], baseline + VALUE_RADIUS)

    while len(results) < INITIAL_RUNS + BO_RUNS + 1:
        if len(results) < INITIAL_RUNS + 1:
            sampler = qmc.Sobol(d=len(SELECTED_PAIRS), scramble=True, seed=RANDOM_SEED + len(results))
            point = qmc.scale(sampler.random(1), lower, upper)[0]
        else:
            point = propose(values, scores, lower, upper, rng)

        run_no = int(results['run_no'].max()) + 1
        run_trial(run_no, point, base_interactions, real_data)
        results = load_results()
        values = results[value_columns].to_numpy(dtype=float)
        scores = results['rmse'].to_numpy(dtype=float)

    best = results.loc[results['rmse'].idxmin()]
    print(f'best RMSE: {best["rmse"]:.6f}')
    print(f'best run: {int(best["run_no"])}')
    print(f'saved results: {RESULTS_PATH}')
    print('best coefficients:')
    for index, (source, target) in enumerate(SELECTED_PAIRS):
        print(f'  {source} -> {target}: {best[f"x{index}"]:.6f}')


if __name__ == '__main__':
    main()
