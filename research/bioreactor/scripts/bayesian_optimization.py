import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bioreactor_common import STUDY_DIR, load_interactions
from scipy.special import ndtr
from scipy.stats import qmc
from screening import (
    CONFIGS_DIR,
    calc_rmse,
    get_real_data,
    get_relative_sim_data,
    save_screening_config,
)

from cellular_automata.config.Parser import parse_json

BIO_NUMBER=4
BASE_CONFIG = 'bo'
RESULTS_PATH = STUDY_DIR / 'results' / 'bo_results.csv'
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


def vector_from_interactions(interactions: dict) -> np.ndarray:
    return np.array([
        interactions[source][target] for source, target in SELECTED_PAIRS
    ], dtype=float)


def interactions_from_vector(base_interactions: dict, values: np.ndarray) -> dict:
    interactions = {
        source: dict(targets) for source, targets in base_interactions.items()
    }
    for (source, target), value in zip(SELECTED_PAIRS, values, strict=True):
        interactions[source][target] = float(value)
    return interactions


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    results = pd.read_csv(path)
    value_columns = [f'x{i}' for i in range(len(SELECTED_PAIRS))]
    required = {'run_no', 'rmse', 'config_path', *value_columns}
    if not required.issubset(results.columns):
        raise ValueError(f'{path} is missing BO columns.')
    return results


def save_result(
    run_no: int, values: np.ndarray, rmse: float, config_path: str,
    path: Path = RESULTS_PATH, rmse_sd: float = 0.0, seeds: int = 1,
) -> None:
    row = {
        'run_no': run_no,
        'rmse': rmse,
        'config_path': config_path,
        'rmse_sd': rmse_sd,
        'seeds': seeds,
    }
    row.update({f'x{i}': value for i, value in enumerate(values)})
    new_row = pd.DataFrame([row])
    existing = load_results(path)
    updated = pd.concat([existing, new_row], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)


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


def run_trial(
    run_no: int,
    values: np.ndarray,
    base_interactions: dict,
    real_data: pd.DataFrame,
    seeds: tuple[int, ...] = (RANDOM_SEED,),
    config_name: str = BASE_CONFIG,
    results_path: Path = RESULTS_PATH,
) -> float:
    """Score one point: the mean RMSE over one simulation per seed.

    Every trial uses the same seeds (common random numbers), so RMSE differences come
    from the interactions; more seeds average out more of the simulation noise.
    """
    interactions = interactions_from_vector(base_interactions, values)
    scores = []
    for seed in seeds:
        config_path = save_screening_config(interactions, run_no, config_name=config_name, seed=seed)
        ca_sim, _ = parse_json(config_path)
        relative_sim = get_relative_sim_data(ca_sim)
        scores.append(calc_rmse(relative_sim, real_data, ca_sim.max_dt, ca_sim.tracker.steps))
    rmse = float(np.mean(scores))
    rmse_sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    save_result(
        run_no, values, rmse, str(config_path.relative_to(CONFIGS_DIR)),
        path=results_path, rmse_sd=rmse_sd, seeds=len(scores),
    )
    print(f'run {run_no}: RMSE={rmse:.6f}' + (f' (sd {rmse_sd:.6f}, {len(scores)} seeds)' if len(scores) > 1 else ''))
    return rmse


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Bayesian optimisation of selected interaction coefficients.'
    )
    parser.add_argument('--bio-number', type=int, default=BIO_NUMBER,
                        help='bioreactor whose measurements are fitted')
    parser.add_argument('--config', default=BASE_CONFIG,
                        help='base config in configs/ (without .json)')
    parser.add_argument('--initial-runs', type=int, default=INITIAL_RUNS,
                        help='Sobol points before the GP takes over')
    parser.add_argument('--bo-runs', type=int, default=BO_RUNS, help='GP-proposed points')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='seed for the sampler and the first simulation seed')
    parser.add_argument('--seeds', type=int, default=1,
                        help='simulations per point, seeded seed, seed+1, ...; '
                             'the score is their mean RMSE')
    parser.add_argument('--radius', type=float, default=VALUE_RADIUS,
                        help='search radius around the base coefficients')
    parser.add_argument('--results', type=Path, default=RESULTS_PATH,
                        help='results CSV (an existing one is resumed)')
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error('--seeds must be at least 1')
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = tuple(args.seed + i for i in range(args.seeds))
    base_interactions = load_interactions(args.config)
    baseline = vector_from_interactions(base_interactions)
    real_data = get_real_data(bio_number=args.bio_number)
    rng = np.random.default_rng(args.seed)

    def trial(run_no, point):
        run_trial(run_no, point, base_interactions, real_data, seeds=seeds,
                  config_name=args.config, results_path=args.results)

    results = load_results(args.results)
    if results.empty:
        trial(0, baseline)
        results = load_results(args.results)

    value_columns = [f'x{i}' for i in range(len(SELECTED_PAIRS))]
    values = results[value_columns].to_numpy(dtype=float)
    scores = results['rmse'].to_numpy(dtype=float)
    lower = np.maximum(VALUE_BOUNDS[0], baseline - args.radius)
    upper = np.minimum(VALUE_BOUNDS[1], baseline + args.radius)

    while len(results) < args.initial_runs + args.bo_runs + 1:
        if len(results) < args.initial_runs + 1:
            sampler = qmc.Sobol(d=len(SELECTED_PAIRS), scramble=True, seed=args.seed + len(results))
            point = qmc.scale(sampler.random(1), lower, upper)[0]
        else:
            point = propose(values, scores, lower, upper, rng)

        run_no = int(results['run_no'].max()) + 1
        trial(run_no, point)
        results = load_results(args.results)
        values = results[value_columns].to_numpy(dtype=float)
        scores = results['rmse'].to_numpy(dtype=float)

    best = results.loc[results['rmse'].idxmin()]
    print(f'best RMSE: {best["rmse"]:.6f}')
    print(f'best run: {int(best["run_no"])}')
    print(f'saved results: {args.results}')
    print('best coefficients:')
    for index, (source, target) in enumerate(SELECTED_PAIRS):
        print(f'  {source} -> {target}: {best[f"x{index}"]:.6f}')


if __name__ == '__main__':
    main()
