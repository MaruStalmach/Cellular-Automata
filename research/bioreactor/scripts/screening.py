import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from bioreactor_common import CONFIGS_DIR, STUDY_DIR, load_interactions

from cellular_automata.config.Parser import parse_json

SAMPLE_RUNS = 12
ACTIVE_PAIRS = 18
VALUE_SCALE = 0.8
BIO_NUMBER = 4
IDX_TO_NAME = {
    0: 'Bacteroidaceae',
    1: 'Lachnospiraceae',
    2: 'Enterobacteriaceae',
    3: 'Synergistaceae',
    4: 'Prevotellaceae',
    5: 'Ruminococcaceae',
    6: 'Muribaculaceae',
    7: 'Oscillospiraceae',
    8: 'Sutterellaceae',
    9: 'Tannerellaceae',
    10: 'Bifidobacteriaceae',
    11: 'Other_bacteria',
}
BACTERIA_KEYS = list(IDX_TO_NAME.values())
# column in filtered_families_fixed.csv for each simulated species
DATA_COLUMNS = {key: key for key in BACTERIA_KEYS} | {'Other_bacteria': 'Rest'}
SECONDS_PER_HOUR = 3600
# the screening starts from the interactions in configs/<BASE_CONFIG>.json
BASE_CONFIG = 'screen'


def get_real_data(bio_number=BIO_NUMBER) -> pd.DataFrame:
    df_meta = pd.read_csv(STUDY_DIR / 'data' / 'meta_final.csv', index_col=0)
    filtered_families = pd.read_csv(STUDY_DIR / 'data' / 'filtered_families_fixed.csv', index_col=0)
    idx = df_meta[df_meta.Bioreactor == bio_number].sort_values(by=['Bioreactor_run', 'Day_Steady_state', 'Hour_Treated']).index[:25]
    df_meta_nona = df_meta.fillna(0)
    time_df = (
        df_meta_nona.loc[idx, 'Day_Steady_state'] * 24 +
        df_meta_nona.loc[idx, 'Hour_Treated'] % 24
    )
    df_out = filtered_families.loc[idx].div(filtered_families.loc[idx].sum(axis=1), axis=0)
    df_out['time'] = time_df
    return df_out


def frames_for_times(
    hours, step_seconds: float, recorded_steps: list[int] | None, n_frames: int
) -> np.ndarray:
    """Tracker frame index for each measurement time (in hours).

    `step_seconds` is the simulation's step length (`ca_sim.max_dt`), and
    `recorded_steps` the step number of each frame (`tracker.steps`; None means one
    frame per step). Raises ValueError if a time isn't a whole number of steps, or
    falls on a step that wasn't recorded or after the run ended.
    """
    steps_exact = np.asarray(hours, dtype=float) * SECONDS_PER_HOUR / step_seconds
    steps = np.rint(steps_exact).astype(int)
    if not np.allclose(steps_exact, steps):
        raise ValueError(
            f'measurement times are not whole steps of {step_seconds} s: {sorted(set(hours))}'
        )

    recorded = list(range(n_frames)) if recorded_steps is None else list(recorded_steps)
    frame_of_step = {step: frame for frame, step in enumerate(recorded)}
    missing = sorted({int(step) for step in steps if step not in frame_of_step})
    if missing:
        raise ValueError(
            f'no recorded frame for steps {missing} (the run recorded steps '
            f'{recorded[0]}..{recorded[-1]}); run longer or record those steps'
        )
    return np.array([frame_of_step[step] for step in steps])


def calc_rmse(relative_sim, relative_data, step_seconds: float, recorded_steps=None):
    """RMSE between simulated and measured relative abundances at the measurement times.

    `relative_sim` has one row per species in BACTERIA_KEYS and one column per tracker
    frame; see frames_for_times for `step_seconds` and `recorded_steps`.
    """
    frames = frames_for_times(
        relative_data['time'].values, step_seconds, recorded_steps, relative_sim.shape[1]
    )
    sim_at_measurements = relative_sim[:, frames]
    measured = relative_data[[DATA_COLUMNS[key] for key in BACTERIA_KEYS]].to_numpy().T
    per_measurement_mse = np.mean((measured - sim_at_measurements) ** 2, axis=0)
    return float(np.sqrt(per_measurement_mse.mean()))


def key_abundance_over_time(arr: np.ndarray, layers: int | None = None) -> np.ndarray:
    """Return the number of occupied spatial cells per timestep for a tracked species."""
    if arr.ndim < 2:
        raise ValueError(f'Expected time + spatial dimensions, got shape {arr.shape}')

    if layers is not None and layers > 1:
        cell_nonzero = np.any(arr != 0, axis=-1)
    elif layers is None and arr.ndim >= 5:
        cell_nonzero = np.any(arr != 0, axis=-1)
    else:
        cell_nonzero = arr != 0

    return cell_nonzero.reshape(cell_nonzero.shape[0], -1).sum(axis=1)


def get_relative_sim_data(ca_sim):
    while ca_sim.step() == 0:
        if ca_sim.step_no%20==0:
            print(f'sim at step {ca_sim.step_no}/{ca_sim.max_steps}, time elapsed={time()-ca_sim.start_time:.2f}s')

    stacked_history = {
        key: np.stack(arrays, axis=0) for key, arrays in ca_sim.tracker.history.items()
    }
    abundance_series = []
    for key in BACTERIA_KEYS:
        arr = stacked_history[key]
        abundance_series.append(key_abundance_over_time(arr, 1))

    abundance_matrix = np.vstack(abundance_series)
    relative_totals = abundance_matrix.sum(axis=0, keepdims=True)
    relative_totals = np.where(relative_totals == 0, 1, relative_totals)
    return abundance_matrix / relative_totals


def build_sparse_interactions(base_interactions, active_pairs: int = 18, value_scale: float = 0.8, rng: np.random.Generator | None = None):
    """Randomly perturb a sparse subset of interaction entries while keeping most entries at baseline."""
    if rng is None:
        rng = np.random.default_rng()

    interactions = {
        species_a: dict(base_interactions.get(species_a, {}))
        for species_a in BACTERIA_KEYS
    }

    for species in BACTERIA_KEYS:
        for other in BACTERIA_KEYS:
            interactions.setdefault(species, {})
            interactions[species].setdefault(other, 0.0)

    pair_list = [
        (species_a, species_b)
        for species_a in BACTERIA_KEYS
        for species_b in BACTERIA_KEYS
        if species_a != species_b
    ]

    selected = rng.choice(len(pair_list), size=min(active_pairs, len(pair_list)), replace=False)
    selected_pairs = []
    for idx in selected:
        species_a, species_b = pair_list[int(idx)]
        value = float(rng.uniform(-value_scale, value_scale))
        interactions[species_a][species_b] = value
        selected_pairs.append((species_a, species_b, value))

    return interactions, selected_pairs


def save_screening_config(interactions, run_no: int, config_name: str = 'screen', seed: int | None = None):
    base_dir = STUDY_DIR / 'configs'
    config_dir = base_dir / 'screening_runs'
    config_dir.mkdir(exist_ok=True)

    template_path = base_dir / f'{config_name}.json'
    with open(template_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    for rule in config.get('rules', []):
        if rule.get('name') == 'SpeciesInteraction':
            rule['args']['interactions'] = interactions
            break
    else:
        raise ValueError('SpeciesInteraction rule was not found in the config JSON.')

    if seed is not None:
        config['sim']['seed'] = seed

    config_path = config_dir / f'{config_name}_{run_no}.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    return config_path


def run_screening_trial(
    run_no: int,
    df_real_data: pd.DataFrame,
    base_matrix: dict | None = None,
    config_name: str = BASE_CONFIG,
    active_pairs: int = ACTIVE_PAIRS,
    value_scale: float = VALUE_SCALE,
    rng: np.random.Generator | None = None,
):
    if base_matrix is None:
        base_matrix = load_interactions(config_name)

    random_matrix, selected_pairs = build_sparse_interactions(
        base_matrix, active_pairs=active_pairs, value_scale=value_scale, rng=rng
    )
    config_path = save_screening_config(random_matrix, run_no, config_name=config_name)
    ca_sim, _ = parse_json(config_path)

    relative_sim = get_relative_sim_data(ca_sim)
    rmse = calc_rmse(relative_sim, df_real_data, ca_sim.max_dt, ca_sim.tracker.steps)

    return {
        'run_no': run_no,
        'seed': ca_sim.seed,
        'rmse': rmse,
        'interactions': random_matrix,
        'active_pairs': selected_pairs,
        'config_path': str(config_path.relative_to(CONFIGS_DIR)),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Screen random sparse perturbations of the interaction matrix.'
    )
    parser.add_argument('--runs', type=int, default=SAMPLE_RUNS, help='number of trials')
    parser.add_argument('--bio-number', type=int, default=BIO_NUMBER,
                        help='bioreactor whose measurements are fitted')
    parser.add_argument('--config', default=BASE_CONFIG,
                        help='base config in configs/ (without .json)')
    parser.add_argument('--active-pairs', type=int, default=ACTIVE_PAIRS,
                        help='interaction entries perturbed per trial')
    parser.add_argument('--value-scale', type=float, default=VALUE_SCALE,
                        help='perturbed entries are drawn from [-scale, scale]')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for drawing the perturbations (default: unseeded)')
    parser.add_argument('--output', type=Path, default=STUDY_DIR / 'results' / 'screening_results.csv')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    df_real_data = get_real_data(bio_number=args.bio_number)
    base_interactions = load_interactions(args.config)
    rng = np.random.default_rng(args.seed)
    results = []
    for run_no in range(args.runs):
        result = run_screening_trial(
            run_no, df_real_data, base_interactions, config_name=args.config,
            active_pairs=args.active_pairs, value_scale=args.value_scale, rng=rng,
        )
        results.append(result)
        print(f'run {run_no}: RMSE={result["rmse"]:.6f}')

    df_results = pd.DataFrame([
        {
            'run_no': row['run_no'],
            'seed': row['seed'],
            'rmse': row['rmse'],
            'active_pairs': row['active_pairs'],
            'config_path': row['config_path'],
        }
        for row in results
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.output, index=False)
    print('Saved screening results to', args.output)


if __name__ == '__main__':
    main()
