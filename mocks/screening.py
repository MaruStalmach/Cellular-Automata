import json
from pathlib import Path

import numpy as np
import pandas as pd

from libs.config.Parser import parse_json
from time import time

SAMPLE_RUNS = 12
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
BASE_CONFIG = 'screen'

BASE_INTERACTIONS = {
    'Bacteroidaceae': {'Bacteroidaceae': 0.0, 'Lachnospiraceae': 0.0, 'Enterobacteriaceae': -0.2, 'Synergistaceae': 0.0, 'Prevotellaceae': -0.9, 'Ruminococcaceae': 0.0, 'Muribaculaceae': -0.9, 'Oscillospiraceae': 0.0, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.5, 'Other_bacteria': -0.3},
    'Lachnospiraceae': {'Bacteroidaceae': 0.0, 'Lachnospiraceae': 0.0, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.5, 'Ruminococcaceae': 0.5, 'Muribaculaceae': 0.5, 'Oscillospiraceae': 0.5, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.5, 'Other_bacteria': -0.3},
    'Enterobacteriaceae': {'Bacteroidaceae': -0.2, 'Lachnospiraceae': -0.5, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': -0.5, 'Muribaculaceae': 0.0, 'Oscillospiraceae': -0.5, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.0, 'Other_bacteria': -0.3},
    'Synergistaceae': {'Bacteroidaceae': 0.0, 'Lachnospiraceae': 0.0, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': 0.0, 'Muribaculaceae': 0.0, 'Oscillospiraceae': 0.0, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.0, 'Other_bacteria': -0.3},
    'Prevotellaceae': {'Bacteroidaceae': -0.9, 'Lachnospiraceae': 0.2, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': 0.0, 'Muribaculaceae': 0.0, 'Oscillospiraceae': 0.0, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.0, 'Other_bacteria': -0.3},
    'Ruminococcaceae': {'Bacteroidaceae': 0.0, 'Lachnospiraceae': 0.5, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': 0.0, 'Muribaculaceae': 0.0, 'Oscillospiraceae': 0.0, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.0, 'Other_bacteria': -0.3},
    'Muribaculaceae': {'Bacteroidaceae': -0.2, 'Lachnospiraceae': 0.0, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': 0.0, 'Muribaculaceae': 0.0, 'Oscillospiraceae': 0.0, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.5, 'Other_bacteria': -0.3},
    'Oscillospiraceae': {'Bacteroidaceae': 0.0, 'Lachnospiraceae': 0.4, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': 0.0, 'Muribaculaceae': 0.0, 'Oscillospiraceae': 0.0, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.0, 'Other_bacteria': -0.3},
    'Sutterellaceae': {'Bacteroidaceae': 0.0, 'Lachnospiraceae': 0.0, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': 0.0, 'Muribaculaceae': 0.0, 'Oscillospiraceae': 0.0, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.0, 'Other_bacteria': -0.3},
    'Tannerellaceae': {'Bacteroidaceae': 0.0, 'Lachnospiraceae': 0.0, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': 0.0, 'Muribaculaceae': 0.0, 'Oscillospiraceae': 0.0, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.0, 'Other_bacteria': -0.3},
    'Bifidobacteriaceae': {'Bacteroidaceae': 0.0, 'Lachnospiraceae': 0.2, 'Enterobacteriaceae': 0.0, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': 0.3, 'Muribaculaceae': 0.4, 'Oscillospiraceae': 0.5, 'Sutterellaceae': 0.0, 'Tannerellaceae': -0.5, 'Bifidobacteriaceae': 0.0, 'Other_bacteria': -0.3},
    'Other_bacteria': {'Bacteroidaceae': 0.0, 'Lachnospiraceae': 0.0, 'Enterobacteriaceae': 0.2, 'Synergistaceae': 0.0, 'Prevotellaceae': 0.0, 'Ruminococcaceae': 0.0, 'Muribaculaceae': 0.0, 'Oscillospiraceae': 0.0, 'Sutterellaceae': 0.0, 'Tannerellaceae': 0.0, 'Bifidobacteriaceae': 0.0, 'Other_bacteria': -0.3},
}


def get_real_data(bio_number=BIO_NUMBER) -> pd.DataFrame:
    df_meta = pd.read_csv('dataset/meta_final.csv', index_col=0)
    filtered_families = pd.read_csv('dataset/filtered_families_fixed.csv', index_col=0)
    idx = df_meta[df_meta.Bioreactor == bio_number].sort_values(by=['Bioreactor_run', 'Day_Steady_state', 'Hour_Treated']).index[:25]
    df_meta_nona = df_meta.fillna(0)
    time_df = (
        df_meta_nona.loc[idx, 'Day_Steady_state'] * 24 +
        df_meta_nona.loc[idx, 'Hour_Treated'] % 24
    )
    df_out = filtered_families.loc[idx].div(filtered_families.loc[idx].sum(axis=1), axis=0)
    df_out['time'] = time_df
    return df_out


def calc_rmse(relative_sim, relative_data):
    time_idx = np.asarray(relative_data['time'].values * 2, dtype=int)
    sim_data_at_experiment_timesteps = relative_sim[:, time_idx]
    mse_total = 0.0
    for i in range(len(relative_data)):
        species_errors = []
        for j in range(len(BACTERIA_KEYS)):
            species_name = BACTERIA_KEYS[j]
            if species_name == 'Other_bacteria':
                species_name = "Rest"
            species_errors.append(
                (relative_data.iloc[i][species_name] - sim_data_at_experiment_timesteps[j, i]) ** 2
            )
        tmp = np.mean(species_errors)
        mse_total += tmp
    return float(np.sqrt(mse_total / len(relative_data)))


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


def save_screening_config(interactions, run_no: int, config_name: str = 'screen'):
    base_dir = Path(__file__).resolve().parent.parent / 'libs' / 'config'
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

    config_path = config_dir / f'{config_name}_{run_no}.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    return f'screening_runs/{config_path.name}'


def run_screening_trial(run_no: int, df_real_data: pd.DataFrame, base_matrix: dict | None = None):
    if base_matrix is None:
        base_matrix = BASE_INTERACTIONS

    random_matrix, selected_pairs = build_sparse_interactions(base_matrix, active_pairs=18, value_scale=0.8)
    config_rel_path = save_screening_config(random_matrix, run_no)
    ca_sim, _ = parse_json(config_rel_path)

    relative_sim = get_relative_sim_data(ca_sim)
    rmse = calc_rmse(relative_sim, df_real_data)

    return {
        'run_no': run_no,
        'rmse': rmse,
        'interactions': random_matrix,
        'active_pairs': selected_pairs,
        'config_path': config_rel_path,
    }


if __name__ == '__main__':
    df_real_data = get_real_data()
    results = []
    for run_no in range(SAMPLE_RUNS):
        result = run_screening_trial(run_no, df_real_data, BASE_INTERACTIONS)
        results.append(result)
        print(f'run {run_no}: RMSE={result["rmse"]:.6f}')

        df_results = pd.DataFrame([
            {
                'run_no': row['run_no'],
                'rmse': row['rmse'],
                'active_pairs': row['active_pairs'],
                'config_path': row['config_path'],
            }
            for row in results
        ])
        df_results.to_csv(Path(__file__).resolve().parent / 'screening_results.csv', index=False)
        print('Saved screening results to', Path(__file__).resolve().parent / 'screening_results.csv')
        
          