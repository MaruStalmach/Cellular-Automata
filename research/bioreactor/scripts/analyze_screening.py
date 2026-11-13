import argparse
import ast
import sys
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
# screening.py writes here; archived screening runs are on the `results` branch
DEFAULT_INPUT = RESULTS_DIR / 'screening_results.csv'
DEFAULT_OUTPUT = RESULTS_DIR / 'screening_pair_analysis.csv'


def load_screening_results(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path).sort_values('rmse').reset_index(drop=True)
    if 'active_pairs' not in results or 'rmse' not in results:
        raise ValueError("CSV must contain 'rmse' and 'active_pairs' columns")
    results['active_pairs'] = results['active_pairs'].apply(ast.literal_eval)
    return results


def analyze_pairs(results: pd.DataFrame, top_fraction: float = 0.25) -> pd.DataFrame:
    top_count = max(1, int(round(len(results) * top_fraction)))
    top_results = results.iloc[:top_count]
    other_results = results.iloc[top_count:]

    pair_records = []
    for run_group, group in [('top', top_results), ('other', other_results)]:
        for pairs in group['active_pairs']:
            for source, target, value in pairs:
                pair_records.append({
                    'group': run_group,
                    'source': source,
                    'target': target,
                    'value': float(value),
                })

    pair_data = pd.DataFrame(pair_records)
    all_pairs = pd.MultiIndex.from_product(
        [sorted(pair_data['source'].unique()), sorted(pair_data['target'].unique())],
        names=['source', 'target'],
    ).to_frame(index=False)
    all_pairs = all_pairs[all_pairs['source'] != all_pairs['target']]

    summary = []
    for row in all_pairs.itertuples(index=False):
        pair = pair_data[
            (pair_data['source'] == row.source) &
            (pair_data['target'] == row.target)
        ]
        top_values = pair.loc[pair['group'] == 'top', 'value']
        other_values = pair.loc[pair['group'] == 'other', 'value']
        summary.append({
            'source': row.source,
            'target': row.target,
            'top_count': len(top_values),
            'other_count': len(other_values),
            'top_rate': len(top_values) / len(top_results),
            'other_rate': len(other_values) / max(1, len(other_results)),
            'rate_difference': (
                len(top_values) / len(top_results)
                - len(other_values) / max(1, len(other_results))
            ),
            'top_mean_value': top_values.mean(),
            'other_mean_value': other_values.mean(),
        })

    return pd.DataFrame(summary).sort_values(
        ['rate_difference', 'top_count', 'top_mean_value'],
        ascending=[False, False, False],
        na_position='last',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Find interaction pairs enriched in the best screening runs.')
    parser.add_argument('-i', '--input', type=Path, default=DEFAULT_INPUT,
                        help=f'screening results CSV (default: {DEFAULT_INPUT})')
    parser.add_argument('-o', '--output', type=Path, default=DEFAULT_OUTPUT,
                        help=f'where to write the pair analysis CSV (default: {DEFAULT_OUTPUT})')
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(
            f'{args.input} not found. Run screening.py first, or pass --input '
            f'(archived screening results are on the `results` branch).'
        )

    results = load_screening_results(args.input)
    pair_analysis = analyze_pairs(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pair_analysis.to_csv(args.output, index=False)

    top_count = max(1, int(round(len(results) * 0.25)))
    print(f'runs: {len(results)}')
    print(f'best RMSE: {results.rmse.iloc[0]:.6f}')
    print(f'median RMSE: {results.rmse.median():.6f}')
    print(f'top group: {top_count} runs')
    print('\nBest runs:')
    print(results[['rmse']].head(top_count).to_string(index=True))
    print('\nPairs enriched in the best group:')
    print(pair_analysis.head(15).to_string(index=False))
    print(f'\nSaved pair analysis to {args.output}')


if __name__ == '__main__':
    main()
