"""
Print relative abundance ratios of each bacteria family from a single
simulation step .npz file (e.g. step_0000.npz).

Usage:
    python3 print_ratios.py step_0000.npz
    python3 print_ratios.py step_0000.npz --include-layers biofilm floating_bacteria
"""

import argparse
from pathlib import Path

import numpy as np

# these keys aren't bacteria families -- exclude them from ratio calculations
# by default (same convention used in the analysis notebook)
DEFAULT_EXCLUSIONS = {"biofilm", "floating_bacteria"}


def load_ratios(file_path: str | Path, exclude: set[str] = DEFAULT_EXCLUSIONS) -> dict[str, float]:
    """Loads an .npz step file and returns {key: mean_value} for every key
    not in `exclude`. mean_value is the fraction of cells where that key is
    present (i.e. its relative abundance across the grid)."""
    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with np.load(path, allow_pickle=True) as data:
            if not data.files:
                raise ValueError(f"{path.name} contains no arrays")

            ratios = {
                key: float(np.mean(data[key]))
                for key in data.files
                if key not in exclude
            }
    except Exception as e:
        raise RuntimeError(f"Failed to process {path.name}: {e}") from e

    if not ratios:
        raise ValueError(
            f"No bacteria keys found in {path.name} after excluding {exclude}"
        )

    return ratios


def print_ratios(ratios: dict[str, float]) -> None:
    total = sum(ratios.values())
    name_width = max(len(k) for k in ratios) + 2

    print(f"{'family':<{name_width}}{'ratio':>10}{'% of total':>14}")
    print("-" * (name_width + 24))

    for key, value in sorted(ratios.items(), key=lambda kv: kv[1], reverse=True):
        pct_of_total = (value / total * 100) if total > 0 else 0.0
        print(f"{key:<{name_width}}{value:>10.6f}{pct_of_total:>13.2f}%")

    print("-" * (name_width + 24))
    print(f"{'sum':<{name_width}}{total:>10.6f}{'':>14}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print bacteria family ratios from a simulation step .npz file"
    )
    parser.add_argument("file", type=str, help="Path to the step_*.npz file")
    parser.add_argument(
        "--include-layers",
        nargs="*",
        default=[],
        help="Non-species layer names to include instead of excluding "
        "(e.g. biofilm floating_bacteria)",
    )
    args = parser.parse_args()

    exclude = DEFAULT_EXCLUSIONS - set(args.include_layers)
    ratios = load_ratios(args.file, exclude=exclude)
    print_ratios(ratios)


if __name__ == "__main__":
    main()