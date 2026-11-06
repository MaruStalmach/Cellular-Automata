"""
Usage: python3 compare_to_reference.py path/to/ratios.csv [--steps-per-day N] [--reference PATH_OR_URL]

Compares your simulated per-family relative abundances against the real
'male' donor dataset from the dynamo repo (the same dataset your
sim_config.json abundances were seeded from -- confirmed by matching row 0
values exactly), and reports per-family MSE so you can see which family's
dynamics diverge the most from the real data.

CAVEAT: the real reference data is one row per real day. We don't yet have a
confirmed mapping from simulation steps to real days. --steps-per-day lets
you collapse N sim steps into 1 "day" (by averaging) before comparing; until
that mapping is nailed down, treat the MSE numbers as directionally useful
(which family diverges most) rather than an absolute, calibrated error.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_REFERENCE = (
    "./dynamo_data/male_taxonomy_for_plot.csv"
)

# maps the reference CSV's full taxonomy-string columns to your short family
# names (same 8 families your sim tracks, post name-fix)
FAMILY_SUFFIX_MAP = {
    "f__Bacteroidaceae": "Bacteroidaceae",
    "f__Muribaculaceae": "Muribaculaceae",
    "f__Tannerellaceae": "Tannerellaceae",
    "f__Lachnospiraceae": "Lachnospiraceae",
    "f__Oscillospiraceae_88309": "Oscillospiraceae_88309",
    "f__Ruminococcaceae": "Ruminococcaceae",
    "f__Peptoniphilaceae": "Peptoniphilaceae",
    "other": "other",
}


def load_reference(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url, index_col=0)

    rename = {}
    for col in df.columns:
        matched = None
        for suffix, short_name in FAMILY_SUFFIX_MAP.items():
            if col == suffix or col.endswith("; " + suffix):
                matched = short_name
                break
        if matched:
            rename[col] = matched
    df = df.rename(columns=rename)

    keep = [v for v in FAMILY_SUFFIX_MAP.values() if v in df.columns]
    missing = [v for v in FAMILY_SUFFIX_MAP.values() if v not in df.columns]
    if missing:
        print(f"[warning] reference data is missing columns for: {missing}")

    return df[keep].reset_index(drop=True)


def load_simulated(csv_path: str, steps_per_day: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    families = [c for c in FAMILY_SUFFIX_MAP.values() if c in df.columns]

    if steps_per_day <= 1:
        return df[families].reset_index(drop=True)

    # collapse every `steps_per_day` rows into one "day" via mean
    df = df[families].copy()
    df["day"] = df.index // steps_per_day
    return df.groupby("day")[families].mean().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ratios_csv", help="path to your simulation's ratios.csv")
    parser.add_argument(
        "--steps-per-day", type=int, default=1,
        help="how many simulation steps make up one real day (default: 1, i.e. step==day)",
    )
    parser.add_argument(
        "--reference", default=DEFAULT_REFERENCE,
        help="path or URL to the reference taxonomy CSV (default: dynamo repo's male_taxonomy_for_plot.csv)",
    )
    args = parser.parse_args()

    sim_df = load_simulated(args.ratios_csv, args.steps_per_day)
    ref_df = load_reference(args.reference)

    n = min(len(sim_df), len(ref_df))
    if n == 0:
        sys.exit("No overlapping time points between simulated and reference data.")

    sim_df = sim_df.iloc[:n].reset_index(drop=True)
    ref_df = ref_df.iloc[:n].reset_index(drop=True)

    families = [c for c in FAMILY_SUFFIX_MAP.values() if c in sim_df.columns and c in ref_df.columns]

    per_family_mse = {}
    per_timepoint_sq_err = pd.DataFrame(index=range(n))
    for fam in families:
        sq_err = (sim_df[fam] - ref_df[fam]) ** 2
        per_timepoint_sq_err[fam] = sq_err
        per_family_mse[fam] = sq_err.mean()

    mse_series = pd.Series(per_family_mse).sort_values(ascending=False)

    print(f"Compared {n} aligned time points (steps_per_day={args.steps_per_day}).\n")
    print(f"{'family':<25}{'MSE':>12}")
    print("-" * 37)
    for fam, mse in mse_series.items():
        print(f"{fam:<25}{mse:>12.6f}")

    plt.figure(figsize=(8, 4))
    mse_series.plot(kind="bar", color="#9b2226")
    plt.ylabel("MSE vs. reference")
    plt.title("Per-family divergence from real (male donor) data")
    plt.tight_layout()
    out_path = args.ratios_csv.rsplit(".", 1)[0] + "_mse_by_family.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved bar chart to {out_path}")

    # also show divergence over time for the worst-offending family, useful
    # for spotting *when* it diverges (e.g. right after the freeze/plateau)
    worst = mse_series.index[0]
    plt.figure(figsize=(10, 3))
    plt.plot(sim_df[worst], label=f"simulated {worst}")
    plt.plot(ref_df[worst], label=f"reference {worst}")
    plt.xlabel("time point")
    plt.ylabel("relative abundance")
    plt.title(f"Worst-diverging family: {worst}")
    plt.legend()
    plt.tight_layout()
    worst_path = args.ratios_csv.rsplit(".", 1)[0] + f"_{worst}_vs_reference.png"
    plt.savefig(worst_path, dpi=150)
    print(f"Saved {worst}-specific comparison to {worst_path}")


if __name__ == "__main__":
    main()