"""
Usage: python3 check_occupancy.py path/to/ratios.csv

Plots occupied_cells / total_cells over time. If this climbs to ~1.0 and
flattens, the grid has re-saturated (no empty cells left anywhere), which
means every spawn-into-empty-cell rule (SpeciesInteraction's symbiotic
spawn, SpontaneousSpawn) can no longer place anything -- that's the freeze.
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    if "occupied_cells" not in df.columns or "total_cells" not in df.columns:
        sys.exit(
            "ratios.csv doesn't have occupied_cells/total_cells columns -- "
            "make sure you're running the patched experiment.py that records them."
        )

    df["occupancy_fraction"] = df["occupied_cells"] / df["total_cells"]

    print(f"{'step':>6}  {'occupancy_fraction':>20}")
    print("-" * 30)
    for _, row in df.iterrows():
        print(f"{int(row['step']):>6}  {row['occupancy_fraction']:>20.4f}")

    plateau_at = df.loc[df["occupancy_fraction"] >= 0.999, "step"]
    if not plateau_at.empty:
        print(f"\nGrid hits >=99.9% occupancy at step {int(plateau_at.iloc[0])}.")
    else:
        print(f"\nGrid never reaches 99.9% occupancy (max = {df['occupancy_fraction'].max():.4f}).")

    plt.figure(figsize=(10, 3))
    plt.plot(df["step"], df["occupancy_fraction"], marker="o")
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1, label="100% occupied")
    plt.xlabel("step")
    plt.ylabel("occupied_cells / total_cells")
    plt.title("Grid occupancy over time")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    out_path = csv_path.rsplit(".", 1)[0] + "_occupancy.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 check_occupancy.py test/ratios.csv")
    main(sys.argv[1])