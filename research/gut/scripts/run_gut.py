"""Run one gut simulation from a config and compare its final composition with the data.

The config is an ordinary simulation config (it builds with `parse_json`) plus
two blocks the parser ignores:

- `abundances` - `before` (the starting composition, one entry per species
  state key) and `after` (the target). Each must sum to 1.
- `initial_state` - how the starting grid is drawn, all lengths in grid cells:
  `mean_thickness`, `thickness_spread` and `patch_size` for the biofilm mat,
  and `lumen_density` for the floating cells above it.

The starting grid is built in three steps, all seeded from `sim.seed`:

1. A biofilm mat grows from every wall of the `BiofilmDetachment` rule. Its
   height over each wall cell is a smooth random landscape: white noise blurred
   so that mounds are about `patch_size` cells across, scaled to `mean_thickness` +/-
   `thickness_spread` and cut off at 0. Every column is filled from the wall up
   to its height, so the mat is linked to the wall by construction.
2. Every cell outside the mat is a floating bacterium with probability
   `lumen_density`.
3. Mat and lumen cells together get species in the `before` proportions; the
   cells lost to rounding go to the most abundant species.

Run from the repository root:

    python research/gut/scripts/run_gut.py
    python research/gut/scripts/run_gut.py --config <config.json> --seed 3
"""

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from cellular_automata.config.Parser import build_simulation
from cellular_automata.registry import TRACKERS
from cellular_automata.rules import BiofilmDetachment, GutDrift, SpeciesInteraction
from cellular_automata.Sim import CellularAutomaton, resolve_seed
from cellular_automata.State import State

logger = logging.getLogger(__name__)

STUDY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = STUDY_DIR / "configs" / "gut.json"
RESULTS_DIR = STUDY_DIR / "results"
BIOFILM_KEY = "biofilm"
FLOATING_KEY = "floating_bacteria"
ABUNDANCE_TOLERANCE = 1e-3
# spawn key of the initial-state generator; the simulation uses (0,) and (1, i)
INITIAL_STATE_STREAM = (2,)
# a mound of the smoothed mat is about this many Gaussian sigmas across
PATCH_SIGMAS = 4
PARSER_LOGGER = "cellular_automata.config.Parser"


@dataclass(frozen=True)
class InitialState:
    """The starting grid: `labels` holds 0 for empty and `i + 1` for `species[i]`."""

    species: tuple[str, ...]
    labels: np.ndarray
    biofilm: np.ndarray
    floating: np.ndarray


# --- config -------------------------------------------------------------------


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _rule_args(config: dict, rule_name: str) -> dict:
    matches = [rule for rule in config["rules"] if rule["name"] == rule_name]
    if len(matches) != 1:
        raise ValueError(f"the config needs exactly one {rule_name} rule, found {len(matches)}")
    return matches[0].setdefault("args", {})


def validate_config(config: dict) -> list[str]:
    """Check the blocks this script needs and return the species keys, in order.

    The species are the keys of `abundances.before`. `abundances.after`, the
    state keys, the `SpeciesInteraction` coefficients and the `species_keys` of
    `BiofilmDetachment` (and `GutDrift`, if present) must name the same ones.
    """
    abundances = config.get("abundances", {})
    before, after = abundances.get("before"), abundances.get("after")
    if not before or not after:
        raise ValueError("the config needs abundances.before and abundances.after")
    species = list(before)
    if set(after) != set(species):
        raise ValueError(
            f"abundances.before and .after name different species: {sorted(before)} vs "
            f"{sorted(after)}"
        )
    for name, values in (("before", before), ("after", after)):
        if any(value < 0 for value in values.values()):
            raise ValueError(f"abundances.{name} has a negative abundance")
        total = sum(values.values())
        if abs(total - 1.0) > ABUNDANCE_TOLERANCE:
            raise ValueError(f"abundances.{name} must sum to 1, got {total:.6f}")
    if len(species) > np.iinfo(np.uint8).max:
        raise ValueError(f"at most {np.iinfo(np.uint8).max} species fit the label grid")

    state_keys = {key["name"] for key in config["state_keys"]}
    missing = [key for key in [*species, BIOFILM_KEY, FLOATING_KEY] if key not in state_keys]
    if missing:
        raise ValueError(f"state_keys is missing {missing}")

    interaction_species = set(_rule_args(config, SpeciesInteraction.__name__)["interactions"])
    if interaction_species != set(species):
        raise ValueError(
            f"SpeciesInteraction interactions name {sorted(interaction_species)}, but the "
            f"abundances name {sorted(species)}"
        )
    species_key_rules = [BiofilmDetachment.__name__]
    if any(rule["name"] == GutDrift.__name__ for rule in config["rules"]):
        species_key_rules.append(GutDrift.__name__)
    for rule_name in species_key_rules:
        rule_species = set(_rule_args(config, rule_name).get("species_keys") or [])
        if rule_species != set(species):
            raise ValueError(
                f"{rule_name} species_keys name {sorted(rule_species)}, but the abundances "
                f"name {sorted(species)}"
            )

    initial = config.get("initial_state", {})
    for name in ("mean_thickness", "thickness_spread", "patch_size", "lumen_density"):
        if name not in initial:
            raise ValueError(f"the config needs initial_state.{name}")
    if initial["mean_thickness"] < 0 or initial["thickness_spread"] < 0:
        raise ValueError("initial_state.mean_thickness and .thickness_spread must be >= 0")
    if initial["patch_size"] <= 0:
        raise ValueError("initial_state.patch_size must be > 0")
    if not 0.0 <= initial["lumen_density"] <= 1.0:
        raise ValueError("initial_state.lumen_density must be in [0, 1]")
    return species


# --- initial state ---------------------------------------------------------------


def mat_heights(
    plane_shape: tuple[int, ...],
    periodic: list[bool],
    mean_thickness: float,
    thickness_spread: float,
    patch_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mat height (in cells, >= 0) over every cell of a wall.

    White noise blurred with a Gaussian of sigma `patch_size / 4`, so mounds and
    bare patches are about `patch_size` cells across (blurred noise is still
    correlated at 37% over 2 sigma). The field is rescaled to mean
    `mean_thickness` and standard deviation `thickness_spread`, then rounded
    and cut off at 0, so the realised mean is a little higher when the spread
    is large. Periodic axes of the wall wrap around; other edges reflect.
    """
    noise = rng.standard_normal(plane_shape)
    modes = ["wrap" if wraps else "reflect" for wraps in periodic]
    field = gaussian_filter(noise, sigma=patch_size / PATCH_SIGMAS, mode=modes)
    spread = field.std()
    if spread > 0:
        field = (field - field.mean()) / spread
    else:
        field = np.zeros_like(field)
    return np.clip(np.rint(mean_thickness + thickness_spread * field), 0, None).astype(int)


def build_mat(detachment: BiofilmDetachment, initial: dict, rng: np.random.Generator) -> np.ndarray:
    """Biofilm grown from every wall of `detachment`, each column filled from the wall up."""
    geometry = detachment.geometry
    size = geometry.size
    mat = np.zeros(size, dtype=bool)
    for axis, side in detachment.walls:
        plane_axes = [a for a in range(len(size)) if a != axis]
        heights = mat_heights(
            tuple(size[a] for a in plane_axes),
            [a in geometry.periodic_dims for a in plane_axes],
            float(initial["mean_thickness"]),
            float(initial["thickness_spread"]),
            float(initial["patch_size"]),
            rng,
        )
        index = np.arange(size[axis])
        depth = index if side < 0 else size[axis] - 1 - index  # cells from this wall
        depth_shape = [1] * len(size)
        depth_shape[axis] = -1
        mat |= depth.reshape(depth_shape) < np.expand_dims(heights, axis)
    return mat


def assign_species(
    occupied: np.ndarray, abundances: dict[str, float], rng: np.random.Generator
) -> np.ndarray:
    """Label the occupied cells with species in the given proportions.

    Each species gets `floor(n * abundance)` of the `n` occupied cells, and the
    cells lost to rounding go to the most abundant species. Labels follow the
    order of `abundances` (1 for the first species, 0 for empty).
    """
    n_occupied = int(np.count_nonzero(occupied))
    # the small epsilon stops 28.999... (float error for 29) from losing a cell
    counts = {sp: int(np.floor(n_occupied * share + 1e-9)) for sp, share in abundances.items()}
    counts[max(counts, key=counts.__getitem__)] += n_occupied - sum(counts.values())
    occupied_labels = np.repeat(
        np.arange(1, len(counts) + 1, dtype=np.uint8), list(counts.values())
    )
    rng.shuffle(occupied_labels)
    labels = np.zeros(occupied.shape, dtype=np.uint8)
    labels[occupied] = occupied_labels
    return labels


def build_initial_state(
    config: dict, species: list[str], sim: CellularAutomaton, rng: np.random.Generator
) -> InitialState:
    """Draw the mat, the lumen and the species for `sim`'s grid (config already validated)."""
    detachment = next(rule for rule in sim.rules if isinstance(rule, BiofilmDetachment))
    initial = config["initial_state"]
    mat = build_mat(detachment, initial, rng)
    floating = ~mat & (rng.random(mat.shape) < float(initial["lumen_density"]))
    labels = assign_species(mat | floating, config["abundances"]["before"], rng)
    return InitialState(tuple(species), labels, mat, floating)


def apply_initial_state(state: State, initial: InitialState) -> None:
    for label, sp in enumerate(initial.species, start=1):
        state[sp] = (initial.labels == label).astype(np.uint8)
    state[BIOFILM_KEY] = initial.biofilm.astype(np.uint8)
    state[FLOATING_KEY] = initial.floating.astype(np.uint8)


def initial_state_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=INITIAL_STATE_STREAM))


# --- run and report ---------------------------------------------------------------


def composition(state: State, species: list[str]) -> dict:
    """Relative abundances (shares of the occupied cells) and cell counts.

    `unflagged_cells` are occupied cells that are neither biofilm nor floating;
    `SpeciesInteraction` creates them when it spawns a cell.
    """
    counts = {sp: int(np.count_nonzero(state[sp])) for sp in species}
    occupied_mask = np.zeros(state.shape, dtype=bool)
    for sp in species:
        occupied_mask |= state[sp].astype(bool)
    flagged = state[BIOFILM_KEY].astype(bool) | state[FLOATING_KEY].astype(bool)
    occupied = sum(counts.values())
    return {
        "abundances": {sp: (n / occupied if occupied else 0.0) for sp, n in counts.items()},
        "occupied_cells": occupied,
        "biofilm_cells": int(np.count_nonzero(state[BIOFILM_KEY])),
        "floating_cells": int(np.count_nonzero(state[FLOATING_KEY])),
        "unflagged_cells": int(np.count_nonzero(occupied_mask & ~flagged)),
    }


def abundance_error(final: dict[str, float], target: dict[str, float]) -> float:
    """Sum of squared differences between final and target relative abundances."""
    return float(sum((target[sp] - final[sp]) ** 2 for sp in target))


def prepare(config: dict, seed: int | None = None, tracker_path: Path | None = None):
    """Build the simulation and fill in its starting grid.

    Returns (sim, config used, species). With `tracker_path` the history is
    recorded from the filled-in grid and saved there at the end of the run.
    """
    config = copy.deepcopy(config)
    species = validate_config(config)
    config["sim"]["seed"] = resolve_seed(seed if seed is not None else config["sim"].get("seed"))
    config["sim"]["render"] = False
    tracker_block = config.pop("tracker", None) or {"name": "StateTracker", "args": {}}
    # built without a tracker, which would record step 0 before the grid is filled in;
    # the parser would then log that no history is recorded, which isn't true here
    parser_logger = logging.getLogger(PARSER_LOGGER)
    level = parser_logger.level
    parser_logger.setLevel(logging.WARNING)
    try:
        sim = build_simulation(config)
    finally:
        parser_logger.setLevel(level)
    initial = build_initial_state(config, species, sim, initial_state_rng(config["sim"]["seed"]))
    apply_initial_state(sim.state, initial)
    if tracker_path is not None:
        tracker_args = dict(tracker_block.get("args", {}))
        tracker_args.setdefault("keys", list(sim.state.keys))
        sim.tracker = TRACKERS.get(tracker_block["name"])(**tracker_args)
        sim.tracker.seed = sim.seed
        sim.output_filename = str(tracker_path)
        sim.tracker.record_step(sim.state, 0)
        config["tracker"] = {**tracker_block, "save_as": str(tracker_path)}
    return sim, config, species


def run(config: dict, seed: int | None = None, tracker_path: Path | None = None) -> dict:
    """Run the simulation to `sim.max_steps` and return a summary of the run."""
    sim, config, species = prepare(config, seed, tracker_path)
    start = composition(sim.state, species)
    while sim.step_no < sim.max_steps:
        sim.step()
    sim.save_tracker()  # step() only saves when it reaches max_steps, never with max_steps 0
    final = composition(sim.state, species)
    return {
        "seed": config["sim"]["seed"],
        "steps": sim.step_no,
        "start": start,
        "final": final,
        "error_vs_after": abundance_error(final["abundances"], config["abundances"]["after"]),
        "error_vs_before": abundance_error(final["abundances"], config["abundances"]["before"]),
        "config": config,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one gut simulation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None, help="default: sim.seed in the config")
    parser.add_argument(
        "--name",
        default=None,
        help="base name for the files in research/gut/results/ "
        "(default: <config>_<seed>_<date-time>)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing results with the same name"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    config = load_config(args.config)
    seed = resolve_seed(args.seed if args.seed is not None else config["sim"].get("seed"))
    name = args.name or f"{args.config.stem}_{seed}_{datetime.now():%Y%m%d-%H%M%S}"
    tracker_path = RESULTS_DIR / f"{name}.npz"
    summary_path = RESULTS_DIR / f"{name}.json"
    existing = [path for path in (tracker_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"{', '.join(map(str, existing))} already exists; pick another --name or "
            "pass --overwrite"
        )
    summary = run(config, seed=seed, tracker_path=tracker_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    target = config["abundances"]["after"]
    logger.info("%-12s %8s %8s %8s", "species", "start", "final", "target")
    for sp in target:
        logger.info(
            "%-12s %8.4f %8.4f %8.4f",
            sp,
            summary["start"]["abundances"][sp],
            summary["final"]["abundances"][sp],
            target[sp],
        )
    for key in ("occupied_cells", "biofilm_cells", "floating_cells", "unflagged_cells"):
        logger.info("%s: %d -> %d", key, summary["start"][key], summary["final"][key])
    logger.info("error vs after: %.6f", summary["error_vs_after"])
    logger.info("history: %s, summary: %s", tracker_path, summary_path)


if __name__ == "__main__":
    main()
