"""Monte Carlo search for gut species-interaction coefficients.

Every guess draws a random `SpeciesInteraction` matrix, runs the simulation in
`--config` from one shared initial state, and is scored by how far the final
relative abundances are from `abundances.after` in the config. The best guess
and every scored guess are written to `--output`, which also serves as the
checkpoint while the search runs.

The config is an ordinary simulation config (it builds with `parse_json`) plus
two blocks the parser ignores:

- `abundances` - `before` (initial relative abundances, one entry per species
  key) and `after` (the target)
- `mc_search` - `n_iterations`, `density` (fraction of cells occupied at the
  start), `interaction_bounds` ([low, high] for every coefficient) and
  `checkpoint_every_s`

The `SpeciesInteraction` coefficients in the config are placeholders; each
guess replaces them. Run from the repository root:

    python research/gut/scripts/mc_search.py --iterations 20 --workers 4
"""

import argparse
import copy
import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellular_automata.config.Parser import build_simulation
from cellular_automata.rules import BiofilmDetachment, GutDrift, SpeciesInteraction
from cellular_automata.Sim import CellularAutomaton, resolve_seed
from cellular_automata.State import State

logger = logging.getLogger(__name__)

STUDY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = STUDY_DIR / "configs" / "mc_search.json"
DEFAULT_OUTPUT = STUDY_DIR / "results" / "mc_search_results.json"
BIOFILM_KEY = "biofilm"
FLOATING_KEY = "floating_bacteria"
ABUNDANCE_TOLERANCE = 1e-3

Interactions = dict[str, dict[str, float]]


@dataclass(frozen=True)
class InitialState:
    """The starting grid shared by every guess.

    `labels` holds 0 for an empty cell and `i + 1` for `species[i]`; `biofilm`
    marks the occupied cells that start attached.
    """

    species: tuple[str, ...]
    labels: np.ndarray
    biofilm: np.ndarray


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
    """Check the blocks the search needs and return the species keys, in order.

    The species are the keys of `abundances.before`; `abundances.after`, the
    `SpeciesInteraction` coefficients, the state keys and the `species_keys` of
    `BiofilmDetachment` (and `GutDrift`, if the config has it) must name the same ones.
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

    search = config.get("mc_search", {})
    if int(search.get("n_iterations", 0)) < 1:
        raise ValueError("mc_search.n_iterations must be at least 1")
    if not 0.0 < float(search.get("density", 0.0)) <= 1.0:
        raise ValueError("mc_search.density must be in (0, 1]")
    low, high = search.get("interaction_bounds", [-1.0, 1.0])
    if not low < high:
        raise ValueError(f"mc_search.interaction_bounds must be [low, high], got {[low, high]}")
    if len(species) > np.iinfo(np.uint8).max:
        raise ValueError(f"at most {np.iinfo(np.uint8).max} species fit the label grid")
    return species


def config_for_guess(config: dict, interactions: Interactions, seed: int) -> dict:
    """A copy of `config` with these coefficients and seed, headless and untracked."""
    guess_config = copy.deepcopy(config)
    _rule_args(guess_config, SpeciesInteraction.__name__)["interactions"] = interactions
    guess_config["sim"]["seed"] = seed
    guess_config["sim"]["render"] = False
    guess_config.pop("tracker", None)
    return guess_config


# --- initial state --------------------------------------------------------------


def build_species_grid(
    size: tuple[int, ...],
    abundances: dict[str, float],
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Label grid with `density` of the cells occupied in the given proportions.

    Each species gets `floor(occupied * abundance)` cells, and the cells lost to
    rounding go to the most abundant species. Labels follow the order of
    `abundances` (1 for the first species, 0 for empty).
    """
    total_cells = int(np.prod(size))
    occupied = int(total_cells * density)
    counts = {sp: int(occupied * share) for sp, share in abundances.items()}
    counts[max(counts, key=counts.__getitem__)] += occupied - sum(counts.values())

    labels = np.zeros(total_cells, dtype=np.uint8)
    start = 0
    for label, count in enumerate(counts.values(), start=1):
        labels[start : start + count] = label
        start += count
    rng.shuffle(labels)
    return labels.reshape(size)


def seed_biofilm(
    occupied: np.ndarray, detachment: BiofilmDetachment, rng: np.random.Generator
) -> np.ndarray:
    """Attach each occupied cell with 1 - the rule's detachment probability at its height.

    Uses the rule's own axis and per-layer probabilities, so the starting
    biofilm follows the same wall convention as the detachment (D2 in the plan).
    """
    shape = [1] * occupied.ndim
    shape[detachment.y_axis] = -1
    p_attach = (1.0 - detachment.y_probabilities).reshape(shape)
    return (occupied & (rng.random(occupied.shape) < p_attach)).astype(np.uint8)


def build_initial_state(config: dict, species: list[str], rng: np.random.Generator) -> InitialState:
    """Draw the starting species grid and biofilm for the search."""
    # built only for its geometry and configured BiofilmDetachment; it never steps,
    # so the placeholder coefficients and seed don't matter
    placeholder = _rule_args(config, SpeciesInteraction.__name__)["interactions"]
    sim = build_simulation(config_for_guess(config, placeholder, seed=0))
    detachment = next(rule for rule in sim.rules if isinstance(rule, BiofilmDetachment))
    labels = build_species_grid(
        sim.geometry.size,
        config["abundances"]["before"],
        float(config["mc_search"]["density"]),
        rng,
    )
    biofilm = seed_biofilm(labels > 0, detachment, rng)
    return InitialState(species=tuple(species), labels=labels, biofilm=biofilm)


def initial_state_for(config: dict) -> InitialState:
    """The initial state a search with this config (and its `sim.seed`) starts from."""
    species = validate_config(config)
    init_rng, _ = _split_seed(int(config["sim"]["seed"]))
    return build_initial_state(config, species, init_rng)


def replay_guess(results: dict, index: int) -> CellularAutomaton:
    """Rerun guess `index` of a saved search (a results file loaded with `json.load`)."""
    row = next(row for row in results["guesses"] if row["index"] == index)
    config = results["config"]
    return run_guess(config, initial_state_for(config), row["interactions"], row["seed"])


def apply_initial_state(state: State, initial: InitialState) -> None:
    for label, sp in enumerate(initial.species, start=1):
        state[sp] = (initial.labels == label).astype(np.uint8)
    state[BIOFILM_KEY] = initial.biofilm
    state[FLOATING_KEY] = np.zeros_like(initial.biofilm)


# --- one guess --------------------------------------------------------------------


def random_interactions(
    species: list[str], bounds: tuple[float, float], rng: np.random.Generator
) -> Interactions:
    low, high = bounds
    return {a: {b: float(rng.uniform(low, high)) for b in species if b != a} for a in species}


def relative_abundances(state: State, species: list[str]) -> dict[str, float]:
    """Each species' share of the occupied cells (all 0 if nothing is left)."""
    counts = {sp: int(np.count_nonzero(state[sp])) for sp in species}
    total = sum(counts.values())
    if total == 0:
        return {sp: 0.0 for sp in species}
    return {sp: count / total for sp, count in counts.items()}


def abundance_error(final: dict[str, float], target: dict[str, float]) -> float:
    """Sum of squared differences between final and target relative abundances."""
    return float(sum((target[sp] - final[sp]) ** 2 for sp in target))


def run_guess(
    config: dict, initial: InitialState, interactions: Interactions, seed: int
) -> CellularAutomaton:
    sim = build_simulation(config_for_guess(config, interactions, seed))
    apply_initial_state(sim.state, initial)
    while sim.step_no < sim.max_steps:
        sim.step()
    return sim


def evaluate_guess(
    config: dict, initial: InitialState, interactions: Interactions, seed: int
) -> dict:
    """Run one guess; return its error, final relative abundances and occupied cells.

    `occupied_cells` tells an extinct community (0, every share 0) apart from a
    skewed one with a similar error.
    """
    sim = run_guess(config, initial, interactions, seed)
    species = list(initial.species)
    final = relative_abundances(sim.state, species)
    return {
        "error": abundance_error(final, config["abundances"]["after"]),
        "final_abundances": final,
        "occupied_cells": int(sum(np.count_nonzero(sim.state[sp]) for sp in species)),
    }


# Each worker process gets the config and initial state once, not once per guess.
_worker_inputs: tuple[dict, InitialState] | None = None


def _init_worker(config: dict, initial: InitialState) -> None:
    global _worker_inputs
    _worker_inputs = (config, initial)
    _quiet_parser()


def _quiet_parser() -> None:
    # every guess builds a simulation without a tracker; don't log that each time
    logging.getLogger("cellular_automata.config.Parser").setLevel(logging.WARNING)


def _evaluate_in_worker(interactions: Interactions, seed: int) -> dict:
    if _worker_inputs is None:
        raise RuntimeError("worker was started without _init_worker")
    config, initial = _worker_inputs
    return evaluate_guess(config, initial, interactions, seed)


# --- search --------------------------------------------------------------------------


def _split_seed(
    seed: int,
) -> tuple[np.random.Generator, tuple[np.random.SeedSequence, np.random.SeedSequence]]:
    """Initial-state generator, and the seed sequences for coefficients and simulations."""
    init_seq, coeff_seq, sims_seq = np.random.SeedSequence(seed).spawn(3)
    return np.random.default_rng(init_seq), (coeff_seq, sims_seq)


def draw_guesses(
    species: list[str], seed: int, n_iterations: int, bounds: tuple[float, float]
) -> tuple[np.random.Generator, list[tuple[Interactions, int]]]:
    """Split `seed` into the initial-state generator and one (coefficients, sim seed) per guess.

    `SeedSequence.spawn` gives every stream its own seed, so guesses don't share
    random numbers whichever worker runs them, and the same seed repeats the
    whole search.
    """
    init_rng, (coeff_seq, sims_seq) = _split_seed(seed)
    coeff_rng = np.random.default_rng(coeff_seq)
    sim_seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0] >> 1)
        for child in sims_seq.spawn(n_iterations)
    ]
    guesses = [(random_interactions(species, bounds, coeff_rng), s) for s in sim_seeds]
    return init_rng, guesses


def _write_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(results, indent=2), encoding="utf-8")
    partial.replace(path)


def _results(config: dict, seed: int, scored: list[dict], n_iterations: int, complete: bool):
    ordered = sorted(scored, key=lambda row: row["index"])
    return {
        "seed": seed,
        "n_iterations": n_iterations,
        "n_scored": len(ordered),
        "complete": complete,
        "best": min(ordered, key=lambda row: row["error"]) if ordered else None,
        "guesses": ordered,
        "config": config,
    }


def run_search(
    config: dict,
    output: Path,
    n_iterations: int | None = None,
    workers: int | None = None,
    seed: int | None = None,
) -> dict:
    """Score `n_iterations` random guesses in parallel and write them to `output`.

    `output` is rewritten every `mc_search.checkpoint_every_s` seconds and at the
    end, also when the search is interrupted or a guess fails; `complete` says
    whether every guess was scored. Returns what was written.
    """
    config = copy.deepcopy(config)
    search = config.setdefault("mc_search", {})
    if n_iterations is not None:
        search["n_iterations"] = n_iterations
    species = validate_config(config)
    n_iterations = int(search["n_iterations"])
    seed = resolve_seed(seed if seed is not None else config["sim"].get("seed"))
    config["sim"]["seed"] = seed
    bounds = tuple(search.get("interaction_bounds", [-1.0, 1.0]))
    checkpoint_every_s = float(search.get("checkpoint_every_s", 60))

    init_rng, guesses = draw_guesses(species, seed, n_iterations, bounds)
    initial = build_initial_state(config, species, init_rng)
    logger.info(
        "MC search: %d guesses, seed %d, %d species, grid %s",
        n_iterations,
        seed,
        len(species),
        initial.labels.shape,
    )

    scored: list[dict] = []
    complete = False
    last_checkpoint = time.monotonic()
    try:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(config, initial)
        ) as executor:
            pending: dict[Future, int] = {
                executor.submit(_evaluate_in_worker, interactions, sim_seed): index
                for index, (interactions, sim_seed) in enumerate(guesses)
            }
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        interactions, sim_seed = guesses[index]
                        try:
                            outcome = future.result()
                        except Exception:
                            logger.error("guess %d (sim seed %d) failed", index, sim_seed)
                            raise
                        row = {
                            "index": index,
                            "seed": sim_seed,
                            **outcome,
                            "interactions": interactions,
                        }
                        scored.append(row)
                        best = min(r["error"] for r in scored)
                        logger.info(
                            "[%d/%d] guess %d: error=%.6f (best %.6f)",
                            len(scored),
                            n_iterations,
                            index,
                            row["error"],
                            best,
                        )
                    if time.monotonic() - last_checkpoint > checkpoint_every_s:
                        _write_results(output, _results(config, seed, scored, n_iterations, False))
                        last_checkpoint = time.monotonic()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        complete = True
    except KeyboardInterrupt:
        logger.warning("interrupted after %d of %d guesses", len(scored), n_iterations)
    finally:
        results = _results(config, seed, scored, n_iterations, complete)
        _write_results(output, results)
        logger.info("results written to %s", output)
    return results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo search for gut species-interaction coefficients."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--iterations", type=int, default=None, help="default: mc_search.n_iterations"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="worker processes (default: CPU count)"
    )
    parser.add_argument("--seed", type=int, default=None, help="default: sim.seed in the config")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _quiet_parser()
    args = parse_args(argv)
    results = run_search(
        load_config(args.config),
        args.output,
        n_iterations=args.iterations,
        workers=args.workers or os.cpu_count(),
        seed=args.seed,
    )
    if results["best"] is not None:
        logger.info(
            "best guess %d: error=%.6f\n%s",
            results["best"]["index"],
            results["best"]["error"],
            json.dumps(results["best"]["interactions"], indent=2),
        )


if __name__ == "__main__":
    main()
