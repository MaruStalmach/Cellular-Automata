import argparse
import inspect
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.Geometry import Geometry
from libs.rules.BiofilmDetachment import BiofilmDetachment
from libs.rules.GutDrift import GutDrift
from libs.rules.SpeciesInteraction import SpeciesInteraction
from libs.Sim import CellularAutomaton
from libs.State import State

RULE_REGISTRY = {
    "species_interaction": SpeciesInteraction,
    "biofilm_detachment": BiofilmDetachment,
    "gut_drift": GutDrift,
}


def get_rule_params(rule_specs: list[dict], name: str) -> dict:
    for spec in rule_specs:
        if spec.get("name") == name:
            return spec.get("params", {})
    return {}


def build_rules(
    rule_specs: list[dict],
    geometry: Geometry,
    default_time_step: float,
    species_keys: list[str],
    interactions_guess: dict,
) -> list:
    rules = []
    for spec in rule_specs:
        name = spec.get("name")
        if not name or name not in RULE_REGISTRY:
            raise ValueError(
                f"Unknown or missing rule '{name}' in config. Known rules: {list(RULE_REGISTRY)}"
            )
        rule_cls = RULE_REGISTRY[name]

        kwargs = dict(spec.get("params", {}))
        time_step = spec.get("time_step", default_time_step)

        runtime_context = {
            "geometry": geometry,
            "time_step": time_step,
            "species_keys": species_keys,
            "interactions": interactions_guess,
        }

        accepted_params = inspect.signature(rule_cls.__init__).parameters
        for key, value in runtime_context.items():
            if key in accepted_params and key not in kwargs:
                kwargs[key] = value

        rules.append(rule_cls(**kwargs))

    return rules


def build_species_grid(geometry: Geometry, abundances: dict, density: float) -> np.ndarray:
    total_cells = geometry.num_cells
    total_occupied = int(total_cells * density)

    species_counts = {
        sp: int(total_occupied * proportion)
        for sp, proportion in abundances.items()
    }
    remainder = total_occupied - sum(species_counts.values())
    
    if remainder != 0 and species_counts:
        largest_species = max(species_counts, key=species_counts.get)
        species_counts[largest_species] += remainder

    flat_grid = np.zeros(total_cells, dtype=np.uint8)
    cursor = 0
    for idx, (sp, count) in enumerate(species_counts.items(), start=1):
        if count > 0:
            flat_grid[cursor : cursor + count] = idx
            cursor += count

    np.random.shuffle(flat_grid)
    return flat_grid.reshape(geometry.size)


def seed_biofilm(
    geometry: Geometry,
    species_occupied: np.ndarray,
    detachment_rate: float,
    scaling: float,
) -> np.ndarray:
    if geometry.ndim != 3:
        raise ValueError("Geometry must be 3-dimensional to seed biofilm.")
        
    z_axis = geometry.ndim - 1
    total_layers = geometry.size[z_axis]

    detachment_probability = detachment_rate * scaling
    z_indices = np.arange(total_layers)
    chance_detachment_z = np.clip(detachment_probability * (z_indices ** 2), 0.0, 1.0)
    p_attach_z = 1.0 - chance_detachment_z

    broadcast_shape = [1] * geometry.ndim
    broadcast_shape[z_axis] = total_layers
    p_attach_grid = p_attach_z.reshape(broadcast_shape)

    random_draw = np.random.random(geometry.size)
    biofilm = species_occupied & (random_draw < p_attach_grid)
    return biofilm.astype(np.uint8)


def state_from_cached_grid(
    geometry: Geometry,
    grid_3d: np.ndarray,
    species_keys: list[str],
    additional_keys: list[str],
    detachment_rate: float,
    scaling: float,
) -> State:
    state = State(geometry, random=False, cell_keys=species_keys + additional_keys)

    for idx, sp in enumerate(species_keys, start=1):
        state[sp] = (grid_3d == idx).astype(np.uint8)

    species_occupied = np.logical_or.reduce(
        [state[sp].astype(bool) for sp in species_keys]
    ) if species_keys else np.zeros(geometry.size, dtype=bool)

    for key in additional_keys:
        if key == "biofilm":
            state[key] = seed_biofilm(geometry, species_occupied, detachment_rate, scaling)
        else:
            state[key] = np.zeros(geometry.size, dtype=np.uint8)

    return state


def generate_random_coeffs(species_keys: list[str], bounds: tuple[float, float] = (-1.0, 1.0)) -> dict:
    low, high = bounds
    return {
        sp_a: {sp_b: float(np.random.uniform(low, high)) for sp_b in species_keys if sp_a != sp_b}
        for sp_a in species_keys
    }


def evaluate_guess(
    geometry: Geometry,
    cached_grid: np.ndarray,
    interactions_guess: dict,
    max_steps: int,
    species: list[str],
    additional_keys: list[str],
    abundance_after: dict,
    rule_specs: list[dict],
    default_time_step: float = 1.0,
    verbose_timing: bool = False,
) -> float:
    biofilm_params = get_rule_params(rule_specs, "biofilm_detachment")
    detachment_rate = biofilm_params.get("detachment_rate", 0.1)
    scaling = biofilm_params.get("scaling", 0.0005)

    t_state0 = time.time()
    state = state_from_cached_grid(
        geometry, cached_grid, species, additional_keys, detachment_rate, scaling
    )
    t_state = time.time() - t_state0

    rules = build_rules(
        rule_specs, geometry, default_time_step, species, interactions_guess
    )

    all_keys = species + additional_keys
    # state_tracker = StateTracker(all_keys)

    sim = CellularAutomaton(
        geometry=geometry,
        rules=rules,
        state=state,
        max_steps=max_steps,
        tracker=None,
        output_filename=None,
    )

    t_sim0 = time.time()
    while sim.step_no < sim.max_steps:
        sim.step()
    t_sim = time.time() - t_sim0

    if verbose_timing:
        print(f"{max_steps} sim steps: {t_sim:.2f}s ({t_sim / max_steps:.3f}s/step)")

    total_cells = geometry.num_cells
    if total_cells == 0:
        return float('inf')

    final_proportions = {sp: np.sum(sim.state[sp]) / total_cells for sp in species}
    error = sum((abundance_after[sp] - final_proportions[sp]) ** 2 for sp in species)
    
    return float(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="mc_config.json")
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

    grid_size = tuple(config.get("sim", {}).get("size", (100, 100, 50)))
    max_steps = config.get("sim", {}).get("max_steps", 100)
    density = config.get("sim", {}).get("density", 0.7)
    default_time_step = config.get("sim", {}).get("default_time_step", 1.0)

    rule_specs = config.get("rules", [])

    mc_cfg = config.get("mc_search", {})
    n_iterations = mc_cfg.get("n_iterations", 5)
    checkpoint_every_s = mc_cfg.get("checkpoint_every_s", 60)
    checkpoint_path = Path(mc_cfg.get("checkpoint_path", "best_coeffs_checkpoint.json"))
    interaction_bounds = tuple(mc_cfg.get("interaction_bounds", [-1.0, 1.0]))

    abundance_before = config.get("abundances", {}).get("before", {})
    abundance_after = config.get("abundances", {}).get("after", {})
    species = list(abundance_before.keys())
    additional_keys = config.get("additional_keys", ["biofilm", "floating_bacteria"])

    total_start_time = time.time()
    geometry = Geometry(size=grid_size, axes="xyz", periodicity="")
    cached_grid = build_species_grid(geometry, abundance_before, density)

    print("Active rules, in order:")
    for spec in rule_specs:
        print(f"  - {spec.get('name')}: {spec.get('params', {})} (dt: {spec.get('time_step', default_time_step)})")

    best_error = float("inf")
    best_coeffs = None
    last_checkpoint = time.time()

    print("\nstarting iterations...")

    guesses = [generate_random_coeffs(species, interaction_bounds) for _ in range(n_iterations)]
    
    try:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    evaluate_guess,
                    geometry,
                    cached_grid,
                    guess,
                    max_steps,
                    species,
                    additional_keys,
                    abundance_after,
                    rule_specs,
                    default_time_step,
                    False,
                ): guess
                for guess in guesses
            }

            for i, future in enumerate(as_completed(futures)):
                guess = futures[future]
                try:
                    error = future.result()
                    print(f"[{i + 1}/{n_iterations}] error={error:.6f}")

                    if error < best_error:
                        best_error = error
                        best_coeffs = guess
                        print(f"NEW BEST (error={best_error:.6f})")

                    if best_coeffs is not None and time.time() - last_checkpoint > checkpoint_every_s:
                        checkpoint_path.write_text(
                            json.dumps(
                                {"iteration": i + 1, "best_error": best_error, "best_coeffs": best_coeffs},
                                indent=2,
                            ),
                            encoding="utf-8"
                        )
                        last_checkpoint = time.time()
                        
                except Exception as e:
                    print(f"[{i + 1}/{n_iterations}] Task failed with exception: {e}")
                    
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    print("\noptimization finished - best parameters found:")
    if best_coeffs:
        print(json.dumps(best_coeffs, indent=4))
    else:
        print("No parameters found.")

    total_elapsed = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\ntotal execution time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")


if __name__ == "__main__":
    main()