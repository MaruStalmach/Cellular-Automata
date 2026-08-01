import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from libs.Geometry import Geometry
from libs.rules.GutDrift import GutDrift
from libs.rules.SpeciesInteraction import SpeciesInteraction
from libs.Sim import CellularAutomaton
from libs.State import State
from libs.tracking.StateTracker import StateTracker

ABUNDANCE_BEFORE = {
    "BBBBB": 0.725667,
    "BBBBM": 0.000111,
    "BBBBT": 0.0,
    "BFCLL": 0.162,
    "BFCOO": 0.024611,
    "BFCOR": 0.002056,
    "BFCTP": 0.001167,
    "other": 0.084389,
}

ABUNDANCE_AFTER = {
    "BBBBB": 0.757833,
    "BBBBM": 0.033444,
    "BBBBT": 0.047278,
    "BFCLL": 0.029722,
    "BFCOO": 0.029722,
    "BFCOR": 0.033333,
    "BFCTP": 0.0,
    "other": 0.068667,
}

SPECIES = list(ABUNDANCE_BEFORE.keys())
ADDITIONAL_KEYS = ["biofilm", "floating_bacteria"]
ALL_KEYS = SPECIES + ADDITIONAL_KEYS

GRID_SIZE = (100, 100, 50)   
DENSITY = 0.7
MAX_STEPS = 50
N_ITERATIONS = 1000

CHECKPOINT_PATH = Path("best_coeffs_checkpoint.json")
CHECKPOINT_EVERY_S = 60  


def build_species_grid(geometry: Geometry, abundances: dict, density: float) -> np.ndarray:
    total_cells = geometry.num_cells
    total_occupied = int(total_cells * density)

    species_counts = {
        sp: int(total_occupied * proportion)
        for sp, proportion in abundances.items()
    }
    remainder = total_occupied - sum(species_counts.values())
    if remainder != 0:
        largest_species = max(species_counts, key=species_counts.get)
        species_counts[largest_species] += remainder

    flat_grid = np.zeros(total_cells, dtype=np.uint8)
    cursor = 0
    for idx, (sp, count) in enumerate(species_counts.items(), start=1):
        flat_grid[cursor: cursor + count] = idx
        cursor += count

    np.random.shuffle(flat_grid) 
    return flat_grid.reshape(geometry.size)


def state_from_cached_grid(geometry: Geometry, grid_3d: np.ndarray, species_keys: list, additional_keys: list) -> State:
    state = State(geometry, random=False, cell_keys=species_keys + additional_keys)

    for idx, sp in enumerate(species_keys, start=1):
        state[sp] = (grid_3d == idx).astype(np.uint8)

    for key in additional_keys:
        state[key] = np.zeros(geometry.size, dtype=np.uint8)

    return state

def generate_random_coeffs(species_keys: list) -> dict:
    interactions = {}
    for sp_a in species_keys:
        interactions[sp_a] = {}
        for sp_b in species_keys:
            if sp_a != sp_b:
                interactions[sp_a][sp_b] = float(np.random.uniform(-1.0, 1.0))
    return interactions


#### MONTE CARLO 

def evaluate_guess(geometry: Geometry, cached_grid: np.ndarray,
                    interactions_guess: dict, max_steps: int,
                    verbose_timing: bool = False) -> float:
    t_state0 = time.time()
    state = state_from_cached_grid(geometry, cached_grid, SPECIES, ADDITIONAL_KEYS)
    t_state = time.time() - t_state0

    interaction_rule = SpeciesInteraction(
        geometry=geometry,
        time_step=1.0,
        interactions=interactions_guess,
    )

    state_tracker = StateTracker(ALL_KEYS)

    gut_drift_rule = GutDrift(
        geometry=geometry, 
        time_step=1.0,
        drift_speed=1
    )

    sim = CellularAutomaton(
        geometry=geometry,
        rules=[interaction_rule, gut_drift_rule],
        state=state,
        max_steps=max_steps,
        tracker=state_tracker,
        output_filename="mc_search_run",
    )

    t_sim0 = time.time()
    while sim.step_no < sim.max_steps:
        sim.step()
    t_sim = time.time() - t_sim0

    if verbose_timing:
        print(f"{max_steps} sim steps: {t_sim:.2f}s "
              f"({t_sim / max_steps:.3f}s/step)")

    total_cells = geometry.num_cells
    final_proportions = {
        sp: np.sum(sim.state[sp]) / total_cells for sp in SPECIES
    }
    error = sum(
        (ABUNDANCE_AFTER[sp] - final_proportions[sp]) ** 2 for sp in SPECIES
    )
    return error


def main():
    total_start_time = time.time() #to time search
    
    geometry = Geometry(size=GRID_SIZE, axes="xyz", periodicity="")

    cached_grid = build_species_grid(geometry, ABUNDANCE_BEFORE, DENSITY)

    best_error = float("inf")
    best_coeffs = None
    last_checkpoint = time.time()

    print("\nstarting iterations...")
    
    guesses = [generate_random_coeffs(SPECIES) for _ in range(N_ITERATIONS)]
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(evaluate_guess, geometry, cached_grid, guess, MAX_STEPS, False): guess 
            for guess in guesses
        }

        for i, future in enumerate(as_completed(futures)):
            guess = futures[future]
            error = future.result()
            
            print(f"[{i + 1}/{N_ITERATIONS}] error={error:.6f}")

            if error < best_error:
                best_error = error
                best_coeffs = guess
                print(f"NEW BEST (error={best_error:.6f})")

            if best_coeffs is not None and time.time() - last_checkpoint > CHECKPOINT_EVERY_S:
                CHECKPOINT_PATH.write_text(json.dumps(
                    {"iteration": i + 1, "best_error": best_error, "best_coeffs": best_coeffs},
                    indent=2,
                ))
                last_checkpoint = time.time()

    print("\noptimization finished - best parameters found:")
    print(json.dumps(best_coeffs, indent=4))
    
    total_elapsed = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\ntotal execution time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")


if __name__ == "__main__":
    main()