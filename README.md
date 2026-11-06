# Experiment results

This is an **orphan branch**: it shares no history with `main` and holds only experiment outputs, never code. It keeps results out of `main` until we decide which (if any) belong there.

Every file is a byte-for-byte copy of a file that was committed on a code branch. The tables below say where each came from, so you can check out the code that produced it.

> [!WARNING]
> **All results here predate the correctness fixes in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37).** Treat them as historical, not as results of the current model.
> - **BUG-1:** neighbour counts overflowed `int8`. Dense multi-layer substrate (bioreactor) gave negative counts, so growth and uptake stopped in the richest regions. This affects every bioreactor run. The gut runs use single-layer species (at most 26 neighbours), so this bug doesn't apply to them.
> - **BUG-6:** biofilm detachment (bioreactor version) cleared the whole x=0 slab every step, even at a detachment rate of 0. This affects every bioreactor run that used the rule.
> - **REPRO-1:** no run was seeded, so none of these can be reproduced exactly.
>
> See `docs/code-quality-plan.md` on the code branches for details.

## Layout

```text
bioreactor/
  screening/              pairwise interaction screening (CSV)
  bayesian-optimisation/  BO trials (CSV)
  simulations/            StateTracker outputs (.npz)
gut/
  mc-search/              Monte Carlo search for interaction coefficients (JSON)
  simulations/            StateTracker output (.npz)
```

## Bioreactor

Source branch: `bioreactor`. Code at the time lived in `libs/`, configs in `libs/config/`. The configs now live in `research/bioreactor/configs/`, and the scripts are in `research/bioreactor/scripts/`.

### Screening and Bayesian optimisation

Original path: `optim_results/`. Produced by `screening.py` / `bayesian_optimization.py`.

| File | Commit | Date | Author |
|---|---|---|---|
| `screening/screening_results_0-2.csv`, `_3-5.csv`, `_6-11.csv` | `d241f64` "screen results" | 2026-09-03 | jakwojc |
| `screening/screening_result.csv`, `screening/screening_pair_analysis.csv` | `2dd01e7` "updates" | 2026-09-07 | jakwojc |
| `bayesian-optimisation/bo_results.csv`, `bo_results2.csv`, `bo_results_old.csv` | `2dd01e7` "updates" | 2026-09-07 | jakwojc |

`bayesian_optimization.py` resumes from `results/bo_results.csv`. Don't copy these back to resume a search: the scores come from the code before the fixes.

### Simulations

Original path: `output_files/tracked_simulations/`. The config is the one whose `save_as` matches the file name at that commit. Where `save_as` was edited by hand between runs (`-r2`, `-r3`, …), the config is the one that names the first run of the series.

| File | Commit | Date | Config (at that commit) |
|---|---|---|---|
| `full.npz`, `full2.npz`, `full3.npz` | `b7320a7` "added new visualization notebook and some data" | 2026-08-04 | `full.json`, `full2.json`, `full3.json` (probable: only `full.json` existed at that commit) |
| `full18.npz`, `full18-bo.npz`, `full18-bo-v2.npz` | `2dd01e7` "updates" | 2026-09-07 | `full18-bo-v2`: `screening_runs/bo_9.json` (`save_as: full18-bo-v2`). `full18`, `full18-bo`: probably `screening_runs/full18_8.json` (`save_as: full18-bo1`), no exact match |
| `bo2-r1.npz`, `bo2-r3.npz`, `bo2-r5.npz` | `2ce8f12` "results" | 2026-09-09 | `screening_runs/bo_9.json` (`save_as: bo2-r1`) |
| `bo2-antib-r2.npz`, `bo2-antib-r3.npz` | `2ce8f12` "results" | 2026-09-09 | `calibrated_antib.json` (`save_as: bo2-antib-r1`) |
| `bo2-antib-r5.npz` | `9fb3c8e` "res" | 2026-09-09 | `calibrated_antib.json` |
| `precal-r1.npz` … `precal-r5.npz` | `ce8bc73` "precal results" | 2026-09-10 | `screen2.json` (`save_as: precal-r1`) |

Missing runs in a series (for example `bo2-r2`, `bo2-antib-r1`) were never committed.

## Gut

### MC search (`gut/mc-search/output.json`)

Best interaction coefficients from the parallel Monte Carlo search.

- Commit `1250f5f` "first iteration of tests", 2026-08-01, Maru, branch `gut-experiments`, original path `experiments/output.json`.
- Produced by `experiments/data_preparation.py` **as it was at `1250f5f`**: grid 100×100×50, 50 steps, 1000 iterations, before/after abundances hard-coded in the script. The later `experiments/mc_config.json` (`ddae2e8`, 250×250×100, 20 steps) did **not** produce this file.
- Score: sum of squared errors of the final species proportions.
- Under `fork`, the parallel workers shared the same RNG state, so the simulation noise was correlated across workers (REPRO-1).

### Simulation (`gut/simulations/mc_search_run.npz`)

> [!CAUTION]
> **This file is corrupted, and git has no other version of it.** 7 of its 9 arrays fail to decompress (`BBBBB` … `BFCTP`, `other`). Only `biofilm` and `floating_bacteria` can be read.

- Commit `03d5a7a` "update package name", 2026-08-02, maru, original path `output_files/tracked_simulations/mc_search_run.npz`.
- No config in the repo produces it. Each array holds 51 frames of 100×100×50 cells, which matches the `1250f5f` MC search settings (50 steps + the initial state). It was probably a single tracked run with one coefficient guess.
- If the original is still on someone's disk, replace it here.

## Not included

- Demo outputs `multispecies_experiment.npz` and `simulation_visualisation.npz`: example and visualisation tests, not experiments.
- Gut `ratios.csv` files and plots read by `check_occupancy.py` / `compare_to_reference.py`: never committed. The script that produced them (`experiment.py`) isn't in the repo either.
- Input data (OTU/ASV tables, metadata): these are inputs, not results, and stay in `research/bioreactor/data/` on the code branches.

## Adding results

1. Put them under `<study>/<experiment>/`.
2. Add a row to this README with the code commit, the config, the seed and the date.
3. Commit on this branch only. Never merge `main` into this branch or this branch into `main`.
