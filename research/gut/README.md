# Gut microbiome study

Simulating a gut community with biofilm on the gut wall, and comparing its final composition with a measured one.

| Folder | Contents |
|---|---|
| `configs/` | `gut.json`: the simulation, the starting grid and the measured compositions. |
| `data/` | Input data; see [`data/README.md`](data/README.md). |
| `scripts/` | `run_gut.py`: runs one simulation and compares it with the data. |
| `results/` | Created by the script; git-ignored. |

## Running a simulation

Run from the repository root. Edit `configs/gut.json` between runs, or pass another config:

```bash
python research/gut/scripts/run_gut.py
python research/gut/scripts/run_gut.py --seed 3
python research/gut/scripts/run_gut.py --config <config.json> --name <run name>
```

Each run writes two files to `results/`, named `<config>_<seed>_<date-time>` unless you pass `--name`. An existing name is never overwritten unless you add `--overwrite`.

- `<name>.npz` - the tracked history (`StateTracker`), starting from the filled-in grid; replay it with `examples/Replay_sim.py`. `gut.json` records every 5th step (`tracker.args.record_every`), plus the last: at 250×250×100 each recorded step takes about 62 MB in memory while the run lasts.
- `<name>.json` - the seed, the config used, and the starting and final composition (each species' share of the occupied cells, plus occupied, biofilm, floating and unflagged cell counts), with the error against `abundances.after` and against `abundances.before`.

`unflagged_cells` are occupied cells that are neither biofilm nor floating. `SpeciesInteraction` creates them when it spawns a cell (it doesn't set either flag), so with non-zero coefficients this count grows. `GutDrift` treats such cells as attached.

The error is the sum of squared differences between the final shares and the target shares; lower is better. The script also prints a table of start, final and target shares.

The same config and seed always give the same run.

## The config

`gut.json` is an ordinary simulation config (`sim`, `state_keys`, `rules`, `tracker`), so it also builds with the parser and `examples/CA_sim.py`. On its own it starts from an empty grid, though: only `run_gut.py` fills in the starting grid. It has two extra blocks:

- `abundances` - `before` (the starting composition, one entry per species state key) and `after` (the target). Each must sum to 1.
- `initial_state` - how the starting grid is drawn. One cell is one grid spot, and every length is in cells:
  - `mean_thickness` and `thickness_spread` - the average height of the biofilm mat and how much it varies between patches. Heights are rounded and cut off at 0, so with a large spread the realised mean is slightly higher.
  - `patch_size` - roughly how wide a mound or bare patch of the mat is, along the wall (the smoothing uses a Gaussian with sigma `patch_size / 4`)
  - `lumen_density` - the share of the cells above the mat that start as floating bacteria

The script checks that the abundances, the state keys, the `SpeciesInteraction` coefficients and the `species_keys` of `BiofilmDetachment` and `GutDrift` all name the same species.

The `SpeciesInteraction` coefficients in `gut.json` are all 0, as placeholders.

Some of the target can't be reached by tuning coefficients: `BBBBT` starts at 0 but is 0.047 in `after`, and a species only spawns next to cells of its own kind, so it can never appear. That alone keeps `error_vs_after` above about 0.0022.

## The starting grid

1. **Biofilm mat.** A mat grows from every wall of the `BiofilmDetachment` rule (`walls`, default `["y-"]`, the floor). Its height over each wall cell is a smooth random landscape: random noise blurred so that mounds are about `patch_size` cells across, scaled to `mean_thickness` ± `thickness_spread`, and cut off at 0, so some patches of wall are bare. Every column is filled from the wall up to its height, so the whole mat is attached to the wall.
2. **Lumen.** Every cell outside the mat is a floating bacterium with probability `lumen_density`.
3. **Species.** Mat and lumen cells together get species in the `before` proportions. Cells lost to rounding go to the most abundant species.

## Biofilm in the simulation

`BiofilmDetachment` (in `src/cellular_automata/rules/`) keeps the biofilm attached in bulk:

- A biofilm cell detaches with probability `detachment_rate × scaling × d²`, where `d` is its distance to the nearest wall scaled from 0 (at the wall) to 1 (farthest from every wall). Cells touching a wall, and cells with biofilm both above and below them, never detach.
- A floating bacterium with biofilm below it can attach.
- Biofilm no longer linked to a wall through neighbouring biofilm cells (Moore or von Neumann, as the config's `neighbourhood` says) becomes floating, and `GutDrift` carries it along the flow (z).

With the current `detachment_rate` (0.011111) and `scaling` (0.0005), `detachment_rate × scaling` is 5.6e-6. That is the chance per step for an exposed cell at the far side, so the mat hardly detaches at all. Those values were probably chosen for an older formula that used the raw layer number; raise their product (e.g. to about 0.05 for 5% at the far side) if you expect visible detachment.

## Past results

The Monte Carlo coefficient search from the old `gut-experiments` branch is archived on the orphan [`results`](https://github.com/MaruStalmach/Cellular-Automata/tree/results) branch, under `gut/`. Don't compare new runs with it: it predates the `BiofilmDetachment` fixes and the new starting grid, and it scored shares of all cells rather than of the occupied cells. Results worth keeping from new runs go on the same branch.

The Monte Carlo search script ported to this layout (`mc_search.py`) was dropped in favour of single runs; it is in this branch's history (commit "feat: gut Monte Carlo search on the shared simulation stack") if a search over the `initial_state` parameters is wanted later.
