# Gut microbiome study

Fitting species-interaction coefficients so a simulated gut community moves from one measured composition to another.

| Folder | Contents |
|---|---|
| `configs/` | `mc_search.json`: simulation and search settings for the Monte Carlo search. |
| `data/` | Input data; see [`data/README.md`](data/README.md). |
| `scripts/` | `mc_search.py`: Monte Carlo search for interaction coefficients. |
| `results/` | Created by the scripts; git-ignored. |

## Monte Carlo search

Run from the repository root:

```bash
python research/gut/scripts/mc_search.py                       # defaults from the config
python research/gut/scripts/mc_search.py --iterations 200 --workers 8 --seed 1
python research/gut/scripts/mc_search.py --config <config.json> --output <results.json>
```

Each guess draws every `SpeciesInteraction` coefficient uniformly from `mc_search.interaction_bounds`, runs `sim.max_steps` steps, and scores the result. The score is the sum of squared differences between the final relative abundances (each species' share of the occupied cells) and `abundances.after`. Lower is better.

`mc_search.json` is an ordinary simulation config: it builds with the parser (and so with `examples/CA_sim.py`). It has no initialisers, though, so on its own it starts from an empty grid; `mc_search.py` fills in the species and biofilm. It has two extra blocks that the parser ignores:

- `abundances`: `before` (the starting composition, one entry per species state key) and `after` (the target). Each must sum to 1.
- `mc_search`: `n_iterations`, `density` (share of cells occupied at the start), `interaction_bounds` and `checkpoint_every_s`.

The coefficients under `SpeciesInteraction` in the config are placeholders, and each guess replaces them. The search checks that the abundances, the state keys, the coefficients and the `species_keys` of `BiofilmDetachment` and `GutDrift` all name the same species.

**Initial state.** All guesses start from the same grid. A share of the cells equal to `density` is occupied, in the `before` proportions. Cells lost to rounding go to the most abundant species. An occupied cell starts as biofilm with probability 1 minus `BiofilmDetachment`'s detachment probability at its height. The height is measured along the axis that rule uses, which is y today. The wall axis is still an open decision (D2 in `docs/code-quality-plan.md`).

With the configured `detachment_rate` (0.011111) and `scaling` (0.0005), the rule's probability is at most 5.6e-6, at the top layer. So in practice every occupied cell starts as biofilm, and the rule detaches almost nothing during the run. The old script seeded along z with the raw layer index squared (up to about 0.054 on 100 layers, so about 1.8% of occupied cells started unattached). Those two parameters were probably set with that formula in mind, so revisit them together with D2.

**Seeding.** `sim.seed` (or `--seed`) is split with `SeedSequence.spawn` into three streams: the initial state, the coefficient draws, and one simulation seed per guess. The same seed and config repeat the whole search, whatever the number of workers. Guess *i* is also the same whatever `n_iterations` is.

**Output.** `results/mc_search_results.json` holds the search seed, the config used, every scored guess and the `best` one. Each guess has `index`, `seed`, `error`, `final_abundances`, `occupied_cells` and `interactions`; `occupied_cells` of 0 means the community died out. The file is rewritten every `checkpoint_every_s` seconds and at the end, including when the run is interrupted or a guess fails. `complete` says whether every guess was scored. To repeat one guess, `replay_guess(json.load(open(path)), index)` in `mc_search.py` returns the finished simulation.

## Past results

The Monte Carlo output from the old `gut-experiments` branch (`output.json`, `mc_search_run.npz`) is archived on the orphan [`results`](https://github.com/MaruStalmach/Cellular-Automata/tree/results) branch, under `gut/`. Don't compare new runs with it:

- it predates the `BiofilmDetachment` re-attachment fix (BUG-12) and the seeding fixes
- it scored abundances as shares of all cells, not of the occupied cells

Results worth keeping from new runs go on the same branch.

Gut-specific rules live in the library next to the others (`src/cellular_automata/rules/`), e.g. `GutDrift`, `SpeciesInteraction` and `BiofilmDetachment` (detachment into `floating_bacteria` with re-attachment).
