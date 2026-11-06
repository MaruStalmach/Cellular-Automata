# Gut microbiome study

Work in progress. Follow the same layout as `research/bioreactor/`:

```text
research/gut/
├── configs/     # simulation configs for this study
├── data/        # input data
├── notebooks/   # exploration and visualisation
└── scripts/     # experiment / analysis scripts
```

Scripts may write to `results/`, which is git-ignored. Results worth keeping go on the orphan [`results`](https://github.com/MaruStalmach/Cellular-Automata/tree/results) branch, under `gut/`. It already holds the Monte Carlo coefficient search output from the `gut-experiments` branch.

Gut-specific rules live in the library next to the others (`src/cellular_automata/rules/`), e.g. `GutDrift`, `SpeciesInteraction` and `BiofilmDetachment` (detachment into `floating_bacteria` with re-attachment).
