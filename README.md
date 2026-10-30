# Cellular Automata

A three-dimensional cellular automaton simulator for bacterial populations,
biofilm formation, microbial interactions, and transport processes.

## Installation

The project uses Python 3.12 and `uv`:

```bash
uv sync
```

For the optional bioreactor screening and monitoring tools:

```bash
uv sync --extra research
```

## Running the simulator

Run commands from the repository root.

### Headless example

```bash
python -m examples.sim_no_render -x 20 -y 20 -z 20 -s 100
```

### Rendered simulation

```bash
python -m examples.CA_sim -f research/configs/demo.json
```

The renderer opens an interactive window. Press **P** to advance the
simulation. The OpenGL examples require a compatible display.

Configurations are stored in [research/configs](research/configs). The
configuration parser also accepts an absolute path or a path relative to the
repository root.

## Repository structure

```text
.
├── examples/             # Runnable simulator and visualization examples
├── libs/                 # Core simulator library
├── research/
│   ├── configs/          # Experiment and simulation configurations
│   ├── data/             # Bioreactor research data
│   ├── examples/         # Research notebooks
│   ├── monitoring/       # Profiling and resource monitoring tools
│   ├── results/          # Screening and optimization result tables
│   └── screening/        # Bioreactor screening and optimization scripts
├── resources/            # Diagrams and other documentation assets
├── tests/                # Automated tests
├── pyproject.toml
└── uv.lock
```

Simulation snapshots are generated under `output_files/` and are ignored by
Git. Research snapshots and other generated files under `experiments/` are
also ignored.

## Tests

```bash
uv run pytest tests/
```

## Research tools

The screening scripts use the data in `research/data` and write result tables
to `research/results`:

```bash
python -m research.screening.analyze_screening
python -m research.screening.bayesian_optimization
```

The notebooks in [research/examples](research/examples) are exploratory
research material and are not required to run the core simulator.
