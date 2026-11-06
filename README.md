# Cellular Automata

A three-dimensional cellular automaton simulator for modelling bacterial populations, biofilm formation, microbial interactions, and transport processes. The project is designed for experiments involving bacterial dynamics in bioreactors, laboratory biofilm systems, and the human gut.

The simulator represents a spatial environment as a 3D grid. Each cell may contain bacterial species, biofilm attachment, floating bacteria, substrate concentrations, and other experiment-specific state variables. Configurable rules update the state over time, while an OpenGL renderer provides interactive visualization.


## Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Installation](#installation)
- [Running Simulations](#running-simulations)
- [Running Tests](#running-tests)
- [Authors](#authors)

## Features

- Three-dimensional cellular automaton grid with configurable grid dimensions and independent periodicity per axis.
- Handling of multiple bacterial species.
- Biofilm and floating-bacteria state variables.
- Rule-based simulation architecture with configurable rule-specific time steps.
- Species interaction rules.
- Biofilm detachment rules.
- Spontaneous bacterial spawning.
- Support for multi-layer variables such as substrates.
- Bacterial drift and transport.
- JSON-based configuration.
- OpenGL and GLFW visualization and headless simulation mode.

## Architecture Overview

![image](docs/assets/architecture_CA.png)


### Geometry

`Geometry` defines the spatial domain and boundary behavior.

It handles:

- Grid dimensions.
- Axis ordering.
- Periodic boundaries.
- Neighbor and offset calculations.
- Spatial indexing.

The `periodicity` setting specifies which axes wrap around:

- `"xyz"` — all axes are periodic.
- `"xy"` — x and y are periodic; z has boundaries.
- `"z"` — only z is periodic.
- `""` — all axes are non-periodic.

### State

`State` stores values associated with every grid cell. State variables are stored as separate NumPy arrays in an internal dictionary.

Typical keys include:

- Bacterial species names.
- `biofilm`.
- `floating_bacteria`.
- Substrate variables.
- Experiment-specific fields.

Prefer accessing individual variables:

```python
bacteria = state["Bacteroidaceae"]
state["biofilm"] = biofilm
```

The combined `state.data` property should be used carefully because it stacks variables and expands multi-layer values. Individual keys preserve each variable's data type and layer structure.

### Multi-layer State Variables

Some variables, such as substrates, can contain multiple layers. Layer counts are configured through `key_layers` or the `State` constructor.

A multi-layer variable may have a shape conceptually equivalent to:

```text
(x, y, z, layers)
```

Code operating on state data should account for this additional trailing dimension.

### Rules

Rules are independent processes that modify the simulation state. Each rule can have its own time step and parameters.

Examples include:

- `biofilm_detachment`
- `species_interaction`
- `spontaneous_spawn`
- `gut_drift`

Rules are configured in JSON and executed by the cellular automaton when their scheduled time step is reached.

### Cellular Automaton

The cellular automaton coordinates:

1. Loading the current state.
2. Determining which rules are due.
3. Applying the rules.
4. Updating state variables.
5. Advancing simulation time.
6. Invoking callbacks or rendering.

### Renderer

The renderer uses GLFW and OpenGL to display the simulation in three dimensions.

It supports:

- 3D visualization.
- Shader-based rendering.
- Transparency.
- Order-independent transparency using framebuffer ping-pong.
- Interactive camera controls.

Controls:

- P: advance simulation.
- W/S: Zoom in/out.
- A/D: Rotate left/right.
- Q/E: Rotate up/down.
- C: Toggle cross section mode.
- J/L: cycle cross section layers up/down.
- X: cycle cross section axis.

![gif](docs/assets/full5.gif)

## Repository Structure

```text
.
├── src/cellular_automata/     # the simulation library (installable package)
│   ├── Geometry.py, State.py, Sim.py, Rule.py, Callback.py, Neighbourhood.py
│   ├── neighbourhoods/        # Moore / von Neumann neighbourhoods
│   ├── rules/                 # one file per rule (BacteriaGrowth, Diffusion, GutDrift, ...)
│   ├── rendering/             # OpenGL renderer, camera, shaders/
│   ├── tracking/              # StateTracker: saves simulation state to .npz
│   ├── config/Parser.py       # builds a simulation from a JSON config
│   └── util/
├── configs/examples/          # small, ready-to-run example configs
├── examples/                  # runnable scripts (rendered / headless runs, replay, GIF export)
│   └── notebooks/
├── research/                  # study-specific work, one folder per study
│   ├── bioreactor/            # configs/, data/, notebooks/, scripts/, results/
│   └── gut/                   # gut microbiome study
├── tools/profiling/           # CPU / RAM monitoring and cProfile helpers
├── tests/
├── docs/assets/               # images used in this README
└── pyproject.toml
```

Generated output (tracked simulations, GIFs, profiles) is written to `output_files/`, which is git-ignored.

### Adding new things

- **A new rule** - add a module to `src/cellular_automata/rules/` with a `Rule` subclass and import it in `src/cellular_automata/config/Parser.py`; configs then refer to it by class name.
- **A new study** - create `research/<study>/` with the same `configs/`, `data/`, `notebooks/`, `scripts/`, `results/` layout as `research/bioreactor/`.
- **A new example config** - add it to `configs/examples/`.

## Configuration

Simulations are configured with JSON files. A configuration typically contains:

- Simulation dimensions and timing.
- Boundary conditions.
- Wall-seed initialization.
- Rules and parameters.
- Additional state keys.
- Initial species abundances.

Example configuration:

```json
{
  "sim": {
    "size": [50, 50, 50],
    "periodicity": "xyz",
    "density": 0.2,
    "default_time_step": 1.0,
    "max_steps": 500
  },
  "wall_seed": {
    "enabled": false,
    "density": 0.0,
    "decay_length": 1.0
  },
  "rules": [
    {
      "name": "species_interaction",
      "time_step": 1.0,
      "params": {
        "spawn_claim_scale": 10.0
      }
    },
    {
      "name": "spontaneous_spawn",
      "time_step": 1.0,
      "params": {
        "spawn_probabilities": {
          "Bacteroidaceae": 0.002,
          "Lachnospiraceae": 0.002,
          "other": 0.001
        }
      }
    },
    {
      "name": "gut_drift",
      "time_step": 1.0,
      "params": {
        "drift_speed": 2
      }
    }
  ],
  "additional_keys": [
    "biofilm",
    "floating_bacteria",
    "substrate_glucose"
  ],
  "abundances": {
    "before": {
      "Bacteroidaceae": 0.5,
      "Lachnospiraceae": 0.3,
      "other": 0.2
    }
  }
}
```

## Installation

### Requirements

- Python 3.12.
- An OpenGL-compatible display and GLFW-compatible graphics support for rendered simulations.

### Setup

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # library + dev tools (pytest, ruff)
uv sync --extra research     # also install pandas, imageio, psutil for examples/research/tools
```

Or with pip:

```bash
pip install -e ".[research]"
```

## Running Simulations

Run commands from the repository root (prefix them with `uv run` when using uv).

### Rendered Simulation

```bash
python examples/CA_sim.py -f configs/examples/draft.json
```

The renderer opens an interactive window. Press **P** to advance the simulation.
Configs with `"render": false` in their `sim` section run headless with the same command.

Config paths are resolved relative to the current directory, so research configs work the same way:

```bash
python examples/CA_sim.py -f research/bioreactor/configs/full18.json
```

### Headless Simulation

```bash
python examples/sim_no_render.py -x 20 -y 20 -z 20 -s 100
```

Arguments:

| Argument | Description |
|---|---|
| `-x` | Grid size along the x axis. |
| `-y` | Grid size along the y axis. |
| `-z` | Grid size along the z axis. |
| `-s` | Number of simulation steps. |

Headless mode is useful for:

- Automated experiments.
- Parameter sweeps.
- Performance testing.
- Remote machines.
- Systems without a graphical display.

### Replaying and Exporting

Configs with a `tracker` section save their state to `output_files/tracked_simulations/<save_as>.npz`, which can be replayed or exported:

```bash
python examples/CA_sim.py -f configs/examples/demo.json          # writes output_files/tracked_simulations/demo.npz
python examples/Replay_sim.py -f demo -k bacteria substrate
python examples/generate_gif.py -f demo -k bacteria --duration 8  # writes output_files/gifs/demo.gif
```

## Running Tests

```bash
uv run pytest        # or: python -m pytest
```

## Authors

This repository was developed as a Master's research and software project for modelling bacterial and biofilm dynamics.

- **Authors:** Maria Stalmach, Jakub Wojciechowski
- **Institution:** AGH University of Science and Technology

