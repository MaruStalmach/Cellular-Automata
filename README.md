# Cellular Automata

A three-dimensional cellular automaton simulator for modelling bacterial populations, biofilm formation, microbial interactions, and transport processes. The project is designed for experiments involving bacterial dynamics in bioreactors, laboratory biofilm systems, and the human gut.

The simulator represents a spatial environment as a 3D grid. Each cell may contain bacterial species, biofilm attachment, floating bacteria, substrate concentrations, and other experiment-specific state variables. Configurable rules update the state over time, while an OpenGL renderer provides interactive visualization.


## Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Running Simulations](#running-simulations)
- [Known Limitations](#known-limitations)
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

![image](../Cellular-Automata/resources/architecture_CA.png)


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

The renderer currently advances the simulation when the user presses **P** rather than continuously.

## Repository Structure

```text
.
├── CA_sim.py
├── sim_no_render.py
├── CA_*.py
├── libs/
│   ├── Geometry.py
│   ├── State.py
│   ├── Rules.py
│   ├── Sim.py
│   ├── Callbacks.py
│   ├── Rendering/
│   │   └── shaders/
│   └── config/
│       └── Parser.py
├── experiments/
├── tests/
├── `pyproject.toml`
├── `AGENTS.md`
└── `README.md`
```

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
- NumPy and project dependencies from `pyproject.toml`.
- An OpenGL-compatible display for rendered simulations.
- GLFW-compatible graphics support for the renderer.

## Running Simulations

Run commands from the repository root.

### Rendered Simulation

```bash
python CA_sim.py -f `sim_config.json`
```

The renderer opens an interactive window. Press **P** to advance the simulation.

The equivalent command without `uv` is:

```bash
python CA_sim.py -f `sim_config.json`
```

### Headless Simulation

```bash
python sim_no_render.py -x 20 -y 20 -z 20 -s 100
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

## Authors

This repository was developed as a Master's research and software project for modelling bacterial and biofilm dynamics.

- **Authors:** Maria Stalmach, Jakub Wojciechowski
- **Institution:** AGH University of Science and Technology

