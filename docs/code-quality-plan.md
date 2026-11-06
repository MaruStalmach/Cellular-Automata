# Code Quality Review & Improvement Plan

Branch: `refactor/code-quality-plan` (based on `refactor/repo-structure`, PR #33)
Scope: `src/cellular_automata/`, `tests/`, `examples/`, `research/bioreactor/scripts/`, `tools/`, all configs, and the unmerged `gut-experiments` branch (reviewed at `e3d7313`).
Date: 2026-11-06 (updated the same day for `gut-experiments` commit `e3d7313`)
Status: updated 2026-11-06 with the fixes in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36) (Phase 0-1) and [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37) (Phase 2). Both PRs are open, stacked on #33.

---

## How to read this document

Each finding has an **ID** (e.g. `BUG-3`), a **severity**, a location, and a fix. The [step-by-step plan](#step-by-step-plan) at the end orders them into small, reviewable PRs.

| Severity | Meaning |
|---|---|
| **Critical** | Broken today, or silently produces wrong scientific results |
| **High** | Will bite users or block planned work (e.g. gut features) |
| **Medium** | Maintainability / clarity problem with real cost |
| **Low** | Polish |

**Status.** A struck-through heading means the finding is fixed; the **Status** line under it links the PR. **Partly fixed** findings say what is still open. Findings without a Status line are open.

**Evidence.** Findings marked ✅ were confirmed by running code on tiny inputs (no full simulations) or by static checks (signature comparison, imports, `ruff`). Everything else comes from reading the code and cites file and line.

---

## Summary - the top 10

| # | ID | Severity | Problem | Status |
|---|---|---|---|---|
| 1 | [BROKEN-1](#broken-1) | Critical ✅ | ~~3 example scripts crash on start (`from cellular_automata.rules import X` imports a *module*)~~ | Fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36) |
| 2 | [BROKEN-2](#broken-2) | Critical ✅ | ~~3 of 8 example configs are missing the required `time_step` and fail to parse~~ | Fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36) |
| 3 | [BUG-1](#bug-1) | Critical ✅ | ~~`int8` overflow in `Rule.neighborhood_count`: dense substrate gives **negative** neighbour counts~~ | Fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37) |
| 4 | [REPRO-1](#repro-1) | Critical | No RNG seeding anywhere in the library, so runs can't be reproduced; parallel MC search likely reuses identical random streams | Open |
| 5 | [ARCH-1](#arch-1) | High | Rules are looked up through `globals()` + wildcard imports, which has no validation and hides typos | Open |
| 6 | [ARCH-2](#arch-2) | High | State keys (`"biofilm"`, `"floating_bacteria"`, `"alive"`) and the "wall" axis are hard-coded inside rules, and gut and bioreactor rules disagree on which axis is the wall | Open (needs D2) |
| 7 | [ARCH-4](#arch-4) / [RES-1](#res-1) | High | Two incompatible config formats (Parser vs `gut-experiments`), and the README documents the wrong one. `gut-experiments` no longer merges cleanly (12 conflicts) | Open |
| 8 | [TEST-1](#test-1) | High ✅ | ~~4 tests fail because they are out of sync with the code (2 need a modelling decision)~~ | Fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36) (2 tests xfail until D1) |
| 9 | [HYG-1](#hyg-1) | High | No `LICENSE`, so nobody else may legally use the code | Open (needs D6) |
| 10 | [ARCH-3](#arch-3) | High | Rendering (OpenGL/GLFW) is a hard dependency of the config parser, and the renderer calls `quit()` | Partly fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36) |

---

## 1. Broken right now

<a id="broken-1"></a>
### ~~BROKEN-1 - Example scripts import modules instead of classes (Critical ✅)~~

> **Status:** Fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36). `rules/__init__.py` exports the classes; each example exposes `build_sim(size)` and has a smoke test. Still to do: `gut-experiments: mocks/CA_neighbourhoods.py`, when it is ported (step 20).

- **Where:**
  - `examples/sim_no_render.py:1`
  - `examples/CA_diffusion.py:1`
  - `examples/CA_interaction.py:1`
  - `gut-experiments: mocks/CA_neighbourhoods.py:20`, `:95` (new in `e3d7313`): `super().__init__(geometry)` without `time_step`, and `renderer.run(update_callback=...)`
- **What:** `from cellular_automata.rules import GameOfLife3D` gives the *module* `GameOfLife3D.py`, because `rules/` has no `__init__.py` exporting classes. Calling it raises `TypeError: 'module' object is not callable` (verified).
- **More breakage in the same files:**
  - `renderer.run(update_callback=...)`: `CARenderer.run()` takes no arguments.
  - `SpeciesInteraction(geometry, ic)` passes the interactions dict as `time_step`.
  - `Diffusion(geometry, a=6, p=0.08)` has no `time_step`.
  - `GameOfLife3D(geometry, 2, 3, 3, 3)` passes `time_step=2, eb=3, ...`, which is not the intended rule.
  - `tools/profiling/monitoring_sim.py` launches the broken `sim_no_render.py`.
- **Fix:** add `rules/__init__.py` (see [ARCH-1](#arch-1)), use keyword arguments everywhere, pass the callback via the constructor. Add a smoke test that imports every example and builds (not runs) its simulation.

<a id="broken-2"></a>
### ~~BROKEN-2 - Example configs missing required arguments (Critical ✅)~~

> **Status:** Fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36), with a test that builds every config in the repo.

- **Where:**
  - `configs/examples/diffusion_utilization.json`: `Diffusion`, `Utilization`
  - `configs/examples/growth_decay.json`: `Diffusion`, `BacteriaGrowth`, `BacteriaDecay`
  - `configs/examples/growth_test.json`: `Diffusion`, `BacteriaGrowth`
- **What:** `time_step` became a required constructor argument, but these configs were never updated. `parse_json` raises `TypeError`.
- **Fix:** add `time_step`. Add a test that runs `parse_json` on every config in the repo with `render` forced off.

<a id="broken-3"></a>
### ~~BROKEN-3 - `CA_sim.py` uses an exception for control flow; headless loop is unreachable when rendering (High)~~

> **Status:** Fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36).

- **Where:** `examples/CA_sim.py:24-34`, `rendering/Renderer.py:736-737`
- **What:**
  - With `render: false`, `parse_json` returns `renderer=None`. `renderer.run()` then raises `AttributeError`, which is caught and printed as `"Attribute error in CA_SIM"`, and *then* the headless loop runs. Any genuine `AttributeError` inside the renderer is swallowed the same way.
  - With `render: true`, `CARenderer.run()` ends with `quit()`, killing the process. The tracker is never saved for rendered runs.
- **Fix:** `if renderer is None: run_headless(sim) else: renderer.run()`. Remove `quit()` from the library (call `glfw.terminate()` and return).

<a id="broken-4"></a>
### ~~BROKEN-4 - Parser crashes on valid-looking renderer blocks (High)~~

> **Status:** Fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36), with tests.

- **Where:** `config/Parser.py:177-181`
- **What:**
  - With no `callback` in the `renderer` block, it calls `Callback()` with no arguments, which is a `TypeError`.
  - `cb_args = {} or callback_data.get("args", None)` always evaluates to `callback_data.get("args")`. With no `args`, `**None` raises `TypeError`.
- **Fix:** `callback = Callback(key, ca_sim)` and `cb_args = callback_data.get("args") or {}`.

---

## 2. Correctness bugs in the model

<a id="bug-1"></a>
### ~~BUG-1 - `int8` overflow in neighbour counting (Critical ✅)~~

> **Status:** Fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37), with a regression test. Still to do: re-check key bioreactor runs (step 8).

- **Where:**
  - `Rule.py:52`: `astype(np.int8)`
  - `Rule.py:56`: `count = np.zeros_like(..., dtype=np.int8)`
  - `BacteriaGrowth.py:58`, `Utilization.py:132`: `.astype(np.int8)`
- **What:** with multi-layer keys (substrate has 6 layers) a 3D Moore neighbourhood can hold 26 × 6 = 156 particles. `int8` wraps: a fully surrounded cell reports **-100** (verified). Growth and uptake probabilities are then clipped to 0, so bacteria in the *richest* regions stop growing or eating. Single-layer keys overflow above 127 too, e.g. with a radius > 1 neighbourhood.
- **Fix:** count in `np.int32` (or `np.uint16`), and remove the `int8` casts in the rules. Add a regression test using the 3×3×3×6 all-ones case.
- **Note:** results produced with dense substrate may be affected. Worth re-checking key bioreactor runs after the fix.

<a id="bug-2"></a>
### ~~BUG-2 - `_shift_for_offset` shifts in opposite directions on periodic vs non-periodic axes (Medium ✅)~~

> **Status:** Fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37). Both branches now return `result[i] == grid[i + offset]`; the unused copy in `Utilization` is deleted.

- **Where:** `Rule.py:17-47` (duplicated in `Utilization.py:29-57`)
- **What:** `np.roll(shift=+1)` moves data *forward*, but the pad-and-slice branch moves it *backward* (verified: the same offset `(+1,0,0)` moves a cell to index 2 on a periodic axis and to index 0 on a non-periodic one). Neighbour *counts* are still correct only because Moore and von Neumann offsets are symmetric. Any directional use (drift, chemotaxis, asymmetric neighbourhoods) would silently be wrong.
- **Fix:** pick one convention (`result[i] = grid[i + offset]`), implement both branches to match, and test it on both axis types.

<a id="bug-3"></a>
### BUG-3 - `MooreNeighbourhood` with `radius > 1` skips cells (Medium)

> **Status:** Partly fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37): `_shift_for_offset` now pads by the offset size, so offsets > 1 work on non-periodic axes. **Open, needs D9:** `test_radius_scaling_3d` asserts that radius 2 means only cells at exactly ±2 (a dilated neighbourhood), so skipping the ±1 cells may be intended. Von Neumann behaves the same way.

- **Where:** `neighbourhoods/MooreNeighbourhood.py:12`
- **What:** `product((-radius, 0, radius))` gives only the corners, faces and edges at exactly ±radius. For radius 2 it misses the ±1 cells. Also, `Geometry` never passes `radius`, so it is always 1 today.
- **Fix:** `product(range(-radius, radius + 1), repeat=ndim)`. Either expose `radius` through `Geometry` and the config or remove the parameter.

<a id="bug-4"></a>
### ~~BUG-4 - Cell-wise rule path is dead and would crash (Medium)~~

> **Status:** Fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37). `_apply_cellwise` and `Rule.apply` are deleted; `apply_state` still raises `NotImplementedError` rather than being an `abc` abstract method.

- **Where:** `Sim.py:113-116`, `Rule.py:63-68`
- **What:** `hasattr(rule, "apply_state")` is always `True`, because the base class defines it (raising `NotImplementedError`). So `_apply_cellwise` is never used, and a rule that only implements `apply()` crashes.
- **Fix:** delete the cell-wise path (it is also O(N) Python and would be unusably slow) and make `apply_state` an abstract method. Or keep it and check `type(rule).apply_state is Rule.apply_state`.

<a id="bug-5"></a>
### ~~BUG-5 - Rule scheduling is undocumented and silently capped (High)~~

> **Status:** Fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37). The time model is documented in the README ([#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36)). The scheduler warns when the 500 cap applies or a ratio is not whole, and float ratios are no longer floored (`10 // 0.1 == 99`). Run counts of existing configs are unchanged. **New finding (D10):** every bioreactor config hits the cap: Diffusion and Utilization at `dt=0.01` under an 1800 s step run 500 of 180,000 intended times (0.28%).

- **Where:** `Sim.py:37-52`, `Sim.py:108-117`
- **What:**
  - One "step" lasts `max_dt` (the comment says "smallest dt"; the code takes the largest).
  - Each rule runs `max_dt // rule.dt` times, **clipped to 500** with no warning. For example, diffusion at `dt=0.01` under a `dt=1800` rule would need 180,000 runs per step, but runs 500, so the effective diffusion speed is silently wrong.
  - Rules are interleaved round-robin, so order matters and is invisible to the user.
  - `max_dt // rule.dt` floors non-integer ratios.
- **Fix:** document the time model in the README. Warn or raise when the cap is hit (or make it configurable). Validate that the ratios are integers.

<a id="bug-6"></a>
### ~~BUG-6 - `BioreactorBiofilmDetachment` key handling (Medium)~~

> **Status:** Fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37), with tests. The fix also covers a worse bug found while fixing this one: `biofilm_grid & (chances < p)` is a `uint8` array, so `biofilm_grid[mask] = 0` used integer indexing and cleared the whole x=0 slab every step (and x=1 whenever anything detached), even at a detachment rate of 0. 18 bioreactor configs use this rule, so biofilm results should be re-checked. This should have been rated **Critical**.

- **Where:** `rules/BioreactorBiofilmDetachment.py:17`, `:40`
- **What:**
  - `self.required_keys.extend(self.target_key)` extends with a *string*, so it adds `"b","i","o",...` as keys.
  - It reads `state[biofilm_key]` but writes `state["biofilm"]`, so a custom key name is ignored on write.
- **Fix:** `append(biofilm_key)` and `state[self.target_key] = ...`.

<a id="bug-7"></a>
### BUG-7 - Diffusion mass "restoration" is non-physical and can crash (Medium)

- **Where:** `rules/Diffusion.py:186-195`, `:214-231`
- **What:**
  - Colliding particles are deleted. Once 100 are lost (`RESTORE_THRESHOLD`, hard-coded), 100 new ones are placed at random in the top z-layer. Mass is conserved only on average, and the source location is fixed to "max z" regardless of the experiment.
  - `np.random.choice(..., replace=False)` raises if fewer than 100 empty slots exist at max z.
  - `_restore_deterministic` is unused.
  - The comment at `:185` ("deal with collisions >1 values ... idk") and the commented-out block at `:162-177` show this was a stopgap.
- **Fix:** make the boundary or source behaviour an explicit, configurable choice (reflecting walls, inflow boundary, or an `Inoculation` rule). Guard against too few empty cells. Delete the dead code.

<a id="bug-8"></a>
### BUG-8 - `AntibioticEffect` has hidden constants and odd maths (Medium)

- **Where:** `rules/AntibioticEffect.py:6`, `:18`, `:32-36`, `:40-41`
- **What:**
  - `DIFFUSION_TIME_STEP = 0.01` is a hard-coded coupling to another rule's `dt`.
  - `time_step // 0.01` floors.
  - `[:, :, 0] += 9` assumes the wall is at z = 0.
  - `/27` assumes 3D Moore.
  - Probabilities are `float16` (3 significant digits).
  - `effectiveness` divides the *random roll* rather than scaling the probability.
  - `inoculation_start/end` really means dosing window, and `self.end` is mutated lazily.
  - It doesn't declare `required_keys`.
- **Fix:** take the wall axis and side from geometry ([ARCH-2](#arch-2)), use `len(geometry._offsets) + 1`, `float32` everywhere, `p_death = base * scale * (1 + effectiveness) / protection`, and rename to `dose_start/dose_end`.

<a id="bug-9"></a>
### BUG-9 - Parser config handling hides errors (Medium)

> **Status:** Partly fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36): an unknown `neighbourhood` raises `ValueError` listing the known names. Still open: hard-coded `axes`, required `sim.render`, `RuntimeError` without a message, `random: true` without `random_func`.

- **Where:** `config/Parser.py:73-77`, `:80`, `:99`, `:132-138`, `:165`
- **What:**
  - A bare `except` turns a typo in `neighbourhood` into a silent fallback to Moore.
  - `axes="xyz"` is hard-coded, so 2D configs are impossible.
  - `sim.render` is required (`KeyError` if missing).
  - `RuntimeError` is raised without a message.
  - A state key with `"random": true` but no `random_func` silently stays all zeros (`State.py:80`).
- **Fix:** validate the config up front with clear messages ([ARCH-4](#arch-4)).

<a id="bug-10"></a>
### BUG-10 - `StateTracker` ignores its `keys` argument and loses data on interrupt (Medium)

> **Status:** Mostly fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37): the tracker records only its keys, `save`/`load` resolve paths the same way, and `CellularAutomaton.save_tracker()` (called by `CA_sim.py` when a run stops early) saves partial history. Still open: `record_every`.

- **Where:** `tracking/StateTracker.py:57-63` (`record_step`), `:65-91` (`save`), `:96-128` (`load`); `Sim.py:123-124`
- **What:**
  - `record_step` iterates *all* state keys, so passing a subset of `keys` gives a `KeyError`.
  - The whole history is kept in RAM and saved only when `step_no == max_steps`, so Ctrl-C or a crash loses everything.
  - `save()` always writes under `output_files/tracked_simulations/` relative to the CWD and ignores absolute paths, while `load()` accepts them.
- **Fix:** record only the tracked keys. Add an optional `record_every` stride. Save on interrupt (`try/finally`) or stream to disk. Make `save`/`load` path handling symmetric.

---

## 3. Reproducibility

<a id="repro-1"></a>
### REPRO-1 - No seeding; global `np.random` everywhere (Critical)

- **Where:** every rule (including the new `SpontaneousSpawn`, which uses `np.random.shuffle` / `np.random.random`); `State.py:69`; `gut-experiments: experiments/data_preparation.py:264`
- **What:**
  - There is no `seed` in configs, and the library uses the global legacy RNG. Two runs of the same config give different results and can't be reproduced, which matters for a scientific tool and for calibration.
  - `data_preparation.py` runs simulations in a `ProcessPoolExecutor`. With the Linux `fork` start method, every worker inherits the *same* global RNG state, so parallel "independent" simulations likely reuse the same random streams.
  - `bayesian_optimization.py:119` reseeds globally before each trial (good for common random numbers), but scores a stochastic simulation from **one** realisation with GP noise `1e-6`. That treats noisy RMSE as exact.
- **Fix:**
  - Add `sim.seed` to the config. Create one `np.random.Generator` in `CellularAutomaton` and pass it to rules (`rule.rng`).
  - In parallel code, derive child seeds with `np.random.SeedSequence(seed).spawn(n)`.
  - For BO, average several seeds per point or use a realistic noise term.

---

## 4. Architecture & design

<a id="arch-1"></a>
### ARCH-1 - Rule lookup through `globals()` and wildcard imports (High)

- **Where:** `config/Parser.py:6-30` (21 `import *`; `BacteriaDecay` imported twice), `:34-45`, `:73`, `:132`, `:150`, `:181`
- **What:**
  - Any global name in `Parser.py` can be instantiated from a config (`"name": "Path"` would work).
  - Adding a rule means editing the Parser.
  - Typos only fail late with `KeyError`.
  - `ruff` reports 358 `F403`/`F405` findings from this pattern.
  - The `gut-experiments` branch already built a cleaner explicit `RULE_REGISTRY`.
- **Fix:**
  - Add `rules/__init__.py` with explicit exports and a registry (e.g. a `@register_rule` decorator, or a dict built from `Rule.__subclasses__()`).
  - Use the same idea for neighbourhoods, trackers, callbacks and the allowed `random_func` (`np.random.choice`, `init_block`, `random_seeds`).
  - Unknown names raise `ValueError("Unknown rule 'X'. Known: [...]")`.
  - New gut rules then need **one file, no Parser edit**.

<a id="arch-2"></a>
### ARCH-2 - Hard-coded state keys and inconsistent "wall" axis (High)

- **What:** rules assume key names and geometry instead of taking them as parameters, and different rules disagree on which axis is the wall.

| Rule | Hard-coded | Wall / flow axis |
|---|---|---|
| `BiofilmDetachment` (gut) | `"biofilm"`, `"floating_bacteria"` (`:23`, `:41-42`, `:94-95`) | **y** = distance from wall (`:30`); "below" = y-1 (`:74`) |
| `BioreactorBiofilmDetachment` | writes `"biofilm"` (`:40`) | **z** = last axis (`:24-26`) |
| `GutDrift` | - | flow along **z** (last axis, `:29`) |
| `AntibioticEffect` | - | wall at **z = 0** (`:32`) |
| `SpeciesInteraction` | clears `"biofilm"`, `"floating_bacteria"` on death (`:113`) | - |
| `GameOfLife3D` | ignores `target_key`, uses `"alive"` (`:41-47`) | - |
| `util.random_seeds` | 3D only, seeds at z = 0 | **z** |
| `gut-experiments` `seed_biofilm` | - | **z** |

Gut and bioreactor code don't agree on where the wall is: y in `BiofilmDetachment`, z everywhere else, including the gut branch's own seeding. It is very likely that one is a bug.

- **Fix:** add `wall_axis` and `wall_side` (and optionally `flow_axis`) to `Geometry`, set in the config. Every rule that needs a wall reads them from there. All key names become constructor parameters with today's names as defaults, so existing configs keep working.

<a id="arch-3"></a>
### ARCH-3 - Simulation core is coupled to rendering (High)

> **Status:** Partly fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36): `quit()` is gone from the renderer, a failed window creation raises, and the examples import the renderer lazily. Still open: the parser split, the optional `render` extra, and shader errors. CI installs `libgl1` until then.

- **Where:** `config/Parser.py:29` imports `CARenderer` (`glfw`, `OpenGL`) at module level; `Renderer.py:113-115`, `:736-737`; `Shader.py:86-107`
- **What:**
  - Headless runs on clusters or CI still need OpenGL and GLFW installed and importable.
  - If window creation fails, it carries on with `window=None`.
  - Shader compile and link errors are only printed (and the fragment-shader error is mislabelled "Vertex Shader Error").
  - It needs OpenGL ≥ 4.3 (SSBO), which is undocumented.
- **Fix:**
  - Split `parse_json` into `build_simulation(config)` (core only) and `build_renderer(config, sim)` (imports rendering lazily).
  - Move `glfw`/`pyopengl`/`pyglm` to an optional `render` extra.
  - Raise on window or shader failures.
  - Document the OpenGL requirement.

<a id="arch-4"></a>
### ARCH-4 - Two config formats; no schema; 25 near-duplicate configs (High)

- **What:**
  - **Format A** (Parser, used by examples and bioreactor): `{"sim": {...}, "state_keys": [...], "rules": [{"name": "BiofilmDetachment", "args": {...}}], "tracker", "renderer"}`.
  - **Format B** (`gut-experiments/experiments/mc_config.json`): snake_case rule names, `"params"`, per-rule `"time_step"`, `"additional_keys"`, `"abundances"`, `"mc_search"`. It is built by a separate `build_rules` that injects `geometry/time_step/species_keys/interactions` by reflection.
  - ~~The README's "Example configuration" documents **format B**, so it doesn't work with `CA_sim.py`.~~ Fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36).
  - The Parser also accepts aliases (`neighbourhood`/`neighborhood`, `name`/`type`, `save_as`/`filename`).
  - `research/bioreactor/configs/` has 25 configs of about 290 lines that differ by 24-280 lines each (`full2`…`full18`, `bo`, `screen*`, `*antib*`). Changes have to be copied by hand.
- **Fix:**
  - Define one schema (a `dataclass` or `pydantic` model) for format A, validate on load, and drop the aliases.
  - Port format B's useful ideas into it: a per-rule `time_step`, an optional `initial_conditions` / `abundances` block, an optional `experiment` block for MC/BO settings.
  - Support `"extends": "base.json"` + overrides so `fullN` configs become small diffs.
  - ~~Fix the README example.~~ Done in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36).

<a id="arch-5"></a>
### ARCH-5 - `State` API is fragile (Medium)

- **Where:** `State.py`
- **What:**
  - `:47-48`: `key_layers.copy()` fails for tuples.
  - `:80-92`: a nested `try/except TypeError` guesses the call signature of `random_func` and can hide real `TypeError`s.
  - `:80`: `== None`.
  - `:110-125`: the `data` property copies and restacks every key on each access (used by the dead cell-wise path and the examples).
  - `:153-154`: `copy()` is marked `TODO: FIX THIS`.
  - `:167-193`: `_pad_data` is used only by a test and has unreachable comment lines after `return`.
  - `:218`: `_get_sim_reference` *sets* a back-reference `state.sim`, which `AntibioticEffect` uses to read `step_no`. That creates a circular State↔Sim dependency.
  - Single-layer keys are squeezed to the grid shape while multi-layer keys keep a trailing axis, so every rule has to branch on `ndim` (see `_cell_mask`, `CreateBiofilm:31`, `GutDrift:70`, `Inoculation:18`).
- **Fix:**
  - Accept any iterable for `key_layers`.
  - Make the initialiser explicit: a registry of init functions with known signatures.
  - Pass simulation context (step number, time, rng) to `apply_state(state, ctx)` instead of the back-reference.
  - Fix or test `copy()`.
  - Longer term, consider always keeping the layer axis (shape `(*grid, layers)`) to remove the `ndim` branching.

<a id="arch-6"></a>
### ARCH-6 - Rule interface is inconsistent (Medium)

- **What:**
  - Parameter names vary for the same concept: `bacteria_keys` / `species_keys` / `target_keys` / `b_keys`, and `target_key` / `biofilm_key` / `sub_key`.
  - Physical parameters have opaque names and defaults: `Yca=6.94e-6`, `time_scale=4320000`, `scaling`, `substitute_constant=0.37`, plus magic `27` in `Utilization.py:139`.
  - Validation uses `assert`, which is stripped with `python -O`.
  - `required_keys` is not declared by `AntibioticEffect`, `BacteriaDecay`, `Inoculation` or `Diffusion`. The auto-created state in `Sim.py:64` is therefore incomplete, and the key check in `Sim.py:56-62` always prints a mismatch anyway (list vs tuple, noted in a TODO).
  - `Inoculation` is 3D only, writes only layer 0, and fires every step.
- **Fix:**
  - Standardise names (`species_keys`, `substrate_key`, `biofilm_key`, `floating_key`).
  - Docstrings with units and a reference for every physical parameter.
  - `ValueError` instead of `assert`.
  - Every rule declares `required_keys`, and `Sim` validates them against the state with a clear error.
  - Give `Inoculation` an interval and a layer.

---

## 5. Tests

<a id="test-1"></a>
### ~~TEST-1 - 4 failing tests are out of sync with the code (High ✅)~~

> **Status:** Fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36). The two `SpeciesInteraction` tests are a non-strict `xfail` until D1: they are unseeded and sometimes pass by chance.

All four fail identically on `main` and `bioreactor`; this branch didn't cause them.

| Test | Cause | Action |
|---|---|---|
| `test_sim.py::test_gol_correctness` | `GameOfLife3D(g, 2, 3, 3, 3)` predates `time_step`, so args shift (`time_step=2, eb=3`) | Fix the test: `GameOfLife3D(g, time_step=1, eb=2, eh=3, fb=3, fh=3)` |
| `test_sim.py::test_bacteria_growth_consumes_only_one_substrate_particle_per_cell` | Growth no longer consumes substrate; that moved to `Utilization` (branch `split_growth_utilization`) | Rewrite as a `Utilization` test |
| `test_species_interaction_rule.py::test_destructive_interaction` | Test expects coeff −1 to *guarantee* a kill; code uses `|coeff| × n_neighbours / 8` (one neighbour gives 12.5%) | **Decision D1** |
| `test_species_interaction_rule.py::test_symbiotic_interaction` | Test expects B to *overwrite* A's cell; code only spawns into empty cells | **Decision D1** |

<a id="test-2"></a>
### TEST-2 - Large coverage gaps (High)

> **Status:** Partly fixed. [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36): every config builds, every example imports and builds, parser renderer and neighbourhood handling, the README config. [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37): the scheduler, `BioreactorBiofilmDetachment`, the tracker key/path handling, and regression tests for BUG-1, BUG-2 and BUG-6. Still open: tests for the other rules and seeded invariant tests (after REPRO-1).

There are 28 test functions. **No tests** exist for:
- rules: `Diffusion`, `Utilization`, `BacteriaDecay`, `AntibioticEffect`, `GutDrift`, `BiofilmDetachment`, `BioreactorBiofilmDetachment`, `Inoculation`
- the Parser and config loading
- `StateTracker` save/load round trip with layers
- the scheduler (`Sim.step` with mixed `dt`)

`tests/__init__.py` is unnecessary with the pytest `pythonpath` setting.

**Fix, in priority order:**
1. A test that loads every config.
2. Example smoke tests (build, don't run).
3. Per-rule invariant tests on tiny grids with a seeded RNG, for example:
   - diffusion conserves particles
   - drift moves floating cells by `drift_speed`
   - detachment never creates biofilm
   - no cell is occupied by two species
4. Regression tests for BUG-1, BUG-2 and BUG-6.

<a id="test-3"></a>
### TEST-3 - Tests rely on monkeypatching the global RNG (Medium)

- **Where:** `test_sim.py:170-182`; `test_species_interaction_rule.py` uses `np.random.seed(42)`
- **What:** patching `np.random.random` with call counting (`if self.calls == 3`) breaks whenever a rule's internal order of random calls changes.
- **Fix:** once rules take an injected `rng` ([REPRO-1](#repro-1)), tests can pass a seeded or stub generator.

---

## 6. Duplication & dead code (Medium)

| Item | Where | Action | Status |
|---|---|---|---|
| `_shift_spatial`, a copy of `Rule._shift_for_offset` | `Utilization.py:29-57` | Delete (unused) | Fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37) |
| `_neighbour_candidates` copied in 2 rules (unused in `Utilization`) | `BacteriaGrowth.py:29-46`, `Utilization.py:59-76` | Move to `Geometry.neighbours_of(coords)` or vectorise ([PERF-1](#perf-1)) | Open |
| `_get_neighbour_counts`, a copy of `neighborhood_count` | `SpeciesInteraction.py:43-50` | Use the base method | Open |
| Pointless `apply` override | `Utilization.py:118-119` | Delete | Fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37) |
| `_restore_deterministic`, commented-out wall block, alternative collision code | `Diffusion.py:119-121`, `:162-177`, `:200-212` | Delete | Open |
| Unused `self.buff` | `BacteriaDecay.py:14,21-23`, `AntibioticEffect.py:20` | Delete | Open |
| `_apply_cellwise`, `Rule.apply` | `Sim.py:71-98`, `Rule.py:63-64` | See [BUG-4](#bug-4) | Fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37) |
| Unused `shifted` parameter | `Geometry.py:42,46` | Remove | Open |
| `_parse_keys` / `_frame_for_keys` duplicated | `examples/Replay_sim.py`, `examples/generate_gif.py` | Move into `tracking/playback.py` | Open |
| `sys.path` hack (no longer needed with an installed package) | `examples/CA_drift_sim.py:1-7`, `gut-experiments: data_preparation.py:11-17` | Delete | `CA_drift_sim.py` fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36); gut file open |
| `BASE_INTERACTIONS` hard-coded in the script, duplicating config content | `research/bioreactor/scripts/screening.py:32-45` | Read from the base config | Open |
| `Renderer` `__main__` demo + commented arrays | `Renderer.py:741-752` | Move to `examples/` or delete | Open |

---

## 7. Performance (Medium)

<a id="perf-1"></a>
### PERF-1 - Python loops over cells in hot rules

- **Where:**
  - `BacteriaGrowth.py:74-98`: per dividing cell, it builds a Python list of neighbours and sorts the proposals.
  - `Utilization.py:158-178`: same pattern per consuming cell.
- **What:** these scale with the number of active cells in pure Python and will dominate run time on 250×250×100 grids (the size in `mc_config.json`).
- **Fix:** vectorise conflict resolution:
  1. Draw a random target offset per active cell.
  2. Compute target linear indices.
  3. Resolve ties with `np.unique(..., return_index=True)` after a random permutation.

  Measure with `tools/profiling` before and after.

### PERF-2 - Smaller items

- `Diffusion.py:96` re-allocates `self.rolls` every call (defeating the preallocated buffer), and `rolls` / `layers` are `float64` where `uint8` would do. That is 6 × grid × 8 bytes, twice.
- `Renderer.set_ca_state` calls `glBufferData` with a fresh `flatten()` copy every frame; use `glBufferSubData` on a persistent buffer.
- `StateTracker` keeps every frame in RAM (see [BUG-10](#bug-10)).

---

## 8. Research scripts & `gut-experiments`

<a id="res-1"></a>
### RES-1 - What `gut-experiments` contains (and what to do with it) (High)

As of `e3d7313` the branch has 4 commits not on the restructure branch.

**`ddae2e8` and earlier:**
- `experiments/data_preparation.py`, a parallel Monte Carlo search for interaction coefficients against before/after abundances
- `experiments/mc_config.json`
- `experiments/output.json` (a result)
- a one-line `Geometry` change: `generate_offsets(ndim=self.ndim)`, already compatible

The commit message "first iteration of tests" is misleading: that commit adds no tests.

**`e3d7313` "chore: preserve Python branch changes"** (17 files, +1114 / −101). This commit **copied** files from `main` instead of merging `main`, so a plain `git merge` now gives **12 conflicts**, mostly add/add and file-location conflicts caused by the `libs/` → `src/` and `mocks/` → `examples/` moves. Comparing each file against the restructure branch (with `libs.` mapped to `cellular_automata.`):

| File on `gut-experiments` | Compared with the restructure branch | Action |
|---|---|---|
| `libs/rules/CreateBiofilm.py`, `libs/rules/Inoculation.py`, `libs/rules/BiofilmDetachment.py`, `libs/tracking/StateTracker.py`, `tests/test_create_biofilm.py`, `tests/test_state_tracker.py`, `libs/rendering/Renderer.py` | **Identical** | Nothing to bring over |
| `libs/Sim.py` | **Older:** `step()` still calls `quit()` at `max_steps`, so any script calling `step()` in a loop kills the process; no `state.sim` back-reference | Keep the restructure version |
| `libs/config/Parser.py` | **Older:** no `AntibioticEffect` / `BioreactorBiofilmDetachment` imports, old package-relative config path, **and no `SpontaneousSpawn` import** | Keep the restructure version, then register `SpontaneousSpawn` ([RES-3](#res-3)) |
| `mocks/Replay_sim.py`, `mocks/generate_gif.py` | **Older:** no `--stride`, old paths | Keep the restructure version |
| `libs/State.py` | **Older:** no `sim` back-reference | Keep the restructure version |
| `libs/rules/SpontaneousSpawn.py` | **New** | Port → [RES-3](#res-3) |
| `experiments/scripts/{check_occupancy,compare_to_reference,print_ratios}.py` | **New** | Port → [RES-4](#res-4) |
| `mocks/CA_neighbourhoods.py` | **New** (Moore vs von Neumann demo) | Port to `examples/`, fix → [BROKEN-1](#broken-1) |

**Merge strategy:** don't merge the branch as-is. Port the *new* files into the new layout (`git checkout origin/gut-experiments -- <file>` + `git mv` + an import rewrite), take the restructure version for every shared file, and then retire the branch. A full `git merge` would mean resolving 12 conflicts that are all "ours", with a real risk of reintroducing the `quit()` in `Sim.step()`.

**What to adopt:**
- the explicit `RULE_REGISTRY` → [ARCH-1](#arch-1)
- runtime injection of shared parameters (`species_keys`, `interactions`) → a cleaner way to share parameters across rules in [ARCH-4](#arch-4)
- `build_species_grid` (initialising from abundances) → a reusable initialiser in the library
- checkpointing → reuse in BO and screening

**What to fix:**
- RNG under `fork` ([REPRO-1](#repro-1))
- wall axis: `seed_biofilm` uses z while `BiofilmDetachment` uses y ([ARCH-2](#arch-2))
- `build_species_grid` assigns the rounding remainder to the largest species (fine, but document it)
- `output.json` and the checkpoint should go in `research/gut/results/`

**Plan:** move it to `research/gut/scripts/mc_search.py` and `research/gut/configs/`, rewrite it on top of the unified config + registry, and delete the parallel `build_rules`.

### RES-2 - Bioreactor scripts (Medium)

- `screening.py:62-63`: `calc_rmse` assumes **1 step = 30 min** (`time * 2`), silently tied to `time_step: 1800` in the configs. It raises `IndexError` if a run is shorter than the data. Derive this from the config's `dt` and check the length.
- `screening.py:70-71`: renames `Other_bacteria` to `"Rest"` inline. Put the mapping in one place, next to the data.
- `bayesian_optimization.py`: the hand-rolled GP/EI has a fixed length scale of 0.35, no hyperparameter fitting, and assumes RMSE is noiseless ([REPRO-1](#repro-1)). That's acceptable for exploration, but document it or use `scikit-optimize` / `optuna`.
- `bayesian_optimization.py:16` and `screening.py:10`: both define `STUDY_DIR`. Put shared helpers in `research/bioreactor/scripts/common.py`.
- Constants (`SAMPLE_RUNS`, `BIO_NUMBER`, `SELECTED_PAIRS`, bounds) are hard-coded. Move them to CLI arguments or an experiment config.

<a id="res-3"></a>
### RES-3 - New `SpontaneousSpawn` rule (gut) (Medium)

- **Where:** `gut-experiments: libs/rules/SpontaneousSpawn.py`
- **What it does:** lets species with zero or locally absent abundance appear in empty cells (an independent Bernoulli draw per species per step, in shuffled species order). The design is sound and the docstring is good.
- **Problems:**
  - It is **not registered in the Parser** (no import), so configs can't use it. This is exactly the friction [ARCH-1](#arch-1) removes.
  - It uses the global RNG ([REPRO-1](#repro-1)).
  - It assumes single-layer species (`spawn_mask.reshape(state[sp].shape)` fails for layered keys).
  - It has no tests.
  - Like the symbiotic spawn in `SpeciesInteraction`, it can only place into empty cells. `check_occupancy.py` exists because the grid saturates and both spawn mechanisms then "freeze". That is a modelling issue: there is no turnover once the grid is full. It feeds into **D1** and the new **D7**.
- **Fix:** port to `src/cellular_automata/rules/`, register it, add `rng`, tests (empty-only placement, no double occupancy, probability 0 never spawns, seeded determinism), and an example config.

<a id="res-4"></a>
### RES-4 - Gut analysis scripts depend on things that aren't in the repo (High)

- **Where:** `gut-experiments: experiments/scripts/check_occupancy.py`, `compare_to_reference.py`, `print_ratios.py`
- **What:**
  - **Missing producer:** `check_occupancy.py` and `compare_to_reference.py` read a `ratios.csv` with `step`, per-family columns, `occupied_cells` and `total_cells`, "from the patched `experiment.py`". **No `experiment.py` exists on any branch**, so the pipeline that produced the gut results isn't committed.
  - **Format mismatch:** `print_ratios.py` expects one `step_XXXX.npz` per step. `StateTracker` writes one `.npz` with stacked history plus a `__state_tracker_metadata__` entry, so running `print_ratios.py` on tracker output would average over all steps and treat the metadata as a species.
  - **External data:** `compare_to_reference.py` defaults to `./dynamo_data/male_taxonomy_for_plot.csv` (the "dynamo repo" male donor dataset), which isn't in the repo. The old `cleanup/repository-structure` branch had LFS rules for `experiments/dynamo_data/**`.
  - **Three species naming schemes for the gut:**
    - `mc_config.json` uses codes (`BBBBB`, `BFCLL`, …)
    - `compare_to_reference.py` uses 8 family names (`Bacteroidaceae`, `Oscillospiraceae_88309`, `Peptoniphilaceae`, …)
    - the bioreactor uses 12 different families with `Other_bacteria`/`Rest`
  - **Three separate error metrics:** `screening.calc_rmse` (RMSE over aligned times), `data_preparation.evaluate_guess` (sum of squared errors of the *final* proportions), and `compare_to_reference` (per-family MSE, `--steps-per-day` collapsed).
  - **Unknown time mapping:** the step-to-real-time mapping is "not yet confirmed" (`--steps-per-day`). This is the same hidden coupling as `time * 2` in [RES-2](#res-2---bioreactor-scripts-medium).
  - `print_ratios.py` uses `np.load(..., allow_pickle=True)` (unnecessary; it runs arbitrary code from untrusted files) and wraps every error in a generic `RuntimeError`.
  - Outputs (`*_occupancy.png`, `*_mse_by_family.png`) are written next to the input CSV.
- **Fix:**
  - Commit `experiment.py` (or have it produce its output through `StateTracker`).
  - Add one `analysis` module that turns tracker output into an abundance table (`step, time, <species…>, occupied_cells, total_cells`) and one metric module (RMSE/MSE per species and over time), used by both studies.
  - Keep a species-name mapping file per dataset in `research/<study>/data/`.
  - Document where `dynamo_data` comes from and its licence (**D6**). Fetch it with a script rather than committing it.
  - Drop `allow_pickle`.

---

## 9. Project hygiene

<a id="hyg-1"></a>
### HYG-1 - No LICENSE (High)

Without a licence nobody else may legally use, modify or redistribute the code. Add `LICENSE` (MIT or BSD-3 are common for research code; check AGH/co-author requirements) and `license` in `pyproject.toml`. Also check that the OTU/ASV data in `research/bioreactor/data/` may be redistributed, and cite its source in the study README.

### HYG-2 - No CI, no lint configuration (High)

> **Status:** Mostly fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36): GitHub Actions runs `pytest` and `ruff check` (`E,F,I,B`), both blocking, and all ruff findings are fixed. Still open: `ruff format --check` (adopting it reformats about 30 files, so it needs its own PR).

There is no `.github/workflows`. `ruff` is a dev dependency but has no config, and it reports **374** findings. Most are `import *` (F403/F405), plus a bare `except`, `== None`, `== True`, unused imports and a redefinition.

**Fix:** add `[tool.ruff]` (start with `E,F,I,B`), fix the findings, and add a GitHub Actions job running `uv sync`, `ruff check`, `ruff format --check` and `pytest` on push and PR.

### HYG-3 - Naming & layout (Medium)

- Module files are CamelCase (`Sim.py`, `State.py`, `GutDrift.py`) and the examples use `CA_sim.py`. PEP 8 wants `snake_case` modules.
- Subpackages have no `__init__.py`, and the package root exports nothing. The intended public API is unclear, and this is the root cause of [BROKEN-1](#broken-1).
- `Sim.py` contains `CellularAutomaton`; module and class names don't match.

**Fix:** rename modules (`sim.py`, `state.py`, `rules/gut_drift.py`, …), add `__init__.py` files that re-export the public classes (`from cellular_automata import CellularAutomaton, Geometry, State`), and add a CLI entry point `ca-sim = cellular_automata.cli:main` to replace `examples/CA_sim.py`.

This is a breaking import change, so do it once, together with ARCH-1.

### HYG-4 - Logging, typing, docs (Low)

- `print` is used throughout the library (including `"RENDERER ERROR"` and `"ERROR IN MULTICALLBACK"`, which swallow exceptions in `Callback.py:25-29`, `:52-55`, `:76-80`). Use `logging`, and let real errors propagate.
- Type hints are partial; some are wrong (`state: State = None`, `callable` instead of `Callable`). Add hints and consider `mypy` or `pyright` in CI later.
- README items (**fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36)**, including a test that keeps the example config valid):
  - the controls description: W/S/A/D actually translate the camera rather than zoom or rotate
  - `G` (save GIF) is undocumented
  - the example JSON uses the wrong format ([ARCH-4](#arch-4))
- `requires-python = ">=3.12"`: `np.unstack` (`State.py:122`) requires NumPy ≥ 2.1, which is fine with the current pins. Note it.

---

## Decisions needed before fixing

These are modelling or product choices, not code questions.

| ID | Question | Options |
|---|---|---|
| **D1** | `SpeciesInteraction` semantics: should a negative coefficient be a per-neighbour kill probability (current: `|c| · n/8`) or a guaranteed kill at `|c| = 1`? Can a symbiotic spawn overwrite an occupied cell? | Keep the code and update the tests, *or* change the model and keep the tests |
| **D2** | Where is the wall? Gut `BiofilmDetachment` uses **y**; every other rule and the gut seeding use **z** | Pick one convention per study and make it a `Geometry` setting |
| **D3** | Should the two `BiofilmDetachment` rules be merged now that the keys and axis are parameters? | One rule with `floating_key=None` (just remove) and `attachment_probability=0`, *or* keep two |
| **D4** | Diffusion boundary: what should happen to particles lost in collisions? | Reflecting walls + exact conservation, explicit inflow boundary, or keep the stochastic refill (documented) |
| **D5** | Do the old `fullN` configs need to stay runnable, or are they an archive of past runs? | Archive (freeze + README), *or* migrate to `extends` + overrides |
| **D6** | Licence choice and data redistribution rights ([HYG-1](#hyg-1)), including the external `dynamo_data` reference set ([RES-4](#res-4)) | - |
| **D7** | Grid saturation in the gut model: once every cell is occupied, `SpontaneousSpawn` and the symbiotic spawn can't place anything and the community freezes ([RES-3](#res-3)). Is that intended? | Add turnover (death/decay or washout via `GutDrift`) so cells free up, allow displacement/overwrite (ties to D1), or cap density below 1 |
| **D8** | Where is the `experiment.py` that produces `ratios.csv` for the gut runs, and which species naming is canonical for the gut dataset? ([RES-4](#res-4)) | Commit it as `research/gut/scripts/`; choose one name mapping |
| **D9** | Neighbourhood `radius`: should radius *r* include every cell within *r* (filled), or only cells at exactly ±*r* (dilated, which `test_radius_scaling_3d` asserts today)? ([BUG-3](#bug-3)) | Filled (change the generators and the test), *or* dilated (document it, rename to `dilation`) |
| **D10** | Every bioreactor config caps Diffusion/Utilization at 500 of 180,000 runs per step ([BUG-5](#bug-5)). Is the model calibrated around this? | Keep the cap and document it as an effective diffusion rate, raise `dt` for diffusion, or make the cap configurable |

---

## Step-by-step plan

Each step is meant to be **one small PR** that keeps tests green. Steps within a phase can be reordered; phases mostly can't.

### Phase 0 - Safety net (do first)

1. **Licence + CI skeleton.** Add `LICENSE`, `[tool.ruff]`, and a GitHub Actions workflow running `ruff` (non-blocking at first) and `pytest`. *Done when:* CI runs on PRs. *Status: CI fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36); the licence is open (D6).*
2. **Fix the stale tests.** Fix `test_gol_correctness` and rewrite the growth test as a `Utilization` test. Mark the two `SpeciesInteraction` tests `xfail(reason="D1")` until D1 is decided. *Done when:* CI is green. *Status: fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36).*
3. **Config and example smoke tests.** Parametrised tests that `parse_json` every config (render forced off) and import every example. They will fail first, then pass after Phase 1. *Done when:* the tests exist and failures are listed. *Status: fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36).*

### Phase 1 - Make it work as advertised

4. **[BROKEN-2](#broken-2):** add `time_step` to the 3 example configs. *Status: fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36).*
5. **[BROKEN-1](#broken-1):** add `rules/__init__.py` exporting the classes. Fix the constructor calls and `renderer.run()` in the examples. Fix `monitoring_sim.py`. *Status: fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36).*
6. **[BROKEN-3](#broken-3) / [BROKEN-4](#broken-4):** explicit headless/rendered branch in `CA_sim.py`; remove `quit()` from `Renderer`; fix the callback defaults in the Parser. *Status: fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36).*
7. **README:** fix the example JSON, controls and `G` key; document the time model ([BUG-5](#bug-5)). *Phase done when:* all smoke tests pass and every example starts. *Status: fixed in [#36](https://github.com/MaruStalmach/Cellular-Automata/pull/36).*

### Phase 2 - Correctness and reproducibility

8. **[BUG-1](#bug-1):** `int32` neighbour counts + regression test. **Re-run one key bioreactor calibration** and compare. *Status: fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37); re-running a calibration is still open.*
9. **[REPRO-1](#repro-1):** `sim.seed` in the config, a `Generator` in `CellularAutomaton`, `rule.rng` in every rule, and tests switched to an injected rng ([TEST-3](#test-3)). *Done when:* two runs with the same seed produce identical tracker output (tiny grid test).
10. **[BUG-2](#bug-2), [BUG-3](#bug-3), [BUG-6](#bug-6):** shift convention, Moore radius, bioreactor detachment keys, each with a test. *Status: BUG-2 and BUG-6 fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37); BUG-3 partly fixed (needs D9).*
11. **[BUG-4](#bug-4) / [BUG-5](#bug-5):** remove the cell-wise path; abstract `apply_state`; warn when the 500 cap is hit; validate `dt` ratios. *Status: fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37).*
12. **[BUG-10](#bug-10):** tracker records only the selected keys; `record_every`; save on interrupt; symmetric paths. *Status: fixed in [#37](https://github.com/MaruStalmach/Cellular-Automata/pull/37) except `record_every`.*

### Phase 3 - Architecture for extensibility (unblocks gut work)

13. **[ARCH-1](#arch-1): registry.** `@register_rule` plus registries for neighbourhoods, trackers, callbacks and init functions; remove wildcard imports and `globals()` lookups. *Done when:* adding a rule needs only a new file, and `ruff` F403/F405 is zero.
14. **[ARCH-3](#arch-3): decouple rendering.** `build_simulation` vs `build_renderer`; lazy GL import; `render` optional extra. *Done when:* `pytest` passes in an environment without `glfw`/`pyopengl`.
15. **[ARCH-2](#arch-2): geometry wall/flow axis + parameterised keys** (after **D2**). Update `BiofilmDetachment`, `BioreactorBiofilmDetachment`, `GutDrift`, `AntibioticEffect`, `SpeciesInteraction`, `GameOfLife3D`, `random_seeds`. Then decide **D3**.
16. **[ARCH-6](#arch-6): rule interface cleanup.** Consistent parameter names, `required_keys` everywhere, `ValueError` instead of `assert`, docstrings with units. Add a rule-authoring guide in the README.
17. **[ARCH-5](#arch-5): State cleanup.** Explicit initialisers, a `ctx` argument replacing `state.sim`, a fixed `copy()`.
18. **[ARCH-4](#arch-4): config schema.** Dataclass/pydantic model, validation errors, `extends` support; migrate configs (after **D5**).

### Phase 4 - Bring in `gut-experiments`

19. **Port, don't merge** ([RES-1](#res-1)). Take only the new files from `origin/gut-experiments`:
    - `SpontaneousSpawn.py` → `src/cellular_automata/rules/`
    - `CA_neighbourhoods.py` → `examples/`
    - `experiments/scripts/*`, `data_preparation.py` → `research/gut/scripts/`
    - `mc_config.json` → `research/gut/configs/`
    - `output.json` → `research/gut/results/`

    Rewrite imports to `cellular_automata.` and keep the restructure version of every shared file. Commit `experiment.py` (**D8**). *Done when:* the branch can be deleted with nothing lost.
20. **`SpontaneousSpawn` in the library** ([RES-3](#res-3)): register it, add the injected `rng`, tests and an example gut config. Fix `CA_neighbourhoods.py` with the other examples.
21. **Gut search on the shared stack:** rewrite `data_preparation.py` → `mc_search.py` on the registry + unified config. Fix the RNG under multiprocessing with `SeedSequence.spawn`. Move `build_species_grid` into the library as an initialiser.
22. **Shared analysis + metrics** ([RES-2](#res-2---bioreactor-scripts-medium), [RES-4](#res-4)): one tracker → abundance-table function, one metrics module (RMSE/MSE per species and over time, explicit step→time mapping from the config `dt`), and per-dataset species-name mapping files. Port `check_occupancy`, `compare_to_reference` and `print_ratios` onto it, and the bioreactor `screening.py` / `bayesian_optimization.py` too. Decide **D7** before calibrating against `dynamo_data`.

### Phase 5 - Naming, performance, polish

23. **[HYG-3](#hyg-3---naming--layout-medium):** snake_case modules, public `__init__` exports, a `ca-sim` CLI entry point. One breaking release; note it in a CHANGELOG.
24. **[PERF-1](#perf-1):** vectorise `BacteriaGrowth` / `Utilization` conflict resolution; profile before and after with `tools/profiling`.
25. **[PERF-2](#perf-2---smaller-items), [BUG-7](#bug-7), [BUG-8](#bug-8)** (after **D4**): Diffusion buffers and boundary model; antibiotic maths and constants.
26. **[HYG-4](#hyg-4---logging-typing-docs-low)** and the dead-code table in [section 6](#6-duplication--dead-code-medium): logging, type hints, `mypy`, make `ruff` blocking in CI.

### Suggested branch / PR sequence

```text
main
 └─ refactor/repo-structure (PR #33)
     └─ fix/phase-0-safety-net (PR #36: Phases 0 and 1)
         └─ fix/phase-2-correctness-repro (PR #37: Phase 2 without REPRO-1)
             └─ fix/repro-seeding (REPRO-1, next)
                 └─ refactor/phase-3-registry-and-geometry
                     └─ feat/phase-4-gut-experiments
                         └─ refactor/phase-5-naming-perf
```

Gut-feature work can start **after step 15**. From then on, a new rule is one file with configurable keys and wall axis, and the explicit registry picks it up automatically.

Step 19 (porting the new `gut-experiments` files) doesn't depend on Phases 1-3 and can be done **right away**, so the branch stops drifting further from `main`. Until step 13 lands, `SpontaneousSpawn` just needs one extra import line in `Parser.py` to be usable from configs.
