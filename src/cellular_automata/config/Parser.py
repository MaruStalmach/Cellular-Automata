import inspect
import json
import logging
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import cellular_automata.neighbourhoods  # noqa: F401  (registers the neighbourhoods)
import cellular_automata.rules  # noqa: F401  (registers the rules)
import cellular_automata.tracking.StateTracker  # noqa: F401  (registers the tracker)
import cellular_automata.util.helper_functions  # noqa: F401  (registers the initialisers)
from cellular_automata.Callback import Callback
from cellular_automata.Geometry import Geometry
from cellular_automata.registry import (
    CALLBACKS,
    INITIALISERS,
    NEIGHBOURHOODS,
    RULES,
    TRACKERS,
)
from cellular_automata.Sim import CellularAutomaton, resolve_seed, state_rng
from cellular_automata.State import State

if TYPE_CHECKING:
    from cellular_automata.rendering.Renderer import CARenderer

logger = logging.getLogger(__name__)

RENDER_PACKAGES = {"glfw", "OpenGL", "OpenGL_accelerate", "glm"}
RENDER_INSTALL_HINT = (
    "Rendering needs the optional OpenGL packages: run `uv sync --extra render` "
    "(or `pip install 'cellular-automata[render]'`), or set sim.render to false"
)


def parse_dtype(name: str) -> np.dtype:
    """Turn a config dtype such as "np.uint8" or "float32" into a numeric numpy dtype."""
    if not isinstance(name, str):
        raise ValueError(f"Unknown dtype {name!r}")
    try:
        dtype = np.dtype(name.removeprefix("np."))
    except TypeError as error:
        raise ValueError(f"Unknown dtype '{name}'") from error
    if not (dtype == np.bool_ or np.issubdtype(dtype, np.number)):
        raise ValueError(f"Unknown dtype '{name}': state arrays must be numeric or bool")
    return dtype


def resolve_random_func(name: str, rng: np.random.Generator) -> Callable:
    """Look up a state `random_func` and make it draw from `rng`.

    `np.random.<method>` (or `random.<method>`) names map to that method of
    `rng`; other names are registered initialisers, and those that take an
    `rng` argument (e.g. `random_seeds`) get it bound.
    """
    if name.startswith(("np.random.", "random.")):
        method = name.split("random.", 1)[1]
        if not hasattr(np.random.Generator, method):
            raise ValueError(
                f"random_func '{name}' is not a method of np.random.Generator, so it "
                "can't be seeded; use one that is (e.g. np.random.integers instead of "
                "np.random.randint)"
            )
        return getattr(rng, method)
    func = INITIALISERS.get(name)
    try:
        takes_rng = "rng" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        takes_rng = False
    if takes_rng:
        return partial(func, rng=rng)
    return func


def parse_json(filename) -> tuple[CellularAutomaton, "CARenderer | None"]:
    """Build a simulation, and its renderer if `sim.render` is true, from a JSON config.

    `filename` is a path to the config file, absolute or relative to the
    current working directory (e.g. `configs/examples/demo.json`). Headless
    configs never import OpenGL.
    """
    config_path = Path(filename)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Pass a path relative to the "
            "current directory, e.g. configs/examples/demo.json"
        )

    with open(config_path, "r") as file:
        data = json.load(file)

    if not data["sim"].get("render", False):
        return build_simulation(data), None
    # check for the renderer first, so a missing extra doesn't cost a full build
    renderer = renderer_class()
    ca_sim = build_simulation(data)
    return ca_sim, build_renderer(data, ca_sim, renderer)


def build_simulation(data: dict) -> CellularAutomaton:
    """Build the simulation from a loaded config. Needs no OpenGL."""
    sim_data = data["sim"]

    ### first setup geometry
    size = tuple(sim_data["size"])
    if not 1 <= len(size) <= 3:
        raise ValueError(f"sim.size must have 1 to 3 dimensions, got {list(size)}")
    period = sim_data["periodicity"]
    max_steps = sim_data["max_steps"]
    seed = resolve_seed(sim_data.get("seed"))
    if sim_data.get("seed") is None:
        warnings.warn(
            f'no sim.seed in config; using seed {seed} (add "seed": {seed} to repeat this run)',
            stacklevel=2,
        )
    init_rng = state_rng(seed)
    neighbourhood_name = sim_data.get("neighbourhood", None)
    neighborhood_name = sim_data.get("neighborhood", None)
    n_n = neighborhood_name or neighbourhood_name
    if n_n is None:
        neighborhood = None  # Geometry defaults to Moore
    else:
        neighborhood = NEIGHBOURHOODS.get(n_n)()

    geometry = Geometry(
        size=size, axes="xyz"[: len(size)], periodicity=period, neighbourhood=neighborhood
    )

    ### next create State obj

    state_keys = data["state_keys"]

    cell_keys: list[str] = []
    key_dtypes: dict[str, np.dtype | type] = {}
    key_layers: list[int] = []
    random_func: list[Callable | None] = []
    random_args: list[dict | None] = []

    for sk in state_keys:
        cell_keys.append(sk["name"])
        key_layers.append(sk.get("layers", 1))
        random_args.append(sk.get("random_args", None))
        key_dtypes[sk["name"]] = parse_dtype(
            sk.get("dtype", sim_data.get("dtype", "np.uint8"))
        )
        func_name = sk.get("random_func", None)
        is_random = sk.get("random", False)
        if not isinstance(is_random, bool):
            raise ValueError(
                f"state key '{sk['name']}': \"random\" must be true or false, got {is_random!r}"
            )
        if is_random != (func_name is not None):
            raise ValueError(
                f"state key '{sk['name']}': \"random\": true and \"random_func\" go "
                "together; set both to fill the key at random, or neither to start at 0"
            )
        if func_name is None:
            random_func.append(None)
        else:
            random_func.append(resolve_random_func(func_name, init_rng))

    state = State(
        geometry=geometry,
        cell_keys=cell_keys,
        key_layers=key_layers,
        key_dtypes=key_dtypes,
        random=any(func is not None for func in random_func),
        random_func=random_func,
        random_args=random_args,
        rng=init_rng,
    )

    ### parse tracker data from json and pass config
    tracker_data = data.get("tracker")
    tracker = None
    output_filename = None

    if tracker_data:
        tracker_name = tracker_data.get("name") or tracker_data.get("type")
        tracker_args = tracker_data.get("args", {})

        if "keys" not in tracker_args:
            tracker_args["keys"] = list(state.keys)

        output_filename = tracker_data.get("save_as", None) or tracker_data.get("filename", None)

        tracker = TRACKERS.get(tracker_name)(**tracker_args)
    else:
        logger.info("no 'tracker' block in the config, so no history will be recorded")

    ### create rules list
    rules_data = data["rules"]

    rules = []
    for rd in rules_data:
        args = rd.get("args", {})
        rule_class = RULES.get(rd["name"])
        rule = rule_class(geometry, **args)
        rules.append(rule)

    ### finally create a CA object
    ca_sim = CellularAutomaton(
        geometry=geometry,
        rules=rules,
        state=state,
        max_steps=max_steps,
        tracker=tracker,
        output_filename=output_filename,
        seed=seed,
    )
    return ca_sim


def renderer_class() -> type["CARenderer"]:
    """Import the OpenGL renderer, which is an optional dependency.

    Only a missing OpenGL package gets the install hint; any other import
    error (e.g. libGL not found) propagates unchanged.
    """
    try:
        from cellular_automata.rendering.Renderer import CARenderer
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] in RENDER_PACKAGES:
            raise ModuleNotFoundError(RENDER_INSTALL_HINT, name=error.name) from error
        raise
    return CARenderer


def build_renderer(
    data: dict, ca_sim: CellularAutomaton, renderer: type["CARenderer"] | None = None
) -> "CARenderer":
    """Build the renderer described by the config's `renderer` block for `ca_sim`."""
    renderer = renderer or renderer_class()
    renderer_data = data["renderer"]
    # TODO:multiple key rendering, will have to adjust renderer code also
    key = renderer_data.get("key", ca_sim.state.keys[0])
    window_size = tuple(renderer_data.get("window_size", ()))
    if len(window_size) not in (0, 2):
        raise ValueError(f"renderer window_size must be [width, height], got {list(window_size)}")
    callback_data = renderer_data.get("callback", None)

    if callback_data is None:
        callback = Callback(key, ca_sim)
    else:
        cb_args = callback_data.get("args") or {}
        callback = CALLBACKS.get(callback_data["name"])(key, ca_sim, **cb_args)

    if isinstance(key, list):
        keys_amt = len(key)
    else:
        keys_amt = 1

    if window_size:
        width, height = window_size
        return renderer(
            callback(False), width, height, update_callback=callback, keys_to_render=keys_amt
        )
    return renderer(callback(False), update_callback=callback, keys_to_render=keys_amt)
