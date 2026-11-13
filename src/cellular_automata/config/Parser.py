import inspect
import json
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

from cellular_automata.Callback import *
from cellular_automata.Geometry import Geometry
from cellular_automata.neighbourhoods.MooreNeighbourhood import *
from cellular_automata.neighbourhoods.VonNeumannNeighbourhood import *
from cellular_automata.rendering.Renderer import CARenderer
from cellular_automata.rules.AntibioticEffect import *
from cellular_automata.rules.BacteriaDecay import *
from cellular_automata.rules.BacteriaGrowth import *
from cellular_automata.rules.BiofilmDetachment import *
from cellular_automata.rules.BioreactorBiofilmDetachment import *
from cellular_automata.rules.CreateBiofilm import *
from cellular_automata.rules.Diffusion import *
from cellular_automata.rules.GameOfLife3D import *
from cellular_automata.rules.GutDrift import *
from cellular_automata.rules.Inoculation import *
from cellular_automata.rules.SpeciesInteraction import *
from cellular_automata.rules.Utilization import *
from cellular_automata.Sim import CellularAutomaton, resolve_seed, state_rng
from cellular_automata.State import State
from cellular_automata.tracking.StateTracker import *
from cellular_automata.util.helper_functions import *

NEIGHBOURHOODS = {
    "MooreNeighbourhood": MooreNeighbourhood,
    "VonNeumannNeighbourhood": VonNeumannNeighbourhood,
}


def getnestedattr(module, name: str) -> object:
    names = name.split(".")
    if len(names) == 1:
        return globals().get(name)
    obj = module
    while True:
        n = names.pop(0)
        if globals().get(n, None) == module:
            continue
        obj = getattr(obj, n)
        if not names:
            return obj


def resolve_random_func(name: str, rng: np.random.Generator) -> Callable:
    """Look up a state `random_func` and make it draw from `rng`.

    `np.random.<method>` (or `random.<method>`) names map to that method of
    `rng`; helper functions that take an `rng` argument (e.g. `random_seeds`)
    get it bound.
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
    func = getnestedattr(np, name)
    if func is None:
        raise ValueError(f"Unknown random_func '{name}'")
    try:
        takes_rng = "rng" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        takes_rng = False
    if takes_rng:
        return partial(func, rng=rng)
    return func


def parse_json(filename) -> tuple[CellularAutomaton, CARenderer]:
    """Build a simulation from a JSON config.

    `filename` is a path to the config file, absolute or relative to the
    current working directory (e.g. `configs/examples/demo.json`).
    """
    config_path = Path(filename)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Pass a path relative to the "
            "current directory, e.g. configs/examples/demo.json"
        )

    with open(config_path, "r") as file:
        data = json.load(file)

    sim_data = data["sim"]

    ### first setup geometry
    size = sim_data["size"]  # string of a tuple
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
    elif n_n in NEIGHBOURHOODS:
        neighborhood = NEIGHBOURHOODS[n_n]()
    else:
        raise ValueError(
            f"Unknown neighbourhood '{n_n}'. Known: {sorted(NEIGHBOURHOODS)}"
        )

    geometry = Geometry(
        size=tuple(size), axes="xyz", periodicity=period, neighbourhood=neighborhood
    )

    ### next create State obj

    state_keys = data["state_keys"]

    cell_keys = []
    key_dtypes = {}
    key_layers = []
    random = []
    random_func = []
    random_args = []

    for sk in state_keys:
        cell_keys.append(sk["name"])
        key_layers.append(sk.get("layers", 1))
        random.append(sk.get("random", False))
        random_args.append(sk.get("random_args",None))
        key_dtypes[sk["name"]] = getnestedattr(
            np, sk.get("dtype", sim_data.get("dtype", "np.uint8"))
        )
        func_name = sk.get("random_func", None)
        if func_name is None:
            random_func.append(None)
        else:
            random_func.append(resolve_random_func(func_name, init_rng))

    state = State(
        geometry=geometry,
        cell_keys=cell_keys,
        key_layers=key_layers,
        key_dtypes=key_dtypes,
        random=random,
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

        tracker_class = globals().get(tracker_name)
        if tracker_class is None:
            raise RuntimeError
        try:
            tracker = tracker_class(**tracker_args)
        except Exception as e:
            raise RuntimeError from e
    else:
        print(
            "no 'tracker' block found in config JSON -> sim.tracker will be None and no history will be recorded"
        )

    ### create rules list
    rules_data = data["rules"]

    rules = []
    for rd in rules_data:
        args = rd.get("args", {})
        rule_class = globals()[rd["name"]]
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

    ### if no rendering then its done
    if not sim_data["render"]:
        return ca_sim, None

    ### next setup renderer

    ### get update_callback
    renderer_data = data["renderer"]
    # TODO:multiple key rendering, will have to adjust renderer code also
    key = renderer_data.get("key", ca_sim.state.keys[0])
    window_size = renderer_data.get("window_size", ())
    callback_data = renderer_data.get("callback", None)

    if callback_data is None:
        callback = Callback(key, ca_sim)
    else:
        cb_args = callback_data.get("args") or {}
        callback = globals()[callback_data["name"]](key, ca_sim, **cb_args)

    if isinstance(key, list):
        keys_amt = len(key)
    else:
        keys_amt = 1

    renderer = CARenderer(
        callback(False), *window_size, update_callback=callback, keys_to_render=keys_amt
    )

    return ca_sim, renderer
