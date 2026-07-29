import json
from pathlib import Path

import numpy as np

from libs.Callback import *
from libs.Geometry import Geometry
### RULES ########################
from libs.rules.BacteriaGrowth import *
from libs.rules.GameOfLife3D import *
from libs.rules.GutDrift import *
from libs.rules.Diffusion import *
from libs.rules.BiofilmDetachment import *
from libs.rules.SpeciesInteraction import *
from libs.rules.BacteriaDecay import *
from libs.rules.Utilization import *

### NEIGHBORHOODS #####################
from libs.neighbourhoods.MooreNeighbourhood import *
from libs.neighbourhoods.VonNeumannNeighbourhood import *
from libs.rendering.Renderer import CARenderer
from libs.rules.BacteriaDecay import *

### RULES ########################
from libs.rules.BacteriaGrowth import *
from libs.rules.BiofilmDetachment import *
from libs.rules.Diffusion import *
from libs.rules.GameOfLife3D import *
from libs.rules.GutDrift import *
from libs.rules.SpeciesInteraction import *
from libs.rules.Utilization import *
from libs.Sim import CellularAutomaton
from libs.State import State
from libs.tracking.StateTracker import StateTracker
from libs.util.helper_functions import *


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


def parse_json(filename) -> tuple[CellularAutomaton, CARenderer]:

    config_path = Path(__file__).resolve().parent / filename

    with open(config_path, "r") as file:
        data = json.load(file)

    sim_data = data["sim"]

    ### first setup geometry
    size = sim_data['size'] # string of a tuple
    period = sim_data['periodicity']
    max_steps = sim_data['max_steps']
    neighbourhood_name = sim_data.get('neighbourhood',None)
    neighborhood_name = sim_data.get('neighborhood',None)
    n_n = neighborhood_name or neighbourhood_name
    neigh_class = globals().get(n_n,None)
    try:
        neighborhood = neigh_class()
    except:
        neighborhood = None

    geometry = Geometry(size=tuple(size), axes='xyz', periodicity=period, neighbourhood=neighborhood)


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
        random.append(sk.get("random", True))
        random_args.append(sk["random_args"])
        key_dtypes[sk["name"]] = getnestedattr(
            np, sk.get("dtype", sim_data.get("dtype", "np.uint8"))
        )
        func_name = sk.get("random_func", None)
        if func_name is None:
            random_func.append(None)
        else:
            random_func.append(getnestedattr(np, func_name))

    state = State(
        geometry=geometry,
        cell_keys=cell_keys,
        key_layers=key_layers,
        key_dtypes=key_dtypes,
        random=random,
        random_func=random_func,
        random_args=random_args,
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
 
        output_filename = tracker_data.get("save_as") or tracker_data.get("filename")
 
        tracker_class = globals().get(tracker_name)
        if tracker_class is None:
            raise RuntimeError
        try:
            tracker = tracker_class(**tracker_args)
        except Exception as e:
            raise RuntimeError from e
    else:
        print("no 'tracker' block found in config JSON -> sim.tracker will be None and no history will be recorded")


    ### create rules list
    rules_data = data["rules"]

    rules = []
    for rd in rules_data:
        args = rd.get("args", {})
        rule_class = globals()[rd["name"]]
        rule = rule_class(geometry, **args)
        rules.append(rule)

    ### finally create a CA object
    ca_sim = CellularAutomaton(geometry=geometry, rules= rules, state=state, max_steps=max_steps, tracker=tracker, output_filename=output_filename)
    
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
        callback = Callback()
    else:
        cb_args = {} or callback_data.get("args", None)
        callback = globals()[callback_data["name"]](key, ca_sim, **cb_args)

    if isinstance(key, list):
        keys_amt = len(key)
    else:
        keys_amt = 1

    renderer = CARenderer(
        callback(False), *window_size, update_callback=callback, keys_to_render=keys_amt
    )

    return ca_sim, renderer
