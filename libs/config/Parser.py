import json
from libs.Geometry import Geometry
from libs.Rendering.Renderer import CARenderer
from libs.State import State
from libs.Sim import CellularAutomaton
from libs.Callback import *
from libs.rules.BacteriaGrowth import *
from libs.rules.GameOfLife3D import *
from libs.rules.GutDrift import *
from libs.rules.Diffusion import *
from libs.rules.BiofilmDetachment import *
from libs.rules.SpeciesInteraction import *
from libs.util.helper_functions import *
import numpy as np


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
    with open(f"libs/config/{filename}", "r") as file:
        data = json.load(file)

    sim_data = data["sim"]

    ### first setup geometry
    size = sim_data['size'] # string of a tuple
    period = sim_data['periodicity']
    max_steps = sim_data['max_steps']

    geometry = Geometry(size=tuple(size), axes='xyz', periodicity=period)


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

    ### create rules list
    rules_data = data["rules"]

    rules = []
    for rd in rules_data:
        args = rd.get("args", {})
        rule_class = globals()[rd["name"]]
        rule = rule_class(geometry, **args)
        rules.append(rule)

    ### finally create a CA object
    ca_sim = CellularAutomaton(geometry=geometry, rules= rules, state=state, max_steps=max_steps)
    
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
