import json
from libs.Geometry import *
from libs.Rendering.Renderer import *
from libs.Rules import *
from libs.State import *
from libs.State import *
from libs.Sim import *

def parse_json(filename) -> tuple[CellularAutomaton, CARenderer]:
    with open(f'libs/config/{filename}', 'r') as file:
        data=json.load(file)
        
    sim_data = data['sim']


    ### first setup geometry
    size = sim_data['size'] # string of a tuple
    period = sim_data['periodicity']

    geometry = None

    exec(f"geometry = Geometry(size={size}, axes='xyz', periodicity='{period}')")

    ### next create State obj

    state_keys = data['state_keys']


    cell_keys = []
    key_dtypes = {}
    key_layers = []
    random = []
    random_func = []
    random_args = []

    for sk in state_keys:
        cell_keys.append(sk['name'])
        key_layers.append(sk.get('layers', 1))
        random.append(sk.get('random', True))
        random_args.append(sk['random_args'])
        exec(f"key_dtypes['{sk['name']}']={sk.get('dtype',sim_data.get('dtype', np.uint8))}")
        exec(f"random_func.append({sk.get('random_func',None)})")

    state = State(geometry=geometry, cell_keys=cell_keys, key_layers=key_layers, key_dtypes=key_dtypes, random=random, random_func= random_func, random_args= random_args)

    ### create rules list
    rules_data = data['rules']

    rules = []
    for rd in rules_data:
        args = rd.get('args',{})
        rule = None
        exec(f"rule={rd['name']}(geometry, **args)")
        rules.append(rule)

    ### finally create a CA object
    ca_sim = CellularAutomaton(geometry=geometry, rules= rules, state=state)
    
    ### if no rendering then its done
    if not sim_data['render']:
        return ca_sim, None
    
    
    ### next setup renderer
    renderer = CARenderer()
    
    
    
    