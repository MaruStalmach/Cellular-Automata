from libs.Rules import *
from libs.Geometry import *
from libs.Sim import *
from libs.State import *
from libs.Rendering.Renderer import CARenderer
from libs.Callback import Callback
from libs.config.Parser import parse_json

from time import time
import numpy as np
import threading

import argparse










### example usage: 
#   python -m CA_sim -f draft.json

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='CA_sim', 
                                     description='''simulate microbial growth (including biofilms) via a Cellular Automaton
                                     with customizable rules''')
    
    parser.add_argument('--filename','-f',required=True)
    
    args = parser.parse_args()
    
    
    ca_sim, renderer = parse_json(args.filename)
    
    # Run renderer with CA updates
    try:
        renderer.run()
    except KeyboardInterrupt:
        print("Simulation stopped by user")   