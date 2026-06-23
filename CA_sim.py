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












if __name__=='__main__':
    
    ca_sim, renderer = parse_json('draft.json')
    
    # Run renderer with CA updates
    try:
        renderer.run()
    except KeyboardInterrupt:
        print("Simulation stopped by user")   