from libs.Rules import *
from libs.Geometry import *
from libs.Sim import *
from libs.State import *
from libs.Rendering.Renderer import CARenderer
from libs.Callback import Callback

from time import time
import numpy as np
import threading












if __name__=='__main__':
    #TODO parse args

    size = (100, 100, 100)
    axes = 'xyz'
    p = ''
    geometry = Geometry(size,axes,p)

    rules = [GameOfLife3D(geometry)]

    ca_sim = CellularAutomaton(geometry=geometry, rules=rules)

    
    sim_state = {
        'running': True,
        'step_count': 0,
        'max_steps':1000
    }
        
    ub = Callback('alive',ca_sim,10)
    # Initialize renderer
    renderer = CARenderer(width=1200, height=800, init_state=ub(step=False), update_callback=ub)
    
    # Run renderer with CA updates
    try:
        renderer.run()
    except KeyboardInterrupt:
        print("Simulation stopped by user")   