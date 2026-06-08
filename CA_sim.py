from libs.Rules import *
from libs.Geometry import *
from libs.Cells import *
from libs.Sim import *
from libs.Rendering.Renderer import CARenderer

from time import time
import numpy as np
import threading












if __name__=='__main__':
    #TODO parse args

    size = (50,50,50)
    axes = 'xyz'
    p = ''
    geometry = Geometry(size,axes,p)

    rules = [GameOfLife3D(geometry)]

    ca_sim = CellularAutomaton(geometry=geometry, rules=rules)

    
    
    
                
    
    def update_callback(stp=True):
        if stp:
            ca_sim.step()
        cell_array = np.array([cell['alive'] for cell in ca_sim.state.data.flatten()])
        cell_array = cell_array.reshape(size)
        return cell_array
    
    # Initialize renderer
    renderer = CARenderer(width=1200, height=800, init_state=update_callback(stp=False))
    
    # Run renderer with CA updates
    try:
        renderer.run(update_callback=update_callback)
    except KeyboardInterrupt:
        print("Simulation stopped by user")   