from libs2.Rules import *
from libs.Geometry import *
from libs2.Cells import *
from libs2.Sim import *
from libs2.State import *
from libs.Rendering.Renderer import CARenderer

from time import time
import numpy as np
import threading












if __name__=='__main__':
    #TODO parse args

    size = (500, 500, 100)
    axes = 'xyz'
    p = 'yz'
    geometry = Geometry(size,axes,p)

    rules = [GameOfLife3D(geometry)]

    ca_sim = CellularAutomaton(geometry=geometry, rules=rules)

    
    
    # Store simulation state
    sim_state = {'running': True, 'step_count': 0, 'max_steps': 500}
    
                
    
    def update_callback(step=True):
        """Called each frame to get the latest CA state"""
        if sim_state['running'] and sim_state['step_count'] < sim_state['max_steps']:
            if step:
                ca_sim.step()
                sim_state['step_count'] += 1
            print(f"Step {sim_state['step_count']}/{sim_state['max_steps']}")
        
        # Convert CA state to numpy array for rendering
        # Assuming the CA state is stored in ca_sim.grid or similar
        # This will need to be adjusted based on your actual data structures
        try:
            cell_array = ca_sim.state.data
            cell_array = cell_array.reshape(size)
            return cell_array
        except:
            # Fallback if structure is different
            print('update_callback fallback triggered')
            return None
    # Initialize renderer
    renderer = CARenderer(width=1200, height=800, init_state=update_callback(step=False))
    
    # Run renderer with CA updates
    try:
        renderer.run(update_callback=update_callback)
    except KeyboardInterrupt:
        print("Simulation stopped by user")   