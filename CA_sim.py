from libs.Rules import *
from libs.Geometry import *
from libs.Cells import *
from libs.Sim import *
from libs.Rendering import CARenderer

from time import time
import numpy as np
import threading












if __name__=='__main__':
    #TODO parse args

    size = (5, 5, 5)
    axes = 'xyz'
    p = 'xy'
    geometry = Geometry(size,axes,p)

    rules = [GameOfLife3D(geometry)]

    ca_sim = CellularAutomaton(geometry=geometry, rules=rules)

    # Initialize renderer
    renderer = CARenderer(width=1200, height=800)
    
    # Store simulation state
    sim_state = {'running': True, 'step_count': 0, 'max_steps': 100}
    
    def update_callback():
        """Called each frame to get the latest CA state"""
        if sim_state['running'] and sim_state['step_count'] < sim_state['max_steps']:
            ca_sim.step()
            sim_state['step_count'] += 1
            print(f"Step {sim_state['step_count']}/{sim_state['max_steps']}")
        
        # Convert CA state to numpy array for rendering
        # Assuming the CA state is stored in ca_sim.grid or similar
        # This will need to be adjusted based on your actual data structures
        try:
            cell_array = np.array([cell['alive'] for cell in ca_sim.state.data.flatten()])
            cell_array = cell_array.reshape(size)
            return cell_array
        except:
            # Fallback if structure is different
            print('update_callback fallback triggered')
            return None
    
    # Run renderer with CA updates
    try:
        renderer.run(update_callback=update_callback)
    except KeyboardInterrupt:
        print("Simulation stopped by user")   