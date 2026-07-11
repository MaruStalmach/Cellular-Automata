from libs.Rules import SpeciesInteraction
from libs.Geometry import Geometry
from libs.Sim import CellularAutomaton
from libs.Rendering.Renderer import CARenderer

import numpy as np












if __name__=='__main__':
    #TODO parse args

    size = (100, 100, 100)
    axes = 'xyz'
    p = ''
    geometry = Geometry(size,axes,p)


    ic = {
        'bac1':{
            'bac2':1.0
        },
        'bac2':{
            'bac1':1.0
        }
    }
    rules = [SpeciesInteraction(geometry,ic)]

    ca_sim = CellularAutomaton(geometry=geometry, rules=rules)

    b1s = np.zeros(shape=size,dtype=np.uint8)
    b1s[0,0,:] = 1
    b2s = np.zeros(shape=size,dtype=np.uint8)
    b2s[-1,-1,:] = 1

    ca_sim.state['bac1'] = b1s
    ca_sim.state['bac2'] = b2s

    
    sim_state = {
        'running': True,
        'step_count': 0,
        'max_steps':1000
    }
    
    def update_callback(step=True):
        """Called each frame to get the latest CA state"""
        if sim_state['running'] and sim_state['step_count'] < sim_state['max_steps']:
            if step:
                ca_sim.step()
                sim_state['step_count'] += 1
            print(f"Step {sim_state['step_count']}/{sim_state['max_steps']}")
        
        # Convert CA state to numpy array for rendering
        try:
            cell_array = ca_sim.state['bac1']
            cell_array = cell_array.reshape(size)
            return cell_array
        except Exception:
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