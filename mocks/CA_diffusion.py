from libs.Rules import Diffusion
from libs.Geometry import Geometry
from libs.Sim import CellularAutomaton
from libs.Rendering.Renderer import CARenderer

import numpy as np












if __name__=='__main__':
    #TODO parse args

    size = (50, 50, 4)
    axes = 'xyz'
    p = ''
    geometry = Geometry(size,axes,p)

    rules = [Diffusion(geometry, a=6, p=0.08)]

    ca_sim = CellularAutomaton(geometry=geometry, rules=rules)
    
    init = np.zeros_like(ca_sim.state.data)
    init[0:3,0:3,0:3,0] = 1
    
    
    ca_sim.state.data = init

    
    sim_state = {
        'running': True,
        'step_count': 0,
        'max_steps':1000
    }
    
    def update_callback(step=True):
        """Called each frame to get the latest CA state"""
        if sim_state['running'] and sim_state['step_count'] < sim_state['max_steps']:
            if step:
                for i in range(100):
                    ca_sim.step()
                    sim_state['step_count'] += 1
            print(f"Step {sim_state['step_count']}/{sim_state['max_steps']}")
        # Convert CA state to numpy array for rendering
        try:
            cell_array = ca_sim.state.data
            cell_array = cell_array.reshape(size+(6,))
            return np.maximum.reduce(cell_array, axis=-1)
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