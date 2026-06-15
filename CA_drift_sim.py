from libs2.Rules import GutDrift
from libs2.Geometry import Geometry
from libs2.Sim import CellularAutomaton
from libs.Rendering.Renderer import CARenderer

import numpy as np

if __name__ == '__main__':
    size = (10, 10, 10)
    axes = 'xyz'
    p = 'z'

    geometry = Geometry(size=size, axes=axes, periodicity=p)

    drift_rule = GutDrift(geometry=geometry, drift_speed=1, target_key='biofilm')
    ca = CellularAutomaton(geometry=geometry, rules=[drift_rule])

    ca.state['biofilm'] = 0
    

    ca.state['biofilm'][1:5, 1:5, 0:2] = 1
            
    def update_callback(stp=True):
        if stp:
            ca.step()

        return ca.state.key_grid('biofilm', as_bool=True)
    
    renderer = CARenderer(width=1200, height=800, init_state=update_callback(stp=False))

    try:
        renderer.run(update_callback=update_callback)
    except KeyboardInterrupt:
        print("Simulation stopped by user")