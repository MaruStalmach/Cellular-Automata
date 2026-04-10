from libs.Rules import *
from libs.Geometry import *
from libs.Cells import *
from libs.Sim import *

from time import time












if __name__=='__main__':
    #TODO parse args

    size = (10,10,10)
    axes = 'xyz'
    p = 'xy'
    geometry = Geometry(size,axes,p)

    rules = [GameOfLife3D(geometry)]

    ca_sim = CellularAutomaton(geometry=geometry, rules=rules)

    start = time()
    for steps in range(1000):
        print(f'step {ca_sim.step_no}/1000')
        ca_sim.step()
    end = time()
    print(f'{end-start}s elapsed')   