from libs.Rules import *
from libs.Geometry import *
from libs.Cells import *
from libs.Sim import *

from time import time
import os
from argparse import ArgumentParser




# example usage: python -m sim_no_render -x 10 -y 10 -z 10

def main():
    parser = ArgumentParser(
        
    )
    
    parser.add_argument('-x','--xsize')
    parser.add_argument('-y','--ysize')
    parser.add_argument('-z','--zsize')
    
    args = parser.parse_args()

    size = (int(args.xsize), int(args.ysize), int(args.zsize))
    axes = 'xyz'
    p = ''
    geometry = Geometry(size,axes,p)

    rules = [GameOfLife3D(geometry)]

    ca_sim = CellularAutomaton(geometry=geometry, rules=rules)

    start = time()
    while True:
        end = time()
        # print(f'step {ca_sim.step_no}, time since start = {end-start}s, avg time per step = {(end-start)/(ca_sim.step_no+1e-3)}s')
        ca_sim.step()




if __name__ == '__main__':
    main()
    
                
    
    