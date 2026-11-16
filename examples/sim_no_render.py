"""Headless 3D Game of Life, used for timing and memory profiling.

example usage:
    python examples/sim_no_render.py -x 50 -y 50 -z 50 -s 100
"""

import time
from argparse import ArgumentParser

from cellular_automata.Geometry import Geometry
from cellular_automata.rules import GameOfLife3D
from cellular_automata.Sim import CellularAutomaton


def build_sim(size: tuple[int, int, int], max_steps: int = 100) -> CellularAutomaton:
    geometry = Geometry(size, "xyz", "")
    rules = [GameOfLife3D(geometry, time_step=1, eb=2, eh=3, fb=3, fh=3)]
    return CellularAutomaton(geometry=geometry, rules=rules, max_steps=max_steps)


def main():
    parser = ArgumentParser()
    parser.add_argument("-x", "--xsize", type=int, required=True)
    parser.add_argument("-y", "--ysize", type=int, required=True)
    parser.add_argument("-z", "--zsize", type=int, required=True)
    parser.add_argument("-s", "--max_steps", type=int, default=100)
    args = parser.parse_args()

    ca_sim = build_sim((args.xsize, args.ysize, args.zsize), max_steps=args.max_steps)

    start = time.time()
    while ca_sim.step() == 0:
        pass
    print(f"time={time.time() - start}")


if __name__ == "__main__":
    main()
