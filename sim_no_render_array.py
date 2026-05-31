from libs2.Rules import GameOfLife3D
from libs.Geometry import Geometry
from libs2.Sim import CellularAutomaton

from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('-x', '--xsize')
    parser.add_argument('-y', '--ysize')
    parser.add_argument('-z', '--zsize')
    parser.add_argument('-s', '--max_steps', default=None)

    args = parser.parse_args()

    size = (int(args.xsize), int(args.ysize), int(args.zsize))
    geometry = Geometry(size, 'xyz', '')
    rules = [GameOfLife3D(geometry)]
    ca_sim = CellularAutomaton(geometry=geometry, rules=rules)

    if args.max_steps:
        max_steps = int(args.max_steps)
    else:
        max_steps = 0

    while True:
        ca_sim.step()
        if max_steps > 0 and ca_sim.step_no > max_steps:
            break


if __name__ == '__main__':
    main()
