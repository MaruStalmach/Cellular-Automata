"""Run a simulation from a JSON config, rendered or headless depending on sim.render.

example usage:
    python examples/CA_sim.py -f configs/examples/draft.json
"""

import argparse
from time import time

from cellular_automata.config.Parser import parse_json


def run_headless(ca_sim) -> None:
    while ca_sim.step() == 0:
        print(
            f"sim at step {ca_sim.step_no}/{ca_sim.max_steps}, "
            f"time elapsed={time() - ca_sim.start_time:.2f}s"
        )


def main():
    parser = argparse.ArgumentParser(
        prog="CA_sim",
        description="simulate microbial growth (including biofilms) via a Cellular "
        "Automaton with customizable rules",
    )
    parser.add_argument("--filename", "-f", required=True)
    args = parser.parse_args()

    ca_sim, renderer = parse_json(args.filename)

    try:
        if renderer is None:
            run_headless(ca_sim)
        else:
            renderer.run()
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        # keeps the history recorded so far if the run stopped before max_steps
        ca_sim.save_tracker()


if __name__ == "__main__":
    main()
