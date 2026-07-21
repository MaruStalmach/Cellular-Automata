from libs.Geometry import Geometry
from libs.rules.BacteriaGrowth import BacteriaGrowth
from libs.rules.GameOfLife3D import GameOfLife3D
from libs.Sim import CellularAutomaton
from libs.State import State

import numpy as np


def test_cellular_automaton_initialisation():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    gol = GameOfLife3D(geometry=geometry)
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    assert ca.rules == [gol]
    assert ca.step_no == 0

    # checks for correctness of aggregated rules
    assert "alive" in ca.state.keys


def test_step_incrementation():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    gol = GameOfLife3D(geometry=geometry)
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    assert ca.step_no == 0
    ca.step()
    assert ca.step_no == 1


def test_gol_correctness():
    size = (1, 5, 5)
    g = Geometry(size=size, axes="xyz", periodicity="")
    ca = CellularAutomaton(g, [GameOfLife3D(g, 2, 3, 3, 3)])

    def gol_state_from_array(array: np.ndarray):
        return array

    def array_from_gol_state(state_data):
        arr = state_data["alive"]
        return arr

    # blinker test
    even_state = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
        dtype=np.uint8,
    )
    odd_state = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
        dtype=np.uint8,
    )

    ca.state["alive"] = gol_state_from_array(even_state)

    # should see periodic behavior
    ca.step()
    ca_state_arr = array_from_gol_state(ca.state)
    assert np.array_equal(ca_state_arr, odd_state)

    ca.step()
    ca_state_arr = array_from_gol_state(ca.state)
    assert np.array_equal(ca_state_arr, even_state)

    # test 2
    states = [
        np.array(
            [
                [
                    [1, 0, 0, 1, 0],
                    [0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 0, 1],
                    [0, 0, 0, 1, 0],
                ]
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                ]
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                ]
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                ]
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0],
                ]
            ],
            dtype=np.uint8,
        ),
        np.zeros(shape=(1, 5, 5), dtype=np.uint8),
    ]

    ca.state["alive"] = gol_state_from_array(states[0])

    for i in range(1, len(states)):
        ca.step()
        ca_state_arr = array_from_gol_state(ca.state)
        assert np.array_equal(ca_state_arr, states[i])


def test_bacteria_growth_consumes_only_one_substrate_particle_per_cell(monkeypatch):
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    rule = BacteriaGrowth(
        geometry=geometry, bacteria_keys=["alive"], substrate_key="substrate"
    )
    ca = CellularAutomaton(geometry=geometry, rules=[rule], state=None)
    ca.state = State(
        geometry, random=None, cell_keys=["alive", "substrate"], key_layers=[1, 6]
    )

    ca.state["alive"][1, 1, 1] = 1
    ca.state["substrate"][1, 1, 2] = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)

    class MockRandom:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 3:
                return np.ones(args[0], dtype=float) if args else 1.0

            return np.zeros(args[0], dtype=float) if args else 0.0

    monkeypatch.setattr(np.random, "random", MockRandom())
    monkeypatch.setattr(np.random, "choice", lambda values: int(values[0]))

    ca.step()

    assert int(np.sum(ca.state["substrate"])) == 1
    assert int(np.sum(ca.state["alive"])) == 1


def test_bacteria_growth_claims_each_target_cell_once(monkeypatch):
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    rule = BacteriaGrowth(
        geometry=geometry,
        bacteria_keys=["alive"],
        substrate_key="substrate",
    )
    ca = CellularAutomaton(geometry=geometry, rules=[rule], state=None)
    ca.state = State(
        geometry, random=None, cell_keys=["alive", "substrate"], key_layers=[1, 1]
    )

    ca.state["alive"][...] = np.ones((3, 3, 3), dtype=np.uint8)
    ca.state["alive"][1, 1, 1] = 0

    ca.state["substrate"][...] = np.zeros((3, 3, 3), dtype=np.uint8)
    ca.state["substrate"][1, 1, 1] = 1

    monkeypatch.setattr(
        np.random,
        "random",
        lambda *args, **kwargs: np.zeros(args[0], dtype=float) if args else 0.0,
    )

    ca.step()

    assert ca.state["alive"][1, 1, 1] == 1
    assert int(np.sum(ca.state["alive"])) == 27


if __name__ == "__main__":
    test_gol_correctness()
