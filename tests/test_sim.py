import numpy as np
import pytest

from cellular_automata.Geometry import Geometry
from cellular_automata.Rule import Rule
from cellular_automata.rules.BacteriaGrowth import BacteriaGrowth
from cellular_automata.rules.GameOfLife3D import GameOfLife3D
from cellular_automata.rules.Utilization import Utilization
from cellular_automata.Sim import CellularAutomaton
from cellular_automata.State import State


def test_cellular_automaton_initialisation():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    gol = GameOfLife3D(geometry=geometry, time_step=1000)
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    assert ca.rules == [gol]
    assert ca.step_no == 0

    # checks for correctness of aggregated rules
    assert "alive" in ca.state.keys


def test_step_incrementation():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    gol = GameOfLife3D(geometry=geometry, time_step=1000)
    ca = CellularAutomaton(geometry=geometry, rules=[gol])

    assert ca.step_no == 0
    ca.step()
    assert ca.step_no == 1


def test_gol_correctness():
    size = (1, 5, 5)
    g = Geometry(size=size, axes="xyz", periodicity="")
    ca = CellularAutomaton(g, [GameOfLife3D(g, time_step=1, eb=2, eh=3, fb=3, fh=3)])

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


def _utilization_sim(key_layers):
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    rule = Utilization(
        geometry=geometry,
        time_step=1000,
        bacteria_keys=["alive"],
        substrate_key="substrate",
    )
    ca = CellularAutomaton(geometry=geometry, rules=[rule], state=None)
    ca.state = State(
        geometry, random=None, cell_keys=["alive", "substrate"], key_layers=key_layers
    )
    return ca


def _always_consume(monkeypatch):
    monkeypatch.setattr(
        np.random,
        "random",
        lambda *args, **kwargs: np.zeros(args[0], dtype=float) if args else 0.0,
    )


def test_utilization_consumes_only_one_substrate_particle_per_cell(monkeypatch):
    ca = _utilization_sim(key_layers=[1, 6])
    ca.state["alive"][1, 1, 1] = 1
    ca.state["substrate"][1, 1, 2] = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    _always_consume(monkeypatch)

    ca.step()

    assert int(np.sum(ca.state["substrate"])) == 1
    assert int(np.sum(ca.state["alive"])) == 1


def test_bacteria_growth_claims_each_target_cell_once(monkeypatch):
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    rule = BacteriaGrowth(
        geometry=geometry,
        bacteria_keys=["alive"],
        substrate_key="substrate",
        time_step=1000
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


class _CountingRule(Rule):
    def __init__(self, geometry, time_step):
        super().__init__(geometry, time_step)
        self.calls = 0

    def apply_state(self, state):
        self.calls += 1
        return state


def _geometry():
    return Geometry((2, 2, 2), axes="xyz", periodicity="")


def test_rules_run_max_dt_over_dt_times_per_step():
    g = _geometry()
    slow, fast = _CountingRule(g, time_step=10), _CountingRule(g, time_step=2)
    ca = CellularAutomaton(g, [slow, fast], state=State(g, cell_keys=["x"]))

    ca.step()

    assert (slow.calls, fast.calls) == (1, 5)


def test_warns_when_rule_runs_are_capped():
    g = _geometry()
    with pytest.warns(UserWarning, match="capped"):
        CellularAutomaton(
            g,
            [_CountingRule(g, time_step=1000), _CountingRule(g, time_step=1)],
            state=State(g, cell_keys=["x"]),
        )


def test_warns_when_time_steps_do_not_divide_evenly():
    g = _geometry()
    with pytest.warns(UserWarning, match="not a whole multiple"):
        CellularAutomaton(
            g,
            [_CountingRule(g, time_step=10), _CountingRule(g, time_step=3)],
            state=State(g, cell_keys=["x"]),
        )


def test_no_warning_for_compatible_time_steps(recwarn):
    g = _geometry()
    CellularAutomaton(
        g,
        [_CountingRule(g, time_step=10), _CountingRule(g, time_step=0.1)],
        state=State(g, cell_keys=["x"]),
    )

    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_float_time_step_ratio_is_not_floored_down():
    # 10 // 0.1 == 99.0 in floating point
    g = _geometry()
    slow, fast = _CountingRule(g, time_step=10), _CountingRule(g, time_step=0.1)
    ca = CellularAutomaton(g, [slow, fast], state=State(g, cell_keys=["x"]))

    ca.step()

    assert fast.calls == 100
