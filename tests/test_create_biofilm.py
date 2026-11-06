import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.State import State
from cellular_automata.rules.CreateBiofilm import CreateBiofilm


def test_create_biofilm_unions_bacteria_presence_into_biofilm():
    geometry = Geometry((3, 3, 3), axes="xyz", periodicity="")
    rule = CreateBiofilm(
        geometry=geometry,
        time_step=1,
        biofilm_key="biofilm",
        bacteria_keys=["bact_a", "bact_b"],
    )

    state = State(
        geometry=geometry,
        random=False,
        cell_keys=["biofilm", "bact_a", "bact_b"],
    )

    state["biofilm"][1, 1, 1] = 1
    state["bact_a"][0, 0, 0] = 1
    state["bact_b"][2, 2, 2] = 1

    new_state = rule.apply_state(state)

    assert new_state["biofilm"][1, 1, 1] == 1
    assert new_state["biofilm"][0, 0, 0] == 1
    assert new_state["biofilm"][2, 2, 2] == 1
    assert int(np.sum(new_state["biofilm"])) == 3
