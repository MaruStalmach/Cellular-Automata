import json
import pytest
import numpy as np
from pathlib import Path

from libs.Geometry import Geometry
from libs.State import State
from libs.rules.SpeciesInteraction import SpeciesInteraction


@pytest.fixture
def geometry() -> Geometry:
    return Geometry(size=(5, 5), axes="xy", periodicity="")


@pytest.fixture
def base_state(geometry: Geometry) -> State:
    return State(
        geometry=geometry, random=False, cell_keys=["A", "B", "C"], dtype=np.uint8
    )


def test_load_json_config(tmp_path: Path):
    """tests if json is loaded correctly"""
    config_file = tmp_path / "interactions.json"
    mock_data = {"A": {"B": -0.5, "C": 0.2}}
    config_file.write_text(json.dumps(mock_data))

    loaded = SpeciesInteraction.load_json_config(str(config_file))
    assert loaded == mock_data

    with pytest.raises(FileNotFoundError, match="config file not found"):
        SpeciesInteraction.load_json_config("nonexistent_file.json")

    bad_json_file = tmp_path / "bad.json"
    bad_json_file.write_text("{ invalid json")
    with pytest.raises(ValueError):
        SpeciesInteraction.load_json_config(str(bad_json_file))


def test_get_interaction(geometry: Geometry):
    """tests getting interaction using get_interaction"""
    interactions = {"A": {"B": -0.5}, "B": {}}
    rule = SpeciesInteraction(geometry, interactions)

    assert rule.get_interaction("A", "B") == -0.5
    assert rule.get_interaction("A", "C") == 0.0  # missing val should default to 0

    # TODO: decide if the interactions should and can be asymmetrical
    # assert rule.get_interaction("B", "A") == 0.0


def test_destructive_interaction(geometry: Geometry, base_state: State):
    """
    interaction coeff < 0.
    A should kill Species B ONLY if adjacent
    """
    # -1.0 coeff guarantees a kill
    interactions = {"A": {"B": -1.0}, "B": {}, "C": {}}
    rule = SpeciesInteraction(geometry, interactions)

    base_state["A"][2, 2] = 1

    base_state["B"][2, 3] = 1  # B adjeecent to A
    base_state["B"][0, 0] = 1  # B not adjecent to A

    new_state = rule.apply_state(base_state)

    assert new_state["A"][2, 2] == 1
    assert new_state["B"][2, 3] == 0
    assert new_state["B"][0, 0] == 1


def test_symbiotic_interaction(geometry: Geometry, base_state: State):
    """
    interaction coeff > 0.
    A causes B spawning in the neighbourhood
    """
    interactions = {"A": {"B": 1.0}, "B": {}, "C": {}}
    rule = SpeciesInteraction(geometry, interactions)

    base_state["A"][2, 2] = 1
    base_state["B"][1, 1] = 1

    initial_b_count = np.sum(base_state["B"])
    new_state = rule.apply_state(base_state)
    final_b_count = np.sum(new_state["B"])

    assert initial_b_count < final_b_count

    # A is supposed to be overwritten by B
    assert new_state["A"][2, 2] == 1
    assert new_state["B"][2, 2] == 0

    # ensure one cell is occupied only by one cell type
    are_overlaping = (new_state["A"] == 1) & (new_state["B"] == 1)
    assert not np.any(are_overlaping)

    assert (
        new_state["B"][0, 0] == 0
    )  # TODO: consider if B can spawn somewhere (outside neighbourhood)else without A


def test_no_interaction_preserves_state(geometry: Geometry, base_state: State):
    """
    interaction coeff = 0
    everythign should remain unchanged
    """
    interactions = {"A": {"B": 0.0}, "B": {"A": 0.0}, "C": {}}
    rule = SpeciesInteraction(geometry, interactions)

    base_state["A"][2, 2] = 1
    base_state["B"][2, 3] = 1

    new_state = rule.apply_state(base_state)

    assert new_state["A"][2, 2] == 1
    assert new_state["B"][2, 3] == 1
    assert np.sum(new_state["A"]) == 1
    assert np.sum(new_state["B"]) == 1


def test_tiebreaker(geometry: Geometry, base_state: State):
    """
    tests `np.argmax`+ noise to resolve ties between species with identical scores fot an empty cell"""

    interactions = {"A": {"B": 1.0, "C": 1.0}, "B": {}, "C": {}}
    rule = SpeciesInteraction(geometry, interactions)

    base_state["A"][2, 2] = 1
    base_state["B"][1, 3] = 1
    base_state["C"][3, 3] = 1

    np.random.seed(42)
    new_state = rule.apply_state(base_state)

    b_claimed = new_state["B"][2, 3] == 1
    c_claimed = new_state["C"][2, 3] == 1

    assert b_claimed ^ c_claimed
