"""Parser handling of the renderer block (the OpenGL renderer itself is stubbed)."""

import json
from pathlib import Path

import pytest

from cellular_automata import Callback as callback_module
from cellular_automata.config import Parser


class FakeRenderer:
    def __init__(self, init_state, *window_size, update_callback=None, keys_to_render=1):
        self.init_state = init_state
        self.window_size = window_size
        self.update_callback = update_callback
        self.keys_to_render = keys_to_render


@pytest.fixture(autouse=True)
def no_opengl(monkeypatch):
    monkeypatch.setattr(Parser, "renderer_class", lambda: FakeRenderer)


def _write_config(tmp_path: Path, renderer_block: dict) -> Path:
    config = {
        "sim": {"size": [3, 3, 3], "periodicity": "", "max_steps": 1, "render": True},
        "state_keys": [{"name": "alive"}],
        "rules": [{"name": "GameOfLife3D", "args": {"time_step": 1}}],
        "renderer": renderer_block,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_renderer_without_callback_uses_default_callback(tmp_path: Path):
    ca_sim, renderer = Parser.parse_json(_write_config(tmp_path, {"key": "alive"}))

    assert type(renderer.update_callback) is callback_module.Callback
    assert renderer.update_callback.key == "alive"
    assert renderer.update_callback.sim is ca_sim


def test_renderer_callback_without_args(tmp_path: Path):
    config = _write_config(tmp_path, {"key": "alive", "callback": {"name": "Callback"}})

    _, renderer = Parser.parse_json(config)

    assert renderer.update_callback.stride == 1


def test_renderer_callback_args_are_passed(tmp_path: Path):
    config = _write_config(
        tmp_path,
        {"key": "alive", "callback": {"name": "Callback", "args": {"step_stride": 5}}},
    )

    _, renderer = Parser.parse_json(config)

    assert renderer.update_callback.stride == 5


def test_render_false_returns_no_renderer(tmp_path: Path):
    path = _write_config(tmp_path, {})
    data = json.loads(path.read_text())
    data["sim"]["render"] = False
    path.write_text(json.dumps(data))

    _, renderer = Parser.parse_json(path)

    assert renderer is None


def _headless_config(tmp_path: Path, **sim_overrides) -> Path:
    path = _write_config(tmp_path, {})
    data = json.loads(path.read_text())
    data["sim"].update(render=False, **sim_overrides)
    path.write_text(json.dumps(data))
    return path


def test_known_neighbourhood_is_used(tmp_path: Path):
    path = _headless_config(tmp_path, neighbourhood="VonNeumannNeighbourhood")

    ca_sim, _ = Parser.parse_json(path)

    assert len(ca_sim.geometry._offsets) == 6


def test_unknown_neighbourhood_raises(tmp_path: Path):
    path = _headless_config(tmp_path, neighbourhood="MoorNeighbourhood")

    with pytest.raises(ValueError, match="Unknown neighbourhood 'MoorNeighbourhood'"):
        Parser.parse_json(path)


def _config_data(tmp_path: Path, mutate) -> Path:
    path = _headless_config(tmp_path, seed=1)
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))
    return path


def test_render_defaults_to_false(tmp_path: Path):
    path = _config_data(tmp_path, lambda data: data["sim"].pop("render"))

    _, renderer = Parser.parse_json(path)

    assert renderer is None


@pytest.mark.parametrize(
    "state_key",
    [
        {"name": "alive", "random": True},
        {"name": "alive", "random_func": "np.random.integers", "random_args": {"low": 0, "high": 2}},
    ],
    ids=["random-without-func", "func-without-random"],
)
def test_random_and_random_func_must_come_together(tmp_path: Path, state_key):
    path = _config_data(tmp_path, lambda data: data.update(state_keys=[state_key]))

    with pytest.raises(ValueError, match="state key 'alive'"):
        Parser.parse_json(path)


def test_random_must_be_a_boolean(tmp_path: Path):
    path = _config_data(
        tmp_path, lambda data: data.update(state_keys=[{"name": "alive", "random": "false"}])
    )

    with pytest.raises(ValueError, match="must be true or false"):
        Parser.parse_json(path)


def test_random_key_is_filled_and_plain_key_starts_at_zero(tmp_path: Path):
    def mutate(data):
        data["state_keys"] = [
            {"name": "alive", "random": True, "random_func": "np.random.integers",
             "random_args": {"low": 1, "high": 2}},
            {"name": "other"},
        ]

    ca_sim, _ = Parser.parse_json(_config_data(tmp_path, mutate))

    assert ca_sim.state["alive"].min() == 1
    assert ca_sim.state["other"].max() == 0


def test_two_dimensional_config_builds_and_steps(tmp_path: Path):
    def mutate(data):
        data["sim"].update(size=[5, 5], periodicity="xy")

    ca_sim, _ = Parser.parse_json(_config_data(tmp_path, mutate))
    ca_sim.step()

    assert ca_sim.geometry.axes == "xy"
    assert ca_sim.state["alive"].shape == (5, 5)


def test_more_than_three_dimensions_raises(tmp_path: Path):
    path = _config_data(tmp_path, lambda data: data["sim"].update(size=[2, 2, 2, 2]))

    with pytest.raises(ValueError, match="1 to 3 dimensions"):
        Parser.parse_json(path)
