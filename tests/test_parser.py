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
    monkeypatch.setattr(Parser, "CARenderer", FakeRenderer)


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
