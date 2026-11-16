"""The simulation core works without the OpenGL packages (ARCH-3)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cellular_automata.config import Parser

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_MODULES = ["glfw", "OpenGL", "OpenGL_accelerate", "glm"]

# setting sys.modules[name] = None makes `import name` raise ImportError
BLOCK_RENDERING = "import sys\n" + "".join(
    f"sys.modules[{name!r}] = None\n" for name in RENDER_MODULES
)


def _run_without_rendering(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BLOCK_RENDERING + code],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def _config(tmp_path: Path, render: bool) -> Path:
    config = {
        "sim": {"size": [3, 3, 3], "periodicity": "", "max_steps": 1, "render": render,
                "seed": 1},
        "state_keys": [{"name": "alive"}],
        "rules": [{"name": "GameOfLife3D", "args": {"time_step": 1}}],
    }
    if render:
        config["renderer"] = {"key": "alive"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_headless_config_builds_and_steps_without_opengl(tmp_path: Path):
    path = _config(tmp_path, render=False)

    result = _run_without_rendering(
        "from cellular_automata.config.Parser import parse_json\n"
        f"sim, renderer = parse_json({str(path)!r})\n"
        "assert renderer is None\n"
        "sim.step()\n"
        "print('ok')\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_every_example_imports_without_opengl():
    result = _run_without_rendering(
        "import importlib.util, pathlib\n"
        "for path in sorted(pathlib.Path('examples').glob('*.py')):\n"
        "    spec = importlib.util.spec_from_file_location('ex_' + path.stem, path)\n"
        "    spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "print('ok')\n"
    )

    assert result.returncode == 0, result.stderr


def test_rendered_config_without_opengl_says_how_to_install_it(tmp_path: Path):
    path = _config(tmp_path, render=True)

    result = _run_without_rendering(
        "from cellular_automata.config.Parser import parse_json\n"
        f"parse_json({str(path)!r})\n"
    )

    assert result.returncode != 0
    assert "--extra render" in result.stderr


def test_build_simulation_and_build_renderer_are_separate(tmp_path: Path, monkeypatch):
    built = {}

    class FakeRenderer:
        def __init__(self, init_state, *window_size, update_callback=None, keys_to_render=1):
            built["renderer"] = self

    monkeypatch.setattr(Parser, "renderer_class", lambda: FakeRenderer)
    data = json.loads(_config(tmp_path, render=True).read_text())

    sim = Parser.build_simulation(data)
    renderer = Parser.build_renderer(data, sim)

    assert sim.step_no == 0
    assert renderer is built["renderer"]


def test_parser_module_does_not_import_opengl():
    result = _run_without_rendering("import cellular_automata.config.Parser\n")

    assert result.returncode == 0, result.stderr


def test_every_library_module_outside_rendering_imports_without_opengl():
    result = _run_without_rendering(
        "import importlib, pkgutil, cellular_automata as ca\n"
        "for info in pkgutil.walk_packages(ca.__path__, 'cellular_automata.'):\n"
        "    if not info.name.startswith('cellular_automata.rendering'):\n"
        "        importlib.import_module(info.name)\n"
    )

    assert result.returncode == 0, result.stderr


def test_missing_renderer_is_reported_before_the_simulation_is_built(tmp_path: Path):
    path = _config(tmp_path, render=True)
    config = json.loads(path.read_text())
    config["rules"][0]["name"] = "NoSuchRule"
    path.write_text(json.dumps(config))

    result = _run_without_rendering(
        "from cellular_automata.config.Parser import parse_json\n"
        f"parse_json({str(path)!r})\n"
    )

    assert "--extra render" in result.stderr
    assert "NoSuchRule" not in result.stderr


def test_other_import_errors_are_not_reported_as_a_missing_extra(monkeypatch):
    # a broken renderer module (not a missing OpenGL package) must surface as itself
    monkeypatch.setitem(sys.modules, "cellular_automata.rendering.Renderer", None)

    with pytest.raises(ImportError) as raised:
        Parser.renderer_class()

    assert "--extra render" not in str(raised.value)


def test_renderer_class_finds_the_real_renderer():
    pytest.importorskip("glfw")
    pytest.importorskip("OpenGL.GL")
    from cellular_automata.rendering.Renderer import CARenderer

    assert Parser.renderer_class() is CARenderer
