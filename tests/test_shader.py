"""Shader compile and link failures raise ShaderError (ARCH-3).

OpenGL is replaced by a fake module, so this runs without a GPU or the render extra.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

SHADER_MODULE = "cellular_automata.rendering.Shader"


class FakeGL(types.ModuleType):
    """Just enough of OpenGL.GL for Shader, with switchable failures."""

    GL_COMPILE_STATUS, GL_LINK_STATUS, GL_FALSE, GL_TRUE = 1, 2, 0, 1
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER = 10, 11

    def __init__(self):
        super().__init__("OpenGL.GL")
        self.failing_stage = None
        self.link_fails = False
        self.log = b"0:3: syntax error"
        self.deleted = []
        self.deleted_programs = []
        self.attached = []
        self._stages = {}

    def glCreateProgram(self):
        return 100

    def glCreateShader(self, stage):
        handle = len(self._stages) + 1
        self._stages[handle] = stage
        return handle

    def glShaderSource(self, shader, source):
        pass

    def glCompileShader(self, shader):
        pass

    def glGetShaderiv(self, shader, name):
        failed = self._stages[shader] == self.failing_stage
        return self.GL_FALSE if failed else self.GL_TRUE

    def glGetShaderInfoLog(self, shader):
        return self.log  # PyOpenGL gives bytes, or "" (a str) for an empty log

    def glAttachShader(self, program, shader):
        self.attached.append(shader)

    def glDeleteShader(self, shader):
        self.deleted.append(shader)

    def glDeleteProgram(self, program):
        self.deleted_programs.append(program)

    def glLinkProgram(self, program):
        pass

    def glGetProgramiv(self, program, name):
        return self.GL_FALSE if self.link_fails else self.GL_TRUE

    def glGetProgramInfoLog(self, program):
        return b"undefined varying"

    def glUseProgram(self, program):
        pass


RENDERING_MODULES = ("Renderer", "Camera", "Shader")


@pytest.fixture
def fresh_rendering_modules():
    """Let a test import the rendering modules against fakes, then put back what was there.

    monkeypatch can't do this: with raising=False it records nothing for a module
    that wasn't imported yet, so the fake-backed one would stay behind.
    """
    import cellular_automata.rendering as rendering

    full = {name: f"cellular_automata.rendering.{name}" for name in RENDERING_MODULES}
    saved_modules = {name: sys.modules.pop(full[name], None) for name in RENDERING_MODULES}
    saved_attrs = {name: vars(rendering).pop(name, None) for name in RENDERING_MODULES}
    yield
    for name in RENDERING_MODULES:
        sys.modules.pop(full[name], None)
        vars(rendering).pop(name, None)
        if saved_modules[name] is not None:
            sys.modules[full[name]] = saved_modules[name]
        if saved_attrs[name] is not None:
            setattr(rendering, name, saved_attrs[name])


@pytest.fixture
def gl(monkeypatch, fresh_rendering_modules):
    fake = FakeGL()
    opengl = types.ModuleType("OpenGL")
    opengl.GL = fake
    monkeypatch.setitem(sys.modules, "OpenGL", opengl)
    monkeypatch.setitem(sys.modules, "OpenGL.GL", fake)
    return fake


@pytest.fixture
def shader_module(gl):
    return importlib.import_module(SHADER_MODULE)


@pytest.fixture
def source(tmp_path: Path) -> str:
    path = tmp_path / "shader.glsl"
    path.write_text("void main() {}")
    return str(path)


def test_working_shaders_compile_attach_and_link(gl, shader_module, source):
    shader = shader_module.Shader()

    shader.add_vertex(source)
    shader.add_fragment(source)
    shader.link_program()

    assert gl.attached == [1, 2]


@pytest.mark.parametrize(
    "stage, add, label",
    [
        (FakeGL.GL_VERTEX_SHADER, "add_vertex", "Vertex"),
        (FakeGL.GL_FRAGMENT_SHADER, "add_fragment", "Fragment"),
    ],
)
def test_compile_failure_raises_with_stage_file_and_log(
    gl, shader_module, source, stage, add, label
):
    # the fragment error used to be printed as "Vertex Shader Error" and ignored
    gl.failing_stage = stage
    shader = shader_module.Shader()

    with pytest.raises(shader_module.ShaderError) as raised:
        getattr(shader, add)(source)

    message = str(raised.value)
    assert message.startswith(f"{label} shader {source} failed to compile")
    assert "0:3: syntax error" in message
    assert gl.attached == []
    assert gl.deleted == [1]


def test_link_failure_raises_with_log(gl, shader_module, source):
    gl.link_fails = True
    shader = shader_module.Shader()
    shader.add_vertex(source)

    with pytest.raises(shader_module.ShaderError, match="failed to link:\nundefined varying"):
        shader.link_program()
    assert gl.deleted_programs == [100]


def test_empty_compile_log_still_raises(gl, shader_module, source):
    gl.failing_stage = FakeGL.GL_VERTEX_SHADER
    gl.log = ""

    with pytest.raises(shader_module.ShaderError, match=f"Vertex shader {source} failed"):
        shader_module.Shader().add_vertex(source)


def test_unreadable_file_raises_shader_error(gl, shader_module, tmp_path):
    missing = tmp_path / "missing.frag"

    with pytest.raises(shader_module.ShaderError, match="Fragment shader .* could not be read"):
        shader_module.Shader().add_fragment(str(missing))


def test_build_shader_uses_the_packaged_shader_files(gl, shader_module):
    shader = shader_module.build_shader("solid.vert", "solid.frag")

    assert gl.attached == [1, 2]
    assert isinstance(shader, shader_module.Shader)


def test_shader_error_is_a_runtime_error(shader_module):
    assert issubclass(shader_module.ShaderError, RuntimeError)


def test_renderer_terminates_glfw_when_a_shader_fails(monkeypatch, fresh_rendering_modules):
    """A shader failure must not leave GLFW initialised (fake glfw, glm and OpenGL)."""
    from unittest import mock

    import numpy as np

    for name in ["glfw", "glm", "OpenGL", "OpenGL.GL"]:
        monkeypatch.setitem(sys.modules, name, mock.MagicMock(name=name))
    renderer = importlib.import_module("cellular_automata.rendering.Renderer")
    shader = sys.modules[SHADER_MODULE]

    def fail(vertex, fragment):
        raise shader.ShaderError("Vertex shader solid.vert failed to compile:\nbroken")

    monkeypatch.setattr(renderer, "build_shader", fail)

    with pytest.raises(shader.ShaderError, match="broken"):
        renderer.CARenderer(np.zeros((2, 2, 2)), 64, 64)

    sys.modules["glfw"].terminate.assert_called_once_with()


@pytest.fixture
def mocked_renderer(monkeypatch, fresh_rendering_modules):
    """The renderer module imported against mocked glfw, glm and OpenGL."""
    from unittest import mock

    for name in ["glfw", "glm", "OpenGL", "OpenGL.GL"]:
        monkeypatch.setitem(sys.modules, name, mock.MagicMock(name=name))
    return importlib.import_module("cellular_automata.rendering.Renderer")


def test_incomplete_framebuffer_raises_and_terminates_glfw(mocked_renderer, monkeypatch):
    monkeypatch.setattr(mocked_renderer, "glCheckFramebufferStatus", lambda target: 0x8CD6)
    monkeypatch.setattr(mocked_renderer, "GL_FRAMEBUFFER_COMPLETE", 0x8CD5)

    with pytest.raises(RuntimeError, match="opaque framebuffer is incomplete"):
        mocked_renderer.CARenderer._check_framebuffer("opaque")

    mocked_renderer.glfw.terminate.assert_called_once_with()


def test_complete_framebuffer_passes(mocked_renderer, monkeypatch):
    monkeypatch.setattr(mocked_renderer, "glCheckFramebufferStatus", lambda target: 0x8CD5)
    monkeypatch.setattr(mocked_renderer, "GL_FRAMEBUFFER_COMPLETE", 0x8CD5)

    mocked_renderer.CARenderer._check_framebuffer("opaque")

    mocked_renderer.glfw.terminate.assert_not_called()


def test_no_fake_backed_rendering_module_is_left_behind():
    """Runs after the faked tests in this file; they must have restored sys.modules."""
    import cellular_automata.rendering as rendering

    for name in RENDERING_MODULES:
        for module in (
            sys.modules.get(f"cellular_automata.rendering.{name}"),
            vars(rendering).get(name),
        ):
            if module is None:
                continue
            for dependency in ("glCreateProgram", "glfw", "glm"):
                value = getattr(module, dependency, None)
                assert not isinstance(getattr(value, "__self__", None), FakeGL)
                assert type(value).__module__ != "unittest.mock"
