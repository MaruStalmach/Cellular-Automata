from pathlib import Path

from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_FALSE,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    GL_VERTEX_SHADER,
    glAttachShader,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteProgram,
    glDeleteShader,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glLinkProgram,
    glShaderSource,
    glUseProgram,
)

SHADER_DIR = Path(__file__).resolve().parent / "shaders"


class ShaderError(RuntimeError):
    """A shader failed to compile or a program failed to link."""


def _log_text(log) -> str:
    return log.decode(errors="replace") if isinstance(log, bytes) else str(log)


class Shader:
    def __init__(self):
        self.shaderProgram = glCreateProgram()

    def _add_stage(self, stage, label: str, filename) -> None:
        try:
            with open(filename, "r") as f:
                source = f.read()
        except OSError as error:
            raise ShaderError(f"{label} shader {filename} could not be read: {error}") from error
        shader = glCreateShader(stage)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) == GL_FALSE:
            log = _log_text(glGetShaderInfoLog(shader))
            glDeleteShader(shader)
            raise ShaderError(f"{label} shader {filename} failed to compile:\n{log}")
        glAttachShader(self.shaderProgram, shader)
        glDeleteShader(shader)

    def add_vertex(self, filename):
        self._add_stage(GL_VERTEX_SHADER, "Vertex", filename)

    def add_fragment(self, filename):
        self._add_stage(GL_FRAGMENT_SHADER, "Fragment", filename)

    def link_program(self):
        glLinkProgram(self.shaderProgram)
        if glGetProgramiv(self.shaderProgram, GL_LINK_STATUS) == GL_FALSE:
            log = _log_text(glGetProgramInfoLog(self.shaderProgram))
            glDeleteProgram(self.shaderProgram)
            raise ShaderError(f"Shader program failed to link:\n{log}")

    def use_program(self):
        glUseProgram(self.shaderProgram)

    @property
    def program(self):
        return self.shaderProgram


def build_shader(vertex: str, fragment: str) -> Shader:
    """Compile and link a program from two files in the shaders directory."""
    shader = Shader()
    shader.add_vertex(str(SHADER_DIR / vertex))
    shader.add_fragment(str(SHADER_DIR / fragment))
    shader.link_program()
    return shader
