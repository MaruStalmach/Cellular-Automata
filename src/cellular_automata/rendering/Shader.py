from OpenGL.GL import *


class Shader:
    def __init__(self):
        self.shaderProgram = glCreateProgram()

    def add_vertex(self, filename):
        vs = glCreateShader(GL_VERTEX_SHADER)
        with open(filename, "r") as f:
            source = f.read()
        glShaderSource(vs, source)
        glCompileShader(vs)
        if glGetShaderiv(vs, GL_COMPILE_STATUS) == GL_FALSE:
            print("Vertex Shader Error:")
            print(glGetShaderInfoLog(vs).decode())
        glAttachShader(self.shaderProgram, vs)
        glDeleteShader(vs)

    def add_fragment(self, filename):
        fs = glCreateShader(GL_FRAGMENT_SHADER)
        with open(filename, "r") as f:
            source = f.read()
        glShaderSource(fs, source)
        glCompileShader(fs)
        if glGetShaderiv(fs, GL_COMPILE_STATUS) == GL_FALSE:
            print("Vertex Shader Error:")
            print(glGetShaderInfoLog(fs).decode())
        glAttachShader(self.shaderProgram, fs)
        glDeleteShader(fs)

    def link_program(self):
        glLinkProgram(self.shaderProgram)
        if glGetProgramiv(self.shaderProgram, GL_LINK_STATUS) == GL_FALSE:
            print("Linking error:")
            print(glGetProgramInfoLog(self.shaderProgram).decode())

    def use_program(self):
        glUseProgram(self.shaderProgram)

    @property
    def program(self):
        return self.shaderProgram
