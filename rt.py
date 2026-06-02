import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import ctypes
import glm







vertexShaderSource = """
#version 330 core

layout (location=0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 grid_size;

void main(){
   gl_Position=projection*view*model*vec4(aPos,1.0); 
}
"""

fragmentShaderSource = """
#version 330 core

out vec4 FragColor;

uniform float window_width;
uniform float window_height;

void main(){
   FragColor=vec4(gl_FragCoord.x/window_width, gl_FragCoord.y/window_height, gl_FragCoord.z/100.0, 0.8); 
}
"""



WINDOW_WIDTH=1000
WINDOW_HEIGHT=600

GRID_SIZE = (3,1,1)

if __name__=="__main__":

    if not glfw.init():
                raise Exception("GLFW initialization failed")
            
            




    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "nrt", None, None)
    if not window:
        glfw.terminate()
        
    glfw.set_window_pos(window, 400, 200)
    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.STICKY_KEYS, True)

    # compile shaders
    vertexShader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertexShader, vertexShaderSource)
    glCompileShader(vertexShader)

    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragmentShader, fragmentShaderSource)
    glCompileShader(fragmentShader)

    shaderProgram = glCreateProgram()
    glAttachShader(shaderProgram,vertexShader)
    glAttachShader(shaderProgram,fragmentShader)
    glLinkProgram(shaderProgram)

    glDeleteShader(vertexShader)
    glDeleteShader(fragmentShader)

    glClearColor(0.0,0.0,0.0,1.0)
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_BLEND)
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    glUseProgram(shaderProgram)
    
    colors = np.random.random(size=GRID_SIZE)
    
    vertices = np.array([
        #bottom
        -0.5,-0.5,-0.5,
        0.5,-0.5,-0.5,
        0.5,-0.5,0.5,
        
        -0.5,-0.5,-0.5,
        0.5,-0.5,0.5,
        -0.5,-0.5,0.5,
        #front
        -0.5,-0.5,0.5,
        0.5,-0.5,0.5,
        -0.5,0.5,0.5,
        
        -0.5,0.5,0.5,
        0.5,0.5,0.5,
        0.5,-0.5,0.5,
        #top
        -0.5,0.5,0.5,
        0.5,0.5,0.5,
        -0.5,0.5,-0.5,
        
        0.5,0.5,0.5,
        -0.5,0.5,-0.5,
        0.5,0.5,-0.5,
        #back
        -0.5,-0.5,-0.5,
        0.5,-0.5,-0.5,
        -0.5,0.5,-0.5,
        
        -0.5,0.5,-0.5,
        0.5,0.5,-0.5,
        0.5,-0.5,-0.5,
        #left
        -0.5,-0.5,0.5,
        -0.5,0.5,0.5,
        -0.5,-0.5,-0.5,
        
        -0.5,0.5,0.5,
        -0.5,-0.5,-0.5,
        -0.5,0.5,-0.5,
        #right
        0.5,-0.5,0.5,
        0.5,0.5,0.5,
        0.5,-0.5,-0.5,
        
        0.5,0.5,0.5,
        0.5,-0.5,-0.5,
        0.5,0.5,-0.5
        ],dtype=np.float32)

    vbo = glGenBuffers(1)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    m = glm.identity(glm.mat4)
    m = glm.rotate(glm.radians(45.0),glm.vec3(0,1,0)) * glm.scale(glm.vec3(0.2,0.2,0.2)) * m
    v = glm.lookAt(glm.vec3(0.0, 0.0, 2.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    p = glm.perspective(glm.radians(60.0), WINDOW_WIDTH/WINDOW_HEIGHT, 0.1, 100.0)
    
    
    window_width = glGetUniformLocation(shaderProgram,"window_width")
    glUniform1f(window_width, WINDOW_WIDTH)
    
    window_height = glGetUniformLocation(shaderProgram,"window_height")
    glUniform1f(window_height, WINDOW_HEIGHT)
    
    m_loc = glGetUniformLocation(shaderProgram, "model")
    glUniformMatrix4fv(m_loc, 1, GL_FALSE, glm.value_ptr(m))
    
    v_loc = glGetUniformLocation(shaderProgram, "view")
    glUniformMatrix4fv(v_loc, 1, GL_FALSE, glm.value_ptr(v))
    
    p_loc = glGetUniformLocation(shaderProgram, "projection")
    glUniformMatrix4fv(p_loc, 1, GL_FALSE, glm.value_ptr(p))
    
    grid_size = glGetUniformLocation(shaderProgram, "grid_size")
    glUniform3f(grid_size, GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2])


    def render():
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glBindBuffer(GL_ARRAY_BUFFER,vbo)
        glBindVertexArray(vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6*2*3,np.prod(GRID_SIZE))
        glfw.swap_buffers(window)

    while not glfw.window_should_close(window):
                # Update CA state if callback provided
                render()
                glfw.poll_events()
            
    glfw.terminate()