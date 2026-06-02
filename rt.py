import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import ctypes
import glm







vertexShaderSource = """
#version 430 core

layout (location=0) in vec3 aPos;

flat out int InstanceID;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 grid_size;

void main(){
    InstanceID = gl_InstanceID;
    int x = gl_InstanceID % int(grid_size[0]);
    int y = gl_InstanceID/int(grid_size[0]);
    y = y % int(grid_size[1]);
    int z = gl_InstanceID/(int(grid_size[0])*int(grid_size[1]));
    int num_cubes = int(grid_size[0]*grid_size[1]*grid_size[2]);

    vec3 bPos = aPos;
    bPos.x = bPos.x+1.0*float(x);
    bPos.y = bPos.y+1.0*float(y);
    bPos.z = bPos.z+1.0*float(z);

    gl_Position=projection*view*model*vec4(bPos,1.0);
}
"""

fragmentShaderSource = """
#version 430 core

out vec4 FragColor;

flat in int InstanceID; 

uniform float window_width;
uniform float window_height;
uniform float[1000] color_data;

void main(){
   FragColor=vec4(color_data[InstanceID], gl_FragCoord.y/window_height, gl_FragCoord.z/100.0, 0.8); 
}
"""



WINDOW_WIDTH=1000
WINDOW_HEIGHT=600

GRID_SIZE = (5,5,2)

def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_A and action == glfw.PRESS:
        ROT_L=True
    if key == glfw.KEY_D and action == glfw.PRESS:
        ROT_R=True
    if key == glfw.KEY_Q and action == glfw.PRESS:
        ROT_U=True
    if key == glfw.KEY_E and action == glfw.PRESS:
        ROT_D=True
    if key == glfw.KEY_K and action == glfw.PRESS:
        ZOOM=True
    if key == glfw.KEY_L and action == glfw.PRESS:
        UNZOOM=True
    if key == glfw.KEY_A and action == glfw.RELEASE:
        ROT_L=False
    if key == glfw.KEY_D and action == glfw.RELEASE:
        ROT_R=False
    if key == glfw.KEY_Q and action == glfw.RELEASE:
        ROT_U=False
    if key == glfw.KEY_E and action == glfw.RELEASE:
        ROT_D=False
    if key == glfw.KEY_K and action == glfw.RELEASE:
        ZOOM=False
    if key == glfw.KEY_L and action == glfw.RELEASE:
        UNZOOM=False



ROT_R = False
ROT_L = False
ROT_U = False
ROT_D = False
ZOOM = False
UNZOOM = False 

ZOOM_STEP = 0.05
ROT_STEP = glm.radians(1.0)

if __name__=="__main__":

    if not glfw.init():
                raise Exception("GLFW initialization failed")
            
            




    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "nrt", None, None)
    if not window:
        glfw.terminate()
        
    glfw.set_window_pos(window, 400, 200)
    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.STICKY_KEYS, True)
    glfw.set_key_callback(window,key_callback)

    # compile shaders
    vertexShader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertexShader, vertexShaderSource)
    glCompileShader(vertexShader)
    if glGetShaderiv(vertexShader, GL_COMPILE_STATUS) == GL_FALSE:
        print("Vertex Shader Error:")
        print(glGetShaderInfoLog(vertexShader).decode())

    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragmentShader, fragmentShaderSource)
    glCompileShader(fragmentShader)
    if glGetShaderiv(fragmentShader, GL_COMPILE_STATUS) == GL_FALSE:
        print("Fragment Shader Error:")
        print(glGetShaderInfoLog(fragmentShader).decode())

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
    colors = colors.astype(dtype=np.float32)
    colors = colors.flatten()
    
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
    m = glm.scale(glm.vec3(0.1,0.1,0.1)) * m
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

    color_data = glGetUniformLocation(shaderProgram, "color_data")
    glUniform1fv(color_data,colors.size, colors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))



    def render():
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glBindBuffer(GL_ARRAY_BUFFER,vbo)
        glBindVertexArray(vao)







        glU

        glDrawArraysInstanced(GL_TRIANGLES, 0, 6*2*3,np.prod(GRID_SIZE))
        glfw.swap_buffers(window)

    while not glfw.window_should_close(window):
                # Update CA state if callback provided
                render()
                glfw.poll_events()
            
    glfw.terminate()