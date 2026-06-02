import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import ctypes
import glm

from libs.Rendering.Camera import Camera
from libs.Rendering.Shader import Shader


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
uniform float[20000] color_data;

void main(){
   FragColor=vec4(color_data[InstanceID], gl_FragCoord.y/window_height, gl_FragCoord.z/100.0, 0.8); 
}
"""
import imageio


class Camera:
    """3D camera with WASD and QE controls"""
    
    def __init__(self, position=(10, 10, 10)):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array([0, 0, 0], dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.speed = 0.2
        
        # Key states
        self.keys = {
            'W': False, 'A': False, 'S': False, 'D': False,
            'Q': False, 'E': False
        }
    
    def set_key_state(self, key, state):
        """Update key press state"""
        key_upper = key.upper()
        if key_upper in self.keys:
            self.keys[key_upper] = state
    
    def update(self):
        """Update camera position based on key states"""
        direction = self.target - self.position
        direction_length = np.linalg.norm(direction)
        if direction_length > 0:
            direction = direction / direction_length
        
        # Get right vector (perpendicular to direction and up)
        right = np.cross(direction, self.up)
        right_length = np.linalg.norm(right)
        if right_length > 0:
            right = right / right_length
        
        # Recalculate up to be perpendicular to both
        up = np.cross(right, direction)
        
        movement = np.array([0.0, 0.0, 0.0])
        
        # Horizontal movement
        if self.keys['W']:
            movement += direction * self.speed
        if self.keys['S']:
            movement -= direction * self.speed
        if self.keys['D']:
            movement += right * self.speed
        if self.keys['A']:
            movement -= right * self.speed
        
        # Vertical movement
        if self.keys['Q']:
            movement += up * self.speed
        if self.keys['E']:
            movement -= up * self.speed
        
        self.position += movement
        # self.target += movement
    
    def get_view_matrix(self):
        
        return glm.lookAt(self.position, self.target, self.up)


class CARenderer:
    """Renders 3D cellular automaton"""
    
    def __init__(self, init_state : np.ndarray, width=1000, height=600):

        self.width=width
        self.height=height
        
        self.make_gif=make_gif
        self.frames=[]

        self.camera = Camera(position=(20, 20, 20))
        self.ca_state = init_state.astype(dtype=np.float32)  # Will hold 3D numpy array
        self.cell_size = 1.0
        self.running = True

        if not glfw.init():
                raise Exception("GLFW initialization failed")

        self.window = glfw.create_window(width, height, "nrt", None, None)
        if not self.window:
            glfw.terminate()
            
        glfw.set_window_pos(self.window, 400, 200)
        glfw.make_context_current(self.window)
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS, True)
        glfw.set_key_callback(self.window,self._key_callback)

        # compile shaders
        self.shader = Shader()
        self.shader.add_vertex("libs/Rendering/vertex.vert")
        self.shader.add_fragment("libs/Rendering/fragment.frag")
        self.shader.link_program()
        self.shader.use_program()

        glClearColor(0.0,0.0,0.0,1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glViewport(0, 0, width, height)
        glUseProgram(self.shader.program)

        #cube vertices
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

        self.vbo = glGenBuffers(1)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        m = glm.identity(glm.mat4)
        self.v = self.camera.get_view_matrix()
        p = glm.perspective(glm.radians(60.0), self.width/self.height, 0.1, 100.0)
        
        
        window_width = glGetUniformLocation(self.shader.program,"window_width")
        glUniform1f(window_width, self.width)
        
        window_height = glGetUniformLocation(self.shader.program,"window_height")
        glUniform1f(window_height, self.height)
        
        m_loc = glGetUniformLocation(self.shader.program, "model")
        glUniformMatrix4fv(m_loc, 1, GL_FALSE, glm.value_ptr(m))
        
        v_loc = glGetUniformLocation(self.shader.program, "view")
        glUniformMatrix4fv(v_loc, 1, GL_FALSE, glm.value_ptr(self.v))
        
        p_loc = glGetUniformLocation(self.shader.program, "projection")
        glUniformMatrix4fv(p_loc, 1, GL_FALSE, glm.value_ptr(p))
        
        grid_size = glGetUniformLocation(self.shader.program, "grid_size")
        glUniform3f(grid_size, self.ca_state.shape[0], self.ca_state.shape[1], self.ca_state.shape[2])

        color_data = glGetUniformLocation(self.shader.program, "color_data")
        glUniform1fv(color_data,self.ca_state.size, self.ca_state.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        
        
        
        

        self.p_press = False
        self.o_press = False
    
    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.running = False
            return
        
        if key == glfw.KEY_P and action == glfw.PRESS:
            self.p_press = True
        
        if self.make_gif:    
            if key == glfw.KEY_O and action == glfw.PRESS:
                imageio.mimwrite('gifs/out.gif',self.frames[1:],loop=0,duration=0.2)
        
        # Map GLFW keys to camera keys
        key_map = {
            glfw.KEY_W: 'W',
            glfw.KEY_A: 'A',
            glfw.KEY_S: 'S',
            glfw.KEY_D: 'D',
            glfw.KEY_Q: 'Q',
            glfw.KEY_E: 'E',
        }
        
        if key in key_map:
            self.camera.set_key_state(key_map[key], action != glfw.RELEASE)
    
    def set_ca_state(self, state_array):
        """
        Set the cellular automaton state.
        
        Args:
            state_array: 3D numpy array where True/non-zero = cell alive
                        Shape: (depth, height, width) or similar
        """
        self.ca_state = state_array.astype(np.float32)
    
    
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        # glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        glBindVertexArray(self.vao)

        self.camera.update()
        self.v = self.camera.get_view_matrix()
        v_loc = glGetUniformLocation(self.shader.program, "view")
        glUniformMatrix4fv(v_loc, 1, GL_FALSE, glm.value_ptr(self.v))

        color_data = glGetUniformLocation(self.shader.program, "color_data")
        glUniform1fv(color_data,self.ca_state.size, self.ca_state.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)))


        glDrawArraysInstanced(GL_TRIANGLES, 0, 6*2*3,np.prod(self.ca_state.shape))
        glfw.swap_buffers(self.window)
    
    def run(self, update_callback=None):
        """
        Main loop. update_callback will be called each frame to update CA state.
        
        Args:
            update_callback: Optional function that should return new CA state
        """
        while self.running and not glfw.window_should_close(self.window):
            # Update CA state if callback provided
            if update_callback:
                new_state = None
                if self.p_press:
                    if self.make_gif:
                        glPixelStorei(GL_PACK_ALIGNMENT, 1)
                        data = glReadPixels(0,0,self.width,self.height,GL_RGB,GL_UNSIGNED_BYTE)
                        img = np.frombuffer(data,dtype=np.uint8)
                        img = img.reshape(self.height,self.width,3)
                        img = np.flipud(img)
                        self.frames.append(img)
                    new_state = update_callback()
                    self.p_press = False
                if new_state is not None:
                    self.set_ca_state(new_state)
            
            self.render()
            glfw.poll_events()
        
        glfw.terminate()


# Example usage
if __name__ == "__main__":
    renderer = CARenderer(width=1200, height=800)
    
    # Example: Create a simple 3D pattern
    # Shape: (depth, height, width)
    # example_state = np.random.choice(a=[0,1],p=[0.8,0.2],size = (10,10,10))#np.random.randint(low=2,size=(10,10,10))
    # renderer.set_ca_state(example_state)
    
    # Run without update (static display)
    renderer.run()