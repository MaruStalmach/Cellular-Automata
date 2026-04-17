import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math


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
        """Returns view matrix components for gluLookAt"""
        return self.position, self.target, self.up


class CARenderer:
    """Renders 3D cellular automaton"""
    
    def __init__(self, width=1000, height=600):
        self.width = width
        self.height = height
        
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(width, height, "3D Cellular Automaton", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
        
        glfw.set_window_pos(self.window, 400, 200)
        glfw.make_context_current(self.window)
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS, True)
        
        # Enable VSync
        glfw.swap_interval(1)
        
        # OpenGL setup
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glCullFace(GL_BACK)
        
        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light setup
        light_pos = [5, 10, 5, 0]
        glLight(GL_LIGHT0, GL_POSITION, light_pos)
        glLight(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLight(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        
        self.camera = Camera(position=(20, 20, 20))
        self.ca_state = None  # Will hold 3D numpy array
        self.cell_size = 1.0
        self.running = True
        
        # Setup keyboard callback
        glfw.set_key_callback(self.window, self._key_callback)

        self.p_press = False
    
    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.running = False
            return
        
        if key == glfw.KEY_P and action == glfw.PRESS:
            self.p_press = True
        
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
        self.ca_state = state_array.astype(bool)
    
    def draw_cube(self, x, y, z, size=1.0):
        """Draw a unit cube at position (x, y, z)"""
        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(size, size, size)
        
        glBegin(GL_TRIANGLES) #change to traingle_strip?
        
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, -0.5, -0.5)
        
        glEnd()
        glPopMatrix()
    
    def render(self):
        """Main render loop"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.width / self.height, 0.1, 500.0)
        
        # Setup modelview
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Update camera
        self.camera.update()
        pos, target, up = self.camera.get_view_matrix()
        gluLookAt(pos[0], pos[1], pos[2],
                  target[0], target[1], target[2],
                  up[0], up[1], up[2])
        

        # Draw origin marker (small red cube)
        glColor3f(1.0, 0.0, 0.0)
        self.draw_cube(0, 0, 0, 0.2)
        
        # Draw CA cells
        if self.ca_state is not None:
            glColor4f(0.3, 0.8, 0.3, 0.5)  # Green for alive cells
            
            indices = np.where(self.ca_state)
            for i, j, k in zip(indices[0], indices[1], indices[2]):
                # Center around origin
                x = (i - self.ca_state.shape[0] / 2) * self.cell_size
                y = (j - self.ca_state.shape[1] / 2) * self.cell_size
                z = (k - self.ca_state.shape[2] / 2) * self.cell_size
                self.draw_cube(x, y, z, self.cell_size * 0.95)
        
        
        
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
                    new_state = update_callback()
                    print(new_state)
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
    example_state = np.random.choice(a=[0,1],p=[0.8,0.2],size = (10,10,10))#np.random.randint(low=2,size=(10,10,10))
    renderer.set_ca_state(example_state)
    
    # Run without update (static display)
    renderer.run()