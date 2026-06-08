import numpy as np
import glm

class Camera:
    """3D camera with WASD and QE controls"""
    
    def __init__(self, position=(10, 10, 10)):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array([0, 0, 0], dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.speed = 0.05
        
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
        speed_mod = np.linalg.norm(self.position)
        # Horizontal movement
        if self.keys['W']:
            movement += direction * self.speed * speed_mod
        if self.keys['S']:
            movement -= direction * self.speed * speed_mod
        if self.keys['D']:
            movement += right * self.speed * speed_mod
        if self.keys['A']:
            movement -= right * self.speed * speed_mod
        
        # Vertical movement
        if self.keys['Q']:
            movement += up * self.speed * speed_mod
        if self.keys['E']:
            movement -= up * self.speed * speed_mod
        
        self.position += movement
        # self.target += movement
    
    def get_view_matrix(self):
        
        return glm.lookAt(self.position, self.target, self.up)