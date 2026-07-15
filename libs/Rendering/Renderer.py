import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes
import glm

from libs.Rendering.Camera import Camera
from libs.Rendering.Shader import Shader


class CARenderer:
    """Renders 3D cellular automata"""

    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.running = False
            return

        elif key == glfw.KEY_P and action == glfw.PRESS:
            self.p_press = True

        elif key == glfw.KEY_C and action == glfw.PRESS:
            self.c_press = True

        elif key == glfw.KEY_J and action == glfw.PRESS:
            self.j_press = True

        elif key == glfw.KEY_L and action == glfw.PRESS:
            self.l_press = True
        

        # Map GLFW keys to camera keys
        key_map = {
            glfw.KEY_W: "W",
            glfw.KEY_A: "A",
            glfw.KEY_S: "S",
            glfw.KEY_D: "D",
            glfw.KEY_Q: "Q",
            glfw.KEY_E: "E",
        }

        if key in key_map:
            self.camera.set_key_state(key_map[key], action != glfw.RELEASE)

    def __init__(
        self,
        init_state: np.ndarray,
        width=1000,
        height=600,
        update_callback: callable = None,
        keys_to_render: int = 1,
    ):

        self.width = width
        self.height = height

        self.num_keys = keys_to_render

        self.p_press = False
        self.c_press = False
        self.j_press = False
        self.l_press = False

        self.cross_section = False
        self.layer=0

        if update_callback is None:
            self.update_callback = None
        else:
            self.update_callback = update_callback

        self.ca_state = init_state.astype(dtype=np.float32)
        self.cell_size = 1.0
        self.running = True
        self.camera = Camera(position=(0, 0, init_state.shape[-1] * 1.2))

        # const setup
        self.zero_filler = glm.vec4(0.0)
        self.one_filler = glm.vec4(1.0)

        if not glfw.init():
            raise Exception("GLFW initialization failed")

        self.window = glfw.create_window(width, height, "nrt", None, None)
        if not self.window:
            glfw.terminate()

        glfw.set_window_pos(self.window, 400, 200)
        glfw.make_context_current(self.window)
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS, True)
        glfw.set_key_callback(self.window, self._key_callback)

        # compile shaders
        self.solidShader = Shader()
        self.solidShader.add_vertex("libs/Rendering/shaders/solid.vert")
        self.solidShader.add_fragment("libs/Rendering/shaders/solid.frag")
        self.solidShader.link_program()

        self.transShader = Shader()
        self.transShader.add_vertex("libs/Rendering/shaders/vertex.vert")
        self.transShader.add_fragment("libs/Rendering/shaders/transparent.frag")
        self.transShader.link_program()

        self.compShader = Shader()
        self.compShader.add_vertex("libs/Rendering/shaders/composite.vert")
        self.compShader.add_fragment("libs/Rendering/shaders/composite.frag")
        self.compShader.link_program()

        self.screenShader = Shader()
        self.screenShader.add_vertex("libs/Rendering/shaders/screen.vert")
        self.screenShader.add_fragment("libs/Rendering/shaders/screen.frag")
        self.screenShader.link_program()

        glViewport(0, 0, width, height)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # cube vertices
        vertices = np.array(
            [
                # bottom
                -0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                # front
                -0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                # top
                -0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                # back
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                # left
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                # right
                0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
            ],
            dtype=np.float32,
        )

        vbo = glGenBuffers(1)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)

        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        quadVertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        quadVbo = glGenBuffers(1)
        self.quadVao = glGenVertexArrays(1)
        glBindVertexArray(self.quadVao)
        glBindBuffer(GL_ARRAY_BUFFER, quadVbo)
        glBufferData(GL_ARRAY_BUFFER, quadVertices.nbytes, quadVertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        # maybe wrong
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glBindVertexArray(0)

        # setup FBOs
        self.opqFBO = glGenFramebuffers(1)
        self.transFBO = glGenFramebuffers(1)

        # attachments for opaque fbo
        self.opqTex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.opqTex)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA16F,
            self.width,
            self.height,
            0,
            GL_RGBA,
            GL_HALF_FLOAT,
            None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        depthTex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, depthTex)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_DEPTH_COMPONENT,
            self.width,
            self.height,
            0,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
            None,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, self.opqFBO)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.opqTex, 0
        )
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0
        )

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("something wrong with the opaque framebuffer")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # attachments for transparent fbo
        self.accumTex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.accumTex)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA16F,
            self.width,
            self.height,
            0,
            GL_RGBA,
            GL_HALF_FLOAT,
            None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.revealTex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.revealTex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R8, self.width, self.height, 0, GL_RED, GL_FLOAT, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, self.transFBO)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.accumTex, 0
        )
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, self.revealTex, 0
        )
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0
        )

        glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("transparent fbo failed")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        m = glm.identity(glm.mat4)
        self.v = self.camera.get_view_matrix()
        p = glm.perspective(
            glm.radians(60.0),
            self.width / self.height,
            1.0,
            max(init_state.shape) * 7.0,
        )

        # send uniforms
        self.transShader.use_program()
        # ssbo
        self.ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo)
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            self.ca_state.nbytes,
            self.ca_state.flatten(),
            GL_DYNAMIC_COPY,
        )
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo)
        # unbind
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        window_width = glGetUniformLocation(self.transShader.program, "window_width")
        glUniform1f(window_width, self.width)

        window_height = glGetUniformLocation(self.transShader.program, "window_height")
        glUniform1f(window_height, self.height)

        m_loc = glGetUniformLocation(self.transShader.program, "model")
        glUniformMatrix4fv(m_loc, 1, GL_FALSE, glm.value_ptr(m))

        v_loc = glGetUniformLocation(self.transShader.program, "view")
        glUniformMatrix4fv(v_loc, 1, GL_FALSE, glm.value_ptr(self.v))

        p_loc = glGetUniformLocation(self.transShader.program, "projection")
        glUniformMatrix4fv(p_loc, 1, GL_FALSE, glm.value_ptr(p))

        grid_size = glGetUniformLocation(self.transShader.program, "grid_size")
        glUniform3f(
            grid_size,
            self.ca_state.shape[-3],
            self.ca_state.shape[-2],
            self.ca_state.shape[-1],
        )

        n_keys = glGetUniformLocation(self.transShader.program, "n_keys")
        glUniform1i(n_keys, keys_to_render)

        # send pvm to soild  shader
        self.solidShader.use_program()

        # make smaller cube
        m = glm.scale(glm.vec3(0.3, 0.3, 0.3)) * m

        m_loc = glGetUniformLocation(self.solidShader.program, "model")
        glUniformMatrix4fv(m_loc, 1, GL_FALSE, glm.value_ptr(m))

        v_loc = glGetUniformLocation(self.solidShader.program, "view")
        glUniformMatrix4fv(v_loc, 1, GL_FALSE, glm.value_ptr(self.v))

        p_loc = glGetUniformLocation(self.solidShader.program, "projection")
        glUniformMatrix4fv(p_loc, 1, GL_FALSE, glm.value_ptr(p))

        

    def set_ca_state(self, state_array):
        """
        Set the cellular automaton state.

        Args:
            state_array: 3D numpy array where True/non-zero = cell alive
                        Shape: (depth, height, width) or similar
        """
        self.ca_state = state_array.astype(np.float32)
        self.transShader.use_program()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo)
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            self.ca_state.nbytes,
            self.ca_state.flatten(),
            GL_DYNAMIC_COPY,
        )
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo)
        # unbind
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def render(self):

        self.transShader.use_program()
        # process input
        self.camera.update()
        self.v = self.camera.get_view_matrix()
        v_loc = glGetUniformLocation(self.transShader.program, "view")
        glUniformMatrix4fv(v_loc, 1, GL_FALSE, glm.value_ptr(self.v))

        self.solidShader.use_program()

        v_loc = glGetUniformLocation(self.solidShader.program, "view")
        glUniformMatrix4fv(v_loc, 1, GL_FALSE, glm.value_ptr(self.v))

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        glEnable(GL_BLEND)
        glClearColor(0.0, 0.0, 0.0, 0.0)

        # bind opaque fbo to render solid objects
        glBindFramebuffer(GL_FRAMEBUFFER, self.opqFBO)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw solid objects
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6 * 2 * 3)

        # transparent objects
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunci(0, GL_ONE, GL_ONE)
        glBlendFunci(1, GL_ZERO, GL_ONE_MINUS_SRC_COLOR)
        glBlendEquation(GL_FUNC_ADD)

        glBindFramebuffer(GL_FRAMEBUFFER, self.transFBO)
        # maybe wrong
        glClearBufferfv(GL_COLOR, 0, glm.value_ptr(self.zero_filler))
        glClearBufferfv(GL_COLOR, 1, glm.value_ptr(self.one_filler))

        self.transShader.use_program()

        glBindVertexArray(self.vao)
        glDrawArraysInstanced(
            GL_TRIANGLES, 0, 6 * 2 * 3, np.prod(self.ca_state.shape) // self.num_keys
        )

        # draw composite image
        glDepthFunc(GL_ALWAYS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindFramebuffer(GL_FRAMEBUFFER, self.opqFBO)

        self.compShader.use_program()

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.accumTex)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.revealTex)
        glBindVertexArray(self.quadVao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # draw to backbuffer (final pass)

        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

        # bindbackbuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)  # 0 is default glfw fbo
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        self.screenShader.use_program()

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.opqTex)
        glBindVertexArray(self.quadVao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glfw.swap_buffers(self.window)

    def run(self):
        """
        Main loop. update_callback will be called each frame to update CA state.

        Args:
            update_callback: Optional function that should return new CA state
        """
        while self.running and not glfw.window_should_close(self.window):
            # Update CA state if callback provided
            if self.update_callback:
                new_state = None
                if self.p_press:
                    new_state = self.update_callback()
                    self.p_press = False
                if self.c_press:
                    self.c_press = False
                    self.cross_section = not self.cross_section
                if self.j_press:
                    self.j_press = False
                    if self.cross_section:
                        self.layer -= 1
                        self.layer=np.clip(self.layer,0,self.ca_state.shape[-1])
                if self.l_press:
                    self.l_press = False
                    if self.cross_section:
                        self.layer += 1
                        self.layer=np.clip(self.layer,0,self.ca_state.shape[-1])
                    
                if new_state is not None:
                    self.set_ca_state(new_state)

            self.render()
            glfw.poll_events()

        glfw.terminate()
        quit()


# Example usage
if __name__ == "__main__":
    renderer = CARenderer(
        width=1200,
        height=800,
        init_state=np.random.choice([0, 1], size=(100, 100, 100), p=[0.999, 0.001]),
    )
    renderer.run()

    # np.array([
    #     [[0.1,0.5,1.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],
    #     [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],
    #     [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]],dtype=np.float32)
