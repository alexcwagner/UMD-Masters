import numpy as np
from OpenGL import GL

class Vao:
    def __init__(self, shader, mesh):
        
        self.shader = shader
        self.mesh = mesh
        program = self.program = shader.program
        
        # "In"
        self._position_loc = GL.glGetAttribLocation(program, "position")
        self._normal_loc = GL.glGetAttribLocation(program, "normal")
        self._color_loc = GL.glGetAttribLocation(program, "color")
    
        
        
        self.vertices, self.normals, self.indices, self.colors = mesh.get_buffer_data()
        
        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        self._vbo_vertices, self._vbo_normals, self._vbo_colors, self._ebo = GL.glGenBuffers(4)
        self._bind_vec3_array(self._vbo_vertices, self.vertices, self._position_loc)
        self._bind_vec3_array(self._vbo_normals, self.normals, self._normal_loc)
        self._bind_vec3_array(self._vbo_colors, self.colors, self._color_loc)
    
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, 
                        self.indices.nbytes, self.indices, GL.GL_STATIC_DRAW)
    
        GL.glBindVertexArray(0)
    
    def _bind_vec3_array(self, vbo, array, program_loc):
        '''Binds a 3-float vector to a vbo, and binds the vbo to a shader program'''
        print(f"vbo: {vbo}\narray: {array}\nprogram_loc: {program_loc}")
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 
                        array.nbytes,
                        array,
                        GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(program_loc)    
        GL.glVertexAttribPointer(program_loc,
                                 3, # size of each item; 3 because it's a 3-vector
                                 GL.GL_FLOAT, # each of the 3 items is a float
                                 False, # "normalized"... always false for floats
                                 0, # stride
                                 None # offset. must use ctypes.c_void_p(num) if not None
                                 )

    def draw(self):
        GL.glUseProgram(self.shader.program)
        GL.glBindVertexArray(self._vao)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)





