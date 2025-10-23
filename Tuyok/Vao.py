import numpy as np
from OpenGL import GL
import ctypes

class Vao:
    def __init__(self, shader, mesh):
        
        self.shader = shader
        self.mesh = mesh
        program = self.program = shader.program
        
        # "In"
        pos_loc = GL.glGetAttribLocation(program, "position")
        nrm_loc = GL.glGetAttribLocation(program, "normal")
        col_loc = GL.glGetAttribLocation(program, "color")
    
        
        
        #self.vertices, self.normals, self.indices, self.colors = mesh.get_buffer_data()
        self.vertex_data, self.indices = mesh.get_buffer_data()
        
        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        #self._vbo_vertices, self._vbo_normals, self._vbo_colors, self._ebo = GL.glGenBuffers(4)
        #self._bind_vec3_array(self._vbo_vertices, self.vertices, pos_loc)
        #self._bind_vec3_array(self._vbo_normals, self.normals, nrm_loc)
        #self._bind_vec3_array(self._vbo_colors, self.colors, col_loc)
    
        self._vbo, self._ebo = GL.glGenBuffers(2)
        
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 
                        self.vertex_data.nbytes, 
                        self.vertex_data, 
                        GL.GL_STATIC_DRAW)
    
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, 
                        self.indices.nbytes, self.indices, GL.GL_STATIC_DRAW)
    
        stride = self.vertex_data.strides[0]
        pos_offset, nrm_offset, col_offset = [ 
                        ctypes.c_void_p(self.vertex_data.dtype.fields[s][1])
                        for s in ('pos', 'nrm', 'col')]
    
        self._bind_vec3_array(self._vbo, self.vertex_data, pos_loc, stride, pos_offset)
        self._bind_vec3_array(self._vbo, self.vertex_data, nrm_loc, stride, nrm_offset)
        self._bind_vec3_array(self._vbo, self.vertex_data, col_loc, stride, col_offset)
    
    
    
        GL.glBindVertexArray(0)
    
    def _bind_vec3_array(self, vbo, array, program_loc, stride, offset):
        '''Binds a 3-float vector to a vbo, and binds the vbo to a shader program'''
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
                                 stride,
                                 offset 
                                 )

    def draw(self):
        GL.glUseProgram(self.shader.program)
        
    
        
        
        
        GL.glBindVertexArray(self._vao)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)





