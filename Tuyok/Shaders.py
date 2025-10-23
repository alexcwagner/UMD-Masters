
VERTEX_SHADER_SRC = """
#version 460 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragNormal;
out vec3 baseColor;

void main() 
{
    fragNormal = normalize((model * vec4(normal, 0.0)).xyz);
    baseColor = color;
    
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 460 core
in vec3 fragNormal;
in vec3 baseColor;

out vec4 outColor;

void main() {
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.0));
    float diff = max(dot(fragNormal, lightDir), 0.0);
    vec3 shaded = baseColor * (0.2 + 0.8 * diff); // ambient + diffuse
    outColor = vec4(shaded, 1.0);
}
"""

from OpenGL import GL
#import OpenGL.GL.shaders
import numpy as np

class Shader:

    def __init__(self):
        program = self.program = GL.shaders.compileProgram(
            GL.shaders.compileShader(VERTEX_SHADER_SRC, GL.GL_VERTEX_SHADER),
            GL.shaders.compileShader(FRAGMENT_SHADER_SRC, GL.GL_FRAGMENT_SHADER),
        )


        # Uniforms
        self._model_loc = GL.glGetUniformLocation(program, "model")
        self._view_loc = GL.glGetUniformLocation(program, "view")
        self._projection_loc = GL.glGetUniformLocation(program, "projection")

        self.model = np.identity(4, dtype=np.float32)
        self.view = np.identity(4, dtype=np.float32)
        self.projection = np.identity(4, dtype=np.float32)

        
        
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        self._model = value
        GL.glProgramUniformMatrix4fv(
                self.program,
                self._model_loc, 
                1, GL.GL_TRUE, 
                value.astype(np.float32))

    @property
    def view(self):
        return self._view
    @view.setter
    def view(self, value):
        self._view = value
        GL.glProgramUniformMatrix4fv(
                self.program,
                self._view_loc, 
                1, GL.GL_TRUE, 
                value.astype(np.float32))   
    
    @property
    def projection(self):
        return self._projection
    @projection.setter
    def projection(self, value):
        self._projection = value
        GL.glProgramUniformMatrix4fv(
                self.program,
                self._projection_loc, 
                1, GL.GL_TRUE, 
                value.astype(np.float32))  


        
        
        
        
        
        
        
        
        
        
        
        
        
        