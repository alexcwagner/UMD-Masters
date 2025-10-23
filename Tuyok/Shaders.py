
VERTEX_SHADER_SRC = """
#version 460 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 frag_normal;
out vec3 frag_position;
out vec3 frag_color;

void main() 
{
    mat3 norm_mat = transpose(inverse(mat3(model)));
    frag_normal = normalize(norm_mat * normal);
    vec4 pos_ws = model * vec4(position, 1.0);
    frag_position = pos_ws.xyz;
    frag_color = color;
    
    gl_Position = projection * view * pos_ws;
}
"""

FRAGMENT_SHADER_SRC = """
#version 460 core

struct Material {
    vec3 albedo;
    float specular_strength;
    float shininess;
    vec3 emissive;
    int flags;    
};

uniform Material material;
uniform vec3 light_direction;
uniform vec3 light_color;
uniform vec3 ambient;
uniform vec3 camera;

in vec3 frag_normal;
in vec3 frag_position;
in vec3 frag_color;

out vec4 out_color;

void main() {
    vec3 N = normalize(frag_normal);
    vec3 L = normalize(-light_direction);
    vec3 V = normalize(camera - frag_position);
    vec3 H = normalize(L + V);
    
    float N_dot_L = max(dot(N, L), 0.0);
    
    vec3 base = material.albedo * frag_color;
    vec3 diff = base * N_dot_L * light_color;
    
    float spec = pow(max(dot(N, H), 0.0), max(material.shininess, 1.0));
    vec3 specular = material.specular_strength * spec * light_color;
    
    vec3 color = ambient * base + diff + specular + material.emissive;
    
    out_color = vec4(color, 1.0);
}
"""

from OpenGL import GL
import OpenGL.GL.shaders
import numpy as np
from dataclasses import dataclass

@dataclass
class Material:
    albedo: tuple = (0.8, 0.8, 0.8)
    specular_strength: float = 0.04
    shininess: float = 32.0
    emissive: tuple = (0.0, 0.0, 0.0)
    flags: int = 0

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
        self._camera_loc = GL.glGetUniformLocation(program, "camera")
        self._light_direction_loc = GL.glGetUniformLocation(program, "light_direction")
        self._light_color_loc = GL.glGetUniformLocation(program, "light_color")
        self._ambient_loc = GL.glGetUniformLocation(program, "ambient")

        self.model = np.identity(4, dtype=np.float32)
        self.view = np.identity(4, dtype=np.float32)
        self.projection = np.identity(4, dtype=np.float32)

        self.bind_material(Material())
        
        self.camera = (3.0, 3.0, 1.0)
        self.light_direction = (3.0, -3.0, 3.0)
        self.light_color = (1.0, 1.0, 1.0)
        self.ambient = (0.1, 0.1, 0.2)
        
        
    @property
    def camera(self):
        return self._camera
    @camera.setter
    def camera(self, value):
        self._camera = value
        GL.glProgramUniform3f(
                self.program,
                self._camera_loc,
                *self._camera)
        
    @property
    def light_direction(self):
        return self._light_direction
    @light_direction.setter
    def light_direction(self, value):
        self._light_direction = value
        GL.glProgramUniform3f(
                self.program,
                self._light_direction_loc,
                *self._light_direction)    
        
    @property
    def light_color(self):
        return self._light_color
    @light_color.setter
    def light_color(self, value):
        self._light_color = value
        GL.glProgramUniform3f(
                self.program,
                self._light_color_loc,
                *self._light_color) 

    @property
    def ambient(self):
        return self._ambient
    @ambient.setter
    def ambient(self, value):
        self._ambient = value
        GL.glProgramUniform3f(
                self.program,
                self._ambient_loc,
                *self._ambient) 
        
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


    def bind_material(self, mat: Material):
        # Cache names -> locations in your ShaderProgram; inline here for clarity
        GL.glProgramUniform3f(
                    self.program,
                    GL.glGetUniformLocation(self.program, "material.albedo"), 
                    *mat.albedo)
        GL.glProgramUniform1f(
                    self.program,
                    GL.glGetUniformLocation(self.program, "material.specular_strength"), 
                    mat.specular_strength)
        GL.glProgramUniform1f(
                    self.program,
                    GL.glGetUniformLocation(self.program, "material.shininess"),
                    mat.shininess)
        GL.glProgramUniform3f(
                    self.program,
                    GL.glGetUniformLocation(self.program, "material.emissive"),
                    *mat.emissive)
        GL.glProgramUniform1i(
                    self.program,
                    GL.glGetUniformLocation(self.program, "material.flags"),
                    mat.flags)
        
        
        
        
        
        
        
        
        
        
        
        
        