
VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

uniform mat4 mvp;
uniform mat4 model;

out vec3 fragNormal;
out vec3 baseColor;

void main() {
    fragNormal = normalize((model * vec4(normal, 0.0)).xyz);
    baseColor = color;
    gl_Position = mvp * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 fragNormal;
in vec3 baseColor;

out vec4 outColor;

void main() {
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
    float diff = max(dot(fragNormal, lightDir), 0.0);
    vec3 shaded = baseColor * (0.2 + 0.8 * diff); // ambient + diffuse
    outColor = vec4(shaded, 1.0);
}
"""
