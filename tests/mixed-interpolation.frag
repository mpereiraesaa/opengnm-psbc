#version 450

layout(location = 0) flat in vec3 flat_color;
layout(location = 1) in vec3 smooth_color;
layout(location = 0) out vec4 color;

void main()
{
    color = vec4(flat_color + smooth_color, 1.0);
}
