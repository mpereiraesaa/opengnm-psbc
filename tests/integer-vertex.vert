#version 450

layout(location = 0) in ivec2 position;
layout(location = 1) in uvec4 color;
layout(location = 0) flat out uvec4 vertex_color;

void main()
{
    gl_Position = vec4(position, 0, 1);
    vertex_color = color;
}
