#version 450

layout(location = 0, index = 0) out vec4 primary_color;
layout(location = 0, index = 1) out vec4 secondary_color;

void main()
{
    primary_color = vec4(1.0, 0.0, 0.0, 1.0);
    secondary_color = vec4(1.0, 0.0, 0.0, 1.0);
}
