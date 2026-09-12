#version 450

layout(location = 0) in vec4 in_position;
layout(location = 0) out vec4 captured_position;

void main()
{
    gl_Position = in_position;
    captured_position = in_position;
}
