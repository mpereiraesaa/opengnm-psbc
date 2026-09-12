#version 450

layout(location = 0) out vec4 color;

void main()
{
    color = gl_FrontFacing ? vec4(0.0, 1.0, 0.0, 1.0)
                           : vec4(1.0, 0.0, 0.0, 1.0);
}
