#version 450

layout(location = 0) in vec4 varyings[32];
layout(location = 0) out vec4 color;

void main()
{
    color = vec4(0.0);
    for (int slot = 0; slot < 32; ++slot)
        color += varyings[slot];
}
