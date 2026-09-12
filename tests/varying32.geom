#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) out vec4 varyings[32];

void main()
{
    for (int vertex = 0; vertex < 3; ++vertex) {
        gl_Position = gl_in[vertex].gl_Position;
        for (int slot = 0; slot < 32; ++slot)
            varyings[slot] = vec4(float(slot + 1));
        EmitVertex();
    }
    EndPrimitive();
}
