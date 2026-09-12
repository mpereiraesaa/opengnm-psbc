#version 450
layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;
layout(location = 0) out vec2 captured;

void main()
{
    for (int primitive = 0; primitive < 2; ++primitive) {
        vec2 offset = vec2(float(primitive) * 0.25,
                           float(primitive) * 0.5);
        for (int i = 0; i < 3; ++i) {
            gl_Position = gl_in[i].gl_Position + vec4(offset, 0.0, 0.0);
            captured = gl_Position.xy;
            EmitVertex();
        }
        EndPrimitive();
    }
}
