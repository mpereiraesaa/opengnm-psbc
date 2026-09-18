#version 450
/* The viewport-selection half of the geometry metadata contract: the same
 * interface as merged-geometry.geom, plus the one thing that makes RADV count a
 * parameter export the varying loop cannot describe - gl_ViewportIndex, which
 * a geometry stage uses to select among the pipeline's viewport banks. The
 * merged pair must still compile: the semantic list has to name that export, or
 * the whole linkage field stays unresolved and a viewport-routing pipeline can
 * never be created. */
layout(triangles) in;
layout(triangle_strip, max_vertices = 4) out;
layout(location = 0) in vec3 color[];
layout(location = 0) out vec3 out_color;

out gl_PerVertex {
    vec4 gl_Position;
};

void main()
{
    const vec2 corners[4] = vec2[4](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    for (int i = 0; i < 4; ++i) {
        gl_ViewportIndex = gl_PrimitiveIDIn;
        gl_Position = vec4(corners[i], 0.5, 1.0);
        out_color = color[0];
        EmitVertex();
    }
    EndPrimitive();
}
