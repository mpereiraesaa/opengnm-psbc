#version 450
#extension GL_ARB_shader_draw_parameters : require

/* The DrawIndex built-in (gl_DrawID) is the only reason this shader needs the
 * ABI user-data block entry that RADV places after the base vertex, so the
 * metadata regression can isolate that one slot. The matching negative case
 * compiles tests/tri.vert, which never reads the built-in. */

layout(location = 0) in vec4 in_position;
layout(location = 0) out vec4 out_color;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    float shade = 0.25 * float(gl_DrawIDARB);
    gl_Position = vec4(in_position.xyz, 1.0);
    out_color = vec4(shade, shade, 1.0 - shade, 1.0);
}
