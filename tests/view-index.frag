#version 450
#extension GL_EXT_multiview : require

/* The same built-in in a stage this profile does not deliver a view index to.
 * The compiler must keep lowering it to zero there and must not report a slot,
 * because nothing would write one. */

layout(location = 0) in vec4 in_color;
layout(location = 0) out vec4 out_color;

void main() {
    float shade = 0.25 * float(gl_ViewIndex);
    out_color = vec4(in_color.rgb * shade, 1.0);
}
