#version 450
#extension GL_EXT_multiview : require

/* PS5 replay delivers the current view through the fragment user-data slot. */

layout(location = 0) in vec4 in_color;
layout(location = 0) out vec4 out_color;

void main() {
    float shade = 0.25 * float(gl_ViewIndex);
    out_color = vec4(in_color.rgb * shade, 1.0);
}
