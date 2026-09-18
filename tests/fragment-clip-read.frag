#version 450
in float gl_ClipDistance[2];
layout(location=0) out vec4 o_color;
void main() { o_color = vec4(gl_ClipDistance[1], 0.0, 0.0, 1.0); }
