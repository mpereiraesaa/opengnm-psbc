#version 450

vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 0.0,  1.0)
);

void main() {
    vec2 pos = positions[gl_VertexIndex % 3];
    gl_Position = vec4(pos, 0.0, 1.0);
}
