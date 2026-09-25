#version 450
/* Fragment form of the DXVK sample: s0 at binding 0, t0 at binding 1. */
layout(set=0,binding=0) uniform sampler s0;
layout(set=0,binding=1) uniform texture2D t0;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
void main() {
    color=texture(sampler2D(t0,s0),uv);
}
