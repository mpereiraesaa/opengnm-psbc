#version 450
layout(location=0) in vec2 smooth_value;
layout(location=1) flat in float flat_float;
layout(location=5) flat in int flat_int;
layout(location=9) flat in uvec2 flat_uint;
layout(location=0) out vec4 output_color;
void main()
{
    output_color=vec4(flat_float+smooth_value.x*0.125,
        float(flat_int)*0.125+smooth_value.y*0.125,
        float(flat_uint.x+flat_uint.y)*0.0625,1);
}
