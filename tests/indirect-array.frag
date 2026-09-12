#version 450

layout(location = 0) out vec4 frag_color;

void main()
{
    vec4 values[4];
    int index = int(gl_FragCoord.x) & 3;
    values[index] = vec4(1.0);
    frag_color = values[index];
}
