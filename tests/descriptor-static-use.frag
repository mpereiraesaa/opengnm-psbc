#version 450
/* Static descriptor-use fixture.
 *
 * Set 0 really uses binding 0 (a uniform block) and binding 2 (a sampler array
 * indexed by a constant, so the whole array is the addressable extent). Set 0
 * also declares binding 5 and binding 7, which the shader never dereferences,
 * and sets 1 and 2 exist only in the pipeline layout the caller passes. Sparse
 * binding numbers are deliberate: the reported use is per (set, binding), not a
 * dense count.
 *
 * Binding 9 is reached only through a specialization constant. The
 * specialization map is applied before optimization, so the optimizer can prove
 * the sampler dead for one specialization and live for the other; the reported
 * use must follow the compiled specialization rather than the declaration.
 */
layout(constant_id = 0) const int use_optional_texture = 0;

layout(set = 0, binding = 0) uniform UsedBlock { vec4 color; } used_block;
layout(set = 0, binding = 2) uniform sampler2D sampled_array[4];
layout(set = 0, binding = 5) uniform UnusedBlock { vec4 tint; } unused_block;
layout(set = 0, binding = 7) uniform sampler2D unused_sampler;
layout(set = 0, binding = 9) uniform sampler2D optional_texture;
layout(set = 1, binding = 0) uniform sampler2D also_unused_sampler;
layout(set = 2, binding = 0) uniform AnotherUnused { vec4 shade; } another_unused;

layout(location = 0) out vec4 out_color;

void main()
{
    out_color = used_block.color + texture(sampled_array[1], vec2(0.5));
    if (use_optional_texture != 0)
        out_color += texture(optional_texture, vec2(0.5));
}
