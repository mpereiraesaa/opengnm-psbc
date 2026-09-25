#include "psbc_compile.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t *read_spirv(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb");
    assert(file && !fseek(file, 0, SEEK_END));
    long size = ftell(file);
    assert(size > 0 && !(size & 3));
    rewind(file);
    uint32_t *data = malloc((size_t)size);
    assert(data && fread(data, 1, (size_t)size, file) == (size_t)size);
    fclose(file);
    *bytes = (size_t)size;
    return data;
}

/* Separate SAMPLER (16-byte S#) and SAMPLED_IMAGE (32-byte T#) records, the
 * two texel-buffer V# records, all in one set table:
 *   binding 0 sampler        offset  0 stride 16
 *   binding 1 sampled image  offset 16 stride 32
 *   binding 2 uniform texel  offset 48 stride 16
 *   binding 3 storage texel  offset 64 stride 16 */
static void compute(const char *path)
{
    size_t bytes;
    uint32_t *spirv = read_spirv(path, &bytes);
    PsbcCompileOptions options = {.target = PSBC_TARGET_PS5,
        .stage = PSBC_STAGE_COMPUTE, .entrypoint = "main", .optimise = true,
        .address32_hi = 2, .descriptor_binding_count = 4,
        .static_descriptor_use = true};
    options.descriptor_bindings[0] = (PsbcDescriptorBinding){.set = 0, .binding = 0,
        .type = PSBC_DESCRIPTOR_SAMPLER, .array_size = 1, .offset = 0, .stride = 16};
    options.descriptor_bindings[1] = (PsbcDescriptorBinding){.set = 0, .binding = 1,
        .type = PSBC_DESCRIPTOR_SAMPLED_IMAGE, .array_size = 1, .offset = 16, .stride = 32};
    options.descriptor_bindings[2] = (PsbcDescriptorBinding){.set = 0, .binding = 2,
        .type = PSBC_DESCRIPTOR_UNIFORM_TEXEL_BUFFER, .array_size = 1, .offset = 48, .stride = 16};
    options.descriptor_bindings[3] = (PsbcDescriptorBinding){.set = 0, .binding = 3,
        .type = PSBC_DESCRIPTOR_STORAGE_TEXEL_BUFFER, .array_size = 1, .offset = 64, .stride = 16};
    PsbcShaderOutput output = {0};
    assert(psbc_compile_shader(spirv, bytes, &options, &output) == PSBC_RESULT_OK);
    assert(output.machine_code && output.machine_code_size &&
           output.metadata.descriptor_set_valid[0] &&
           output.metadata.descriptor_used_binding_mask[0] == 0xf);
    psbc_free_output(&output);
    /* A record declared with the wrong width is refused, never reinterpreted. */
    const uint32_t wrong[4] = {48, 48, 32, 32};
    for (unsigned i = 0; i < 4; ++i) {
        PsbcCompileOptions bad = options;
        bad.descriptor_bindings[i].stride = wrong[i];
        assert(psbc_compile_shader(spirv, bytes, &bad, &output) == PSBC_RESULT_INTERNAL_ERROR);
    }
    free(spirv);
}

static void fragment(const char *path)
{
    size_t bytes;
    uint32_t *spirv = read_spirv(path, &bytes);
    PsbcCompileOptions options = {.target = PSBC_TARGET_PS5,
        .stage = PSBC_STAGE_FRAGMENT, .entrypoint = "main", .optimise = true,
        .address32_hi = 2, .primitive_type = 4, .rasterization_samples = 1,
        .descriptor_binding_count = 2, .static_descriptor_use = true};
    options.descriptor_bindings[0] = (PsbcDescriptorBinding){.set = 0, .binding = 0,
        .type = PSBC_DESCRIPTOR_SAMPLER, .array_size = 1, .offset = 0, .stride = 16};
    options.descriptor_bindings[1] = (PsbcDescriptorBinding){.set = 0, .binding = 1,
        .type = PSBC_DESCRIPTOR_SAMPLED_IMAGE, .array_size = 1, .offset = 16, .stride = 32};
    PsbcShaderOutput output = {0};
    assert(psbc_compile_shader(spirv, bytes, &options, &output) == PSBC_RESULT_OK);
    assert(output.machine_code && output.machine_code_size &&
           output.metadata.descriptor_used_binding_mask[0] == 0x3);
    psbc_free_output(&output);
    free(spirv);
}

int main(int argc, char **argv)
{
    assert(argc == 3);
    compute(argv[1]);
    fragment(argv[2]);
    puts("Separate sampler, sampled image and texel buffer compile: pass");
    return 0;
}
