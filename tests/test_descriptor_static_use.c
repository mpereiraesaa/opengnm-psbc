#include "psbc_compile.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BIT(set, binding) (1ull << (binding))

static uint32_t *read_spirv(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb");
    assert(file);
    assert(!fseek(file, 0, SEEK_END));
    long size = ftell(file);
    assert(size > 0 && !(size & 3));
    rewind(file);
    uint32_t *data = malloc((size_t)size);
    assert(data);
    assert(fread(data, 1, (size_t)size, file) == (size_t)size);
    fclose(file);
    *bytes = (size_t)size;
    return data;
}

/* One four-set layout signature: set 0 declares a uniform block at binding 0,
 * a sampler array at binding 2, an unused block at binding 5, an unused sampler
 * at binding 7 and a specialization-gated sampler at binding 9; sets 1 and 2
 * declare one sampler each. */
static PsbcCompileOptions make_options(bool static_use)
{
    PsbcCompileOptions options = {
        .target = PSBC_TARGET_PS5, .stage = PSBC_STAGE_FRAGMENT,
        .entrypoint = "main", .optimise = true, .address32_hi = 2,
        .primitive_type = 4, .rasterization_samples = 1,
        .static_descriptor_use = static_use,
    };
    const struct {
        uint8_t set, binding;
        PsbcDescriptorType type;
        uint32_t array_size, offset, stride;
    } declared[] = {
        {0, 0, PSBC_DESCRIPTOR_UNIFORM_BUFFER, 1, 0, 16},
        {0, 2, PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER, 4, 0, 48},
        {0, 5, PSBC_DESCRIPTOR_UNIFORM_BUFFER, 1, 0, 16},
        {0, 7, PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER, 1, 0, 48},
        {0, 9, PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER, 1, 0, 48},
        {1, 0, PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER, 1, 0, 48},
        {2, 0, PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER, 1, 0, 48},
    };
    options.descriptor_binding_count = (uint32_t)(sizeof(declared) / sizeof(declared[0]));
    for (uint32_t i = 0; i < options.descriptor_binding_count; ++i) {
        options.descriptor_bindings[i] = (PsbcDescriptorBinding){
            .set = declared[i].set, .binding = declared[i].binding,
            .type = declared[i].type, .array_size = declared[i].array_size,
            .offset = declared[i].offset, .stride = declared[i].stride};
    }
    return options;
}

static PsbcShaderOutput compile(const uint32_t *spirv, size_t bytes,
                                PsbcCompileOptions *options)
{
    PsbcShaderOutput output = {0};
    PsbcResult result = psbc_compile_shader(spirv, bytes, options, &output);
    if (result != PSBC_RESULT_OK)
        fprintf(stderr, "descriptor static-use compile failed: %s\n",
                psbc_result_string(result));
    assert(result == PSBC_RESULT_OK);
    assert(output.metadata.version == PSBC_SHADER_METADATA_VERSION);
    assert(output.metadata.descriptor_binding_count ==
           options->descriptor_binding_count);
    return output;
}

static void check_declared_but_unused(void)
{
    size_t bytes = 0;
    uint32_t *spirv = read_spirv("tests/descriptor-static-use.spv", &bytes);

    PsbcCompileOptions static_options = make_options(true);
    PsbcShaderOutput used = compile(spirv, bytes, &static_options);

    /* Only set 0 is dereferenced, and inside it only the block at binding 0 and
     * the sampler array at binding 2 for the default specialization, which
     * proves the gated sampler at binding 9 dead. */
    assert(used.metadata.descriptor_set_valid[0]);
    assert(!used.metadata.descriptor_set_valid[1]);
    assert(!used.metadata.descriptor_set_valid[2]);
    assert(!used.metadata.descriptor_set_valid[3]);
    assert(used.metadata.descriptor_set_user_data_dword[0] <
           used.metadata.user_sgpr_count);
    assert(used.metadata.descriptor_used_binding_mask[0] ==
           (BIT(0, 0) | BIT(0, 2)));
    for (uint32_t set = 1; set < PSBC_MAX_DESCRIPTOR_SETS; ++set)
        assert(used.metadata.descriptor_used_binding_mask[set] == 0);

    /* The same source with the gated specialization enabled admits the sampler:
     * static use follows the optimized NIR, not the declaration. */
    int32_t enable = 1;
    PsbcCompileOptions enabled_options = make_options(true);
    enabled_options.specialization_constant_count = 1;
    enabled_options.specialization_constants[0].constant_id = 0;
    enabled_options.specialization_constants[0].size = sizeof(enable);
    memcpy(enabled_options.specialization_constants[0].data, &enable,
           sizeof(enable));
    PsbcShaderOutput enabled = compile(spirv, bytes, &enabled_options);
    assert(enabled.metadata.descriptor_set_valid[0]);
    assert(!enabled.metadata.descriptor_set_valid[1]);
    assert(enabled.metadata.descriptor_used_binding_mask[0] ==
           (BIT(0, 0) | BIT(0, 2) | BIT(0, 9)));

    /* The legacy layout fallback stays available and conservative: it reports
     * every declared binding of every declared set. */
    PsbcCompileOptions legacy_options = make_options(false);
    PsbcShaderOutput legacy = compile(spirv, bytes, &legacy_options);
    for (uint32_t set = 0; set < 3; ++set)
        assert(legacy.metadata.descriptor_set_valid[set]);
    assert(legacy.metadata.descriptor_used_binding_mask[0] ==
           (BIT(0, 0) | BIT(0, 2) | BIT(0, 5) | BIT(0, 7) | BIT(0, 9)));
    assert(legacy.metadata.descriptor_used_binding_mask[1] == BIT(1, 0));
    assert(legacy.metadata.descriptor_used_binding_mask[2] == BIT(2, 0));

    psbc_free_output(&used);
    psbc_free_output(&enabled);
    psbc_free_output(&legacy);
    free(spirv);
}
int main(void)
{
    check_declared_but_unused();
    puts("Descriptor static use: pass");
    return 0;
}
