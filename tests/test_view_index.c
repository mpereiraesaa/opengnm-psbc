/* ViewIndex user-data slot regression (metadata version 14).
 *
 * The multiview built-in gl_ViewIndex is defined per view of a view mask, and
 * delivering it needs a compiler-declared ABI slot the caller can write. RADV
 * already declares ac.view_index at its own user-data location when the stage
 * reads SYSTEM_VALUE_VIEW_INDEX, so this test pins what the compiler reports:
 * the slot exists exactly when the stage really reads the built-in, it lives
 * inside the reported user-SGPR block, and it is its own location rather than a
 * dword shared with the base-vertex block DrawIndex uses.
 *
 * Both vertex and PS5 fragment stages carry a real argument when read.
 * Negative shaders that never read the built-in must report no slot.
 */
#include "psbc_compile.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

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

static PsbcShaderOutput compile(const uint32_t *spirv, size_t bytes, PsbcStage stage)
{
    PsbcCompileOptions options = {
        .target = PSBC_TARGET_PS5, .stage = stage,
        .entrypoint = "main", .optimise = true, .address32_hi = 2,
        .primitive_type = 4, .rasterization_samples = 1,
        .vertex_attribute_count = 1,
    };
    options.vertex_attributes[0] = (PsbcVertexAttribute){
        .location = 0, .binding = 0, .format = PSBC_VERTEX_FORMAT_R32G32B32A32_FLOAT,
        .stride = 16, .alignment = 1};
    PsbcShaderOutput output = {0};
    PsbcResult result = psbc_compile_shader(spirv, bytes, &options, &output);
    if (result != PSBC_RESULT_OK)
        fprintf(stderr, "view-index compile failed: %s\n", psbc_result_string(result));
    assert(result == PSBC_RESULT_OK);
    assert(output.metadata.version == PSBC_SHADER_METADATA_VERSION);
    return output;
}

static void check_slot_is_declared(void)
{
    size_t bytes = 0;
    uint32_t *spirv = read_spirv("tests/view-index.spv", &bytes);
    PsbcShaderOutput output = compile(spirv, bytes, PSBC_STAGE_VERTEX);

    /* The built-in is read, so the slot must be reported and inside the
     * user-SGPR block the caller writes. */
    assert(output.metadata.view_index_valid);
    assert(output.metadata.view_index_user_data_dword < output.metadata.user_sgpr_count);
    /* A vertex shader always declares the base-vertex argument, and the view
     * index is its own user-data location, so the two slots must be distinct:
     * a compiler that folded the view index into the base-vertex block would
     * report the same dword for both, and a caller writing one would clobber
     * the other. This shader reads neither DrawIndex nor BaseInstance. */
    assert(output.metadata.base_vertex_valid);
    assert(output.metadata.view_index_user_data_dword !=
           output.metadata.base_vertex_user_data_dword);
    assert(!output.metadata.draw_id_valid);
    assert(!output.metadata.start_instance_valid);
    assert(!output.metadata.descriptor_set_valid[0]);

    psbc_free_output(&output);
    free(spirv);
}

static void check_slot_stays_absent(void)
{
    size_t bytes = 0;
    uint32_t *spirv = read_spirv("tests/tri.spv", &bytes);
    PsbcShaderOutput output = compile(spirv, bytes, PSBC_STAGE_VERTEX);

    /* A stage that never reads ViewIndex must not be handed a slot, so a
     * consumer can tell "the compiler placed it" from "the value means nothing
     * here". */
    assert(!output.metadata.view_index_valid);
    assert(!output.metadata.draw_id_valid);
    assert(!output.metadata.start_instance_valid);

    psbc_free_output(&output);
    free(spirv);
}

static void check_fragment_slot(void)
{
    size_t bytes = 0;
    uint32_t *spirv = read_spirv("tests/view-index.frag.spv", &bytes);
    PsbcShaderOutput output = compile(spirv, bytes, PSBC_STAGE_FRAGMENT);
    assert(output.metadata.view_index_valid);
    assert(output.metadata.view_index_user_data_dword < output.metadata.user_sgpr_count);
    assert(output.metadata.hardware_stage == PSBC_HW_STAGE_PIXEL);
    assert(!output.metadata.base_vertex_valid && !output.metadata.draw_id_valid);
    psbc_free_output(&output);
    free(spirv);

    spirv = read_spirv("tests/tri.frag.spv", &bytes);
    output = compile(spirv, bytes, PSBC_STAGE_FRAGMENT);
    assert(!output.metadata.view_index_valid);
    assert(output.metadata.view_index_user_data_dword == 0);
    psbc_free_output(&output);
    free(spirv);
}

int main(void)
{
    check_slot_is_declared();
    check_slot_stays_absent();
    check_fragment_slot();
    puts("ViewIndex user-data slot: declared only when read, at its own location in the user-SGPR block");
    return 0;
}
