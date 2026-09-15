/* DrawIndex user-data slot regression (metadata version 13).
 *
 * The pinned Vulkan specification defines DrawIndex as zero for every direct
 * draw and for a single indirect draw, and the pinned CTS shader used by the
 * base_vertex/base_instance cases requires gl_DrawID == 0 in cases that do not
 * require multiDrawIndirect. Delivering that value needs a compiler-declared
 * slot, which is what this test pins: the slot exists exactly when the stage
 * reads the built-in, it lives inside the reported user-SGPR block, and it is
 * distinct from the base-vertex slot.
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

static PsbcShaderOutput compile(const uint32_t *spirv, size_t bytes)
{
    PsbcCompileOptions options = {
        .target = PSBC_TARGET_PS5, .stage = PSBC_STAGE_VERTEX,
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
        fprintf(stderr, "draw-id compile failed: %s\n", psbc_result_string(result));
    assert(result == PSBC_RESULT_OK);
    assert(output.metadata.version == PSBC_SHADER_METADATA_VERSION);
    return output;
}

static void check_slot_is_declared(void)
{
    size_t bytes = 0;
    uint32_t *spirv = read_spirv("tests/draw-id.spv", &bytes);
    PsbcShaderOutput output = compile(spirv, bytes);

    /* The built-in is read, so the slot must be reported, inside the user-SGPR
     * block the caller writes, and next to - not on top of - the base-vertex
     * slot that the same block already carried. */
    assert(output.metadata.draw_id_valid);
    assert(output.metadata.draw_id_user_data_dword < output.metadata.user_sgpr_count);
    assert(output.metadata.base_vertex_valid);
    assert(output.metadata.base_vertex_user_data_dword < output.metadata.user_sgpr_count);
    assert(output.metadata.draw_id_user_data_dword !=
           output.metadata.base_vertex_user_data_dword);
    /* RADV declares base_vertex first and draw_id immediately after it in that
     * block, so the reported offsets differ by exactly one dword; an
     * implementation that reported a different span would be describing a
     * different block than the compiler wrote. */
    assert(output.metadata.draw_id_user_data_dword ==
           output.metadata.base_vertex_user_data_dword + 1u);
    /* This shader reads neither BaseInstance nor a descriptor table. */
    assert(!output.metadata.start_instance_valid);
    assert(!output.metadata.descriptor_set_valid[0]);

    psbc_free_output(&output);
    free(spirv);
}

static void check_slot_stays_absent(void)
{
    size_t bytes = 0;
    uint32_t *spirv = read_spirv("tests/tri.spv", &bytes);
    PsbcShaderOutput output = compile(spirv, bytes);

    /* A stage that never reads DrawIndex must not be handed a slot, so a
     * consumer can tell "the compiler placed it" from "the value means
     * nothing here". */
    assert(!output.metadata.draw_id_valid);
    assert(!output.metadata.start_instance_valid);

    psbc_free_output(&output);
    free(spirv);
}

int main(void)
{
    check_slot_is_declared();
    check_slot_stays_absent();
    puts("DrawIndex user-data slot: declared only when read, inside the user-SGPR block");
    return 0;
}
