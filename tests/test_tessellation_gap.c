/* The tessellation stages must not look loadable.
 *
 * A TESS_CTRL or TESS_EVAL compile today produces ISA but no pipeline package:
 * the hull shader is a merged vertex+tessellation-control program in radv and
 * the domain half is a separate stage, and the hull/domain state is programmed
 * through the pipeline's context rolls.  The metadata therefore has to say so
 * with an explicit unresolved field, so a consumer fails closed instead of
 * reading an unclassified hardware stage and guessing at registers.
 */
#include "psbc_compile.h"

#include <assert.h>
#include <stdint.h>
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

static PsbcCompileOptions options_for(PsbcStage stage)
{
    return (PsbcCompileOptions){
        .target = PSBC_TARGET_PS5,
        .stage = stage,
        .entrypoint = "main",
        .optimise = true,
        .primitive_type = 4,
        .rasterization_samples = 1,
        .address32_hi = 2,
    };
}

int main(int argc, char **argv)
{
    assert(argc == 4);
    size_t control_bytes = 0, evaluation_bytes = 0, vertex_bytes = 0;
    uint32_t *control_spirv = read_spirv(argv[1], &control_bytes);
    uint32_t *evaluation_spirv = read_spirv(argv[2], &evaluation_bytes);
    uint32_t *vertex_spirv = read_spirv(argv[3], &vertex_bytes);

    /* The hull half compiles and publishes its own program registers, but the
     * package is still explicitly incomplete. */
    {
        PsbcCompileOptions options = options_for(PSBC_STAGE_TESS_CTRL);
        PsbcShaderOutput output = {0};
        assert(psbc_compile_shader(control_spirv, control_bytes, &options,
                                   &output) == PSBC_RESULT_OK);
        assert((output.metadata.unresolved_fields & PSBC_UNRESOLVED_TESS_PIPELINE) != 0);
        assert(output.metadata.hardware_stage == PSBC_HW_STAGE_UNKNOWN);
        assert(!output.metadata.linkage_valid);
        assert(output.metadata.context_register_count == 0);
        /* Program LO/HI plus RSRC1/RSRC2 of the HS half, for this half only. */
        assert(output.metadata.shader_register_count == 4);
        for (unsigned r = 1; r < output.metadata.shader_register_count; ++r)
            assert(output.metadata.shader_registers[r].offset >
                   output.metadata.shader_registers[r - 1].offset);
        assert(output.metadata.shader_registers[2].value != 0);
        psbc_free_output(&output);
    }

    /* The domain half has no package state at all yet and must publish none. */
    {
        PsbcCompileOptions options = options_for(PSBC_STAGE_TESS_EVAL);
        PsbcShaderOutput output = {0};
        assert(psbc_compile_shader(evaluation_spirv, evaluation_bytes, &options,
                                   &output) == PSBC_RESULT_OK);
        assert((output.metadata.unresolved_fields & PSBC_UNRESOLVED_TESS_PIPELINE) != 0);
        assert(output.metadata.hardware_stage == PSBC_HW_STAGE_UNKNOWN);
        assert(!output.metadata.linkage_valid);
        assert(output.metadata.context_register_count == 0);
        assert(output.metadata.shader_register_count == 0);
        psbc_free_output(&output);
    }

    /* Control: a vertex shader is not a tessellation stage and must not carry
     * the tessellation gap bit. */
    PsbcCompileOptions vertex_options = options_for(PSBC_STAGE_VERTEX);
    vertex_options.ngg = true;
    PsbcShaderOutput vertex = {0};
    assert(psbc_compile_shader(vertex_spirv, vertex_bytes, &vertex_options,
                               &vertex) == PSBC_RESULT_OK);
    assert((vertex.metadata.unresolved_fields & PSBC_UNRESOLVED_TESS_PIPELINE) == 0);
    psbc_free_output(&vertex);

    free(control_spirv);
    free(evaluation_spirv);
    free(vertex_spirv);
    printf("tessellation pipeline gap is declared, not guessed\n");
    return 0;
}
