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
#include <string.h>

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

    /* VGT_TF_PARAM: one context register at the register's own offset
     * ((0x028B6C - 0x28000) / 4, SI_CONTEXT_REG_OFFSET), whose value is derived
     * from the interface the control shader declares.  Note that a stand-alone
     * control shader does not carry the domain - that comes from the evaluation
     * shader - so this value is the derivation's default here, which is one more
     * reason the package cannot be complete without the pipeline's other half. */
    const uint16_t tf_param_offset = (uint16_t)((0x028B6Cu - 0x28000u) / 4u);

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
        assert(output.metadata.context_register_count == 1);
        assert(output.metadata.context_registers[0].offset == tf_param_offset);
        /* Program LO/HI plus RSRC1/RSRC2 of the HS half, for this half only. */
        assert(output.metadata.shader_register_count == 4);
        for (unsigned r = 1; r < output.metadata.shader_register_count; ++r)
            assert(output.metadata.shader_registers[r].offset >
                   output.metadata.shader_registers[r - 1].offset);
        assert(output.metadata.shader_registers[2].value != 0);
        assert(output.metadata.context_registers[0].value != 0);
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

    /* The pair entry point compiles both halves, publishes the LS half, and
     * combines the HS resource registers instead of copying one half's. */
    {
        PsbcCompileOptions options = options_for(PSBC_STAGE_TESS_CTRL);
        PsbcShaderOutput pair = {0};
        assert(psbc_compile_tess_pipeline(vertex_spirv, vertex_bytes,
                                          control_spirv, control_bytes,
                                          &options, &pair) == PSBC_RESULT_OK);
        assert(pair.metadata.hull_ls_valid);
        assert(pair.metadata.hull_ls_code_size > 0);
        assert(pair.metadata.hull_ls_code_offset > 0);
        assert(pair.metadata.hull_ls_rsrc1.value != 0);
        assert(pair.metadata.hull_ls_rsrc2.value != 0);
        /* Still not a loadable package: the LS code and the hull state the
         * pipeline owns are not in it. */
        assert((pair.metadata.unresolved_fields & PSBC_UNRESOLVED_TESS_PIPELINE) != 0);
        assert(pair.metadata.shader_register_count == 4);
        {
            PsbcCompileOptions control_options = options_for(PSBC_STAGE_TESS_CTRL);
            PsbcShaderOutput control = {0};
            assert(psbc_compile_shader(control_spirv, control_bytes,
                                       &control_options, &control) == PSBC_RESULT_OK);
            assert((pair.metadata.shader_registers[2].value & 0x3Fu) >=
                   (control.metadata.shader_registers[2].value & 0x3Fu));
            /* The packaged buffer is [HS code][LS code]: the HS half keeps the
             * bytes a single-program consumer would read, and the LS half sits
             * exactly where the metadata says. */
            assert(pair.machine_code_size ==
                   control.machine_code_size + pair.metadata.hull_ls_code_size);
            assert(pair.metadata.hull_ls_code_offset == control.machine_code_size);
            assert(memcmp(pair.machine_code, control.machine_code,
                          control.machine_code_size) == 0);
            psbc_free_output(&control);
        }
        {
            PsbcCompileOptions ls_options = options_for(PSBC_STAGE_VERTEX);
            PsbcShaderOutput ls = {0};
            assert(psbc_compile_shader(vertex_spirv, vertex_bytes, &ls_options,
                                       &ls) == PSBC_RESULT_OK);
            assert(ls.machine_code_size == pair.metadata.hull_ls_code_size);
            assert(memcmp((const uint8_t *)pair.machine_code +
                              pair.metadata.hull_ls_code_offset,
                          ls.machine_code, ls.machine_code_size) == 0);
            psbc_free_output(&ls);
        }
        psbc_free_output(&pair);
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
