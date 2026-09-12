#include "psbc_compile.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t *read_spirv(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb"); assert(file);
    assert(!fseek(file, 0, SEEK_END)); long size = ftell(file);
    assert(size > 0 && !(size & 3)); rewind(file);
    uint32_t *data = malloc((size_t)size); assert(data);
    assert(fread(data, 1, (size_t)size, file) == (size_t)size);
    fclose(file); *bytes = (size_t)size; return data;
}

static PsbcCompileOptions options(void)
{
    PsbcCompileOptions out = {.target=PSBC_TARGET_PS5,
        .stage=PSBC_STAGE_COMPUTE,.entrypoint="main",.optimise=true,
        .address32_hi=2,.descriptor_binding_count=2};
    out.descriptor_bindings[0] = (PsbcDescriptorBinding){.set=0,.binding=0,
        .type=PSBC_DESCRIPTOR_STORAGE_BUFFER,.array_size=1,.stride=16};
    out.descriptor_bindings[1] = (PsbcDescriptorBinding){.set=0,.binding=1,
        .type=PSBC_DESCRIPTOR_STORAGE_BUFFER,.array_size=1,.offset=16,.stride=16};
    return out;
}

static void check(const char *path, bool eight)
{
    size_t bytes = 0; uint32_t *spirv = read_spirv(path, &bytes);
    PsbcCompileOptions opts = options(); PsbcShaderOutput output = {0};
    assert(psbc_compile_shader(spirv, bytes - 1, &opts, &output) ==
           PSBC_RESULT_INVALID_SPIRV);
    const uint32_t first_instruction = spirv[5];
    spirv[5] &= UINT32_C(0xffff);
    assert(psbc_compile_shader(spirv, bytes, &opts, &output) ==
           PSBC_RESULT_INVALID_SPIRV);
    spirv[5] = first_instruction;
    assert(psbc_compile_shader(spirv, bytes, &opts, &output) ==
           PSBC_RESULT_UNSUPPORTED_CAPABILITY);
    if (eight) {
        opts.enable_uniform_and_storage_buffer_8bit_access = true;
        assert(psbc_compile_shader(spirv, bytes, &opts, &output) ==
               PSBC_RESULT_INTERNAL_ERROR);
        opts.enable_uniform_and_storage_buffer_8bit_access = false;
        opts.enable_storage_buffer_8bit_access = true;
    } else {
        opts.enable_uniform_and_storage_buffer_16bit_access = true;
        assert(psbc_compile_shader(spirv, bytes, &opts, &output) ==
               PSBC_RESULT_INTERNAL_ERROR);
        opts.enable_uniform_and_storage_buffer_16bit_access = false;
        opts.enable_storage_buffer_16bit_access = true;
    }
    assert(psbc_compile_shader(spirv, bytes, &opts, &output) == PSBC_RESULT_OK);
    assert(output.machine_code && output.machine_code_size);
    psbc_free_output(&output); free(spirv);
}

int main(void)
{
    check("tests/storage8.spv", true);
    check("tests/storage16.spv", false);
    puts("Narrow-storage capability gates and GFX1013 lowering: pass");
}
