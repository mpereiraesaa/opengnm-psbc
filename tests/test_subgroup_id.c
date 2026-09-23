#include "psbc_compile.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void compile(const char *path)
{
    FILE *file = fopen(path, "rb");
    assert(file);
    assert(!fseek(file, 0, SEEK_END));
    long length = ftell(file);
    assert(length > 0 && !(length & 3));
    rewind(file);
    uint32_t *spirv = malloc((size_t)length);
    assert(spirv);
    assert(fread(spirv, 1, (size_t)length, file) == (size_t)length);
    fclose(file);

    PsbcCompileOptions options = {
        .target = PSBC_TARGET_PS5,
        .stage = PSBC_STAGE_COMPUTE,
        .entrypoint = "main",
        .optimise = true,
        .address32_hi = 2,
        .descriptor_binding_count = 1,
    };
    options.descriptor_bindings[0] = (PsbcDescriptorBinding){
        .set = 0, .binding = 0, .type = PSBC_DESCRIPTOR_STORAGE_BUFFER,
        .array_size = 1, .offset = 0, .stride = 16,
    };
    PsbcShaderOutput output = {0};
    assert(psbc_compile_shader(spirv, (size_t)length, &options, &output) ==
           PSBC_RESULT_OK);
    assert(output.machine_code && output.machine_code_size);
    assert(output.metadata.version == PSBC_SHADER_METADATA_VERSION);
    assert(output.metadata.descriptor_binding_count == 1);
    psbc_free_output(&output);
    free(spirv);
}

int main(int argc, char **argv)
{
    assert(argc == 4);
    compile(argv[1]);
    compile(argv[2]);
    compile(argv[3]);
    puts("PS5 compute subgroup ID lowering: pass (1D, 2D and 3D host compile)");
    return 0;
}
