#include "psbc_compile.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    FILE *file=fopen("tests/storage_image.spv","rb");
    assert(file && !fseek(file,0,SEEK_END));
    long length=ftell(file);
    assert(length>0 && !(length&3));
    rewind(file);
    uint32_t *spirv=malloc((size_t)length);
    assert(spirv && fread(spirv,1,(size_t)length,file)==(size_t)length);
    fclose(file);
    PsbcCompileOptions options={.target=PSBC_TARGET_PS5,
        .stage=PSBC_STAGE_COMPUTE,.entrypoint="main",.optimise=true,
        .address32_hi=2,.descriptor_binding_count=2,
        .static_descriptor_use=true};
    options.descriptor_bindings[0]=(PsbcDescriptorBinding){.set=0,.binding=0,
        .type=PSBC_DESCRIPTOR_STORAGE_IMAGE,.array_size=1,.offset=0,.stride=32};
    options.descriptor_bindings[1]=(PsbcDescriptorBinding){.set=0,.binding=1,
        .type=PSBC_DESCRIPTOR_STORAGE_BUFFER,.array_size=1,.offset=32,.stride=16};
    PsbcShaderOutput output={0};
    assert(psbc_compile_shader(spirv,(size_t)length,&options,&output)==PSBC_RESULT_OK);
    assert(output.machine_code && output.machine_code_size &&
        output.metadata.descriptor_used_binding_mask[0]==3);
    psbc_free_output(&output);
    options.descriptor_bindings[0].stride=16;
    assert(psbc_compile_shader(spirv,(size_t)length,&options,&output)==
        PSBC_RESULT_INTERNAL_ERROR);
    free(spirv);
    puts("R32_UINT storage image plus SSBO compile: pass");
    return 0;
}
