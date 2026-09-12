#include "psbc_compile.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t *read_spirv(const char *path,size_t *bytes)
{
    FILE *file=fopen(path,"rb");assert(file);
    assert(!fseek(file,0,SEEK_END));long size=ftell(file);assert(size>0 && !(size&3));
    rewind(file);uint32_t *data=malloc((size_t)size);assert(data);
    assert(fread(data,1,(size_t)size,file)==(size_t)size);fclose(file);*bytes=(size_t)size;
    return data;
}

static PsbcShaderOutput compile(const uint32_t *spirv,size_t bytes,uint32_t value)
{
    PsbcCompileOptions options={.target=PSBC_TARGET_PS5,.stage=PSBC_STAGE_COMPUTE,
        .entrypoint="main",.optimise=true,.address32_hi=2,.force_indirect_push_constants=true,
        .descriptor_binding_count=1,.specialization_constant_count=1};
    options.descriptor_bindings[0]=(PsbcDescriptorBinding){.set=0,.binding=0,
        .type=PSBC_DESCRIPTOR_STORAGE_BUFFER,.array_size=1,.offset=0,.stride=16};
    options.specialization_constants[0].constant_id=0;
    options.specialization_constants[0].size=sizeof(value);
    memcpy(options.specialization_constants[0].data,&value,sizeof(value));
    PsbcShaderOutput output={0};assert(psbc_compile_shader(spirv,bytes,&options,&output)==PSBC_RESULT_OK);
    assert(output.metadata.version==PSBC_SHADER_METADATA_VERSION &&
        output.metadata.push_constants_valid && output.metadata.push_constant_size==4 &&
        output.metadata.push_constants_user_data_dword<output.metadata.user_sgpr_count);
    return output;
}

int main(int argc,char **argv)
{
    assert(argc==2);size_t bytes=0;uint32_t *spirv=read_spirv(argv[1],&bytes);
    PsbcShaderOutput a=compile(spirv,bytes,5),b=compile(spirv,bytes,9);
    size_t common=a.machine_code_size<b.machine_code_size?a.machine_code_size:b.machine_code_size;
    assert(a.machine_code_size!=b.machine_code_size || memcmp(a.machine_code,b.machine_code,common));
    psbc_free_output(&a);psbc_free_output(&b);free(spirv);
    puts("Runtime push/specialization parameters: pass");
}
