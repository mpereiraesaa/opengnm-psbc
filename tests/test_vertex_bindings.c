#include "psbc_compile.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv)
{
    assert(argc==2);
    FILE *f=fopen(argv[1],"rb");assert(f);
    assert(!fseek(f,0,SEEK_END));long length=ftell(f);assert(length>0 && !(length&3));
    rewind(f);uint32_t *words=malloc((size_t)length);assert(words);
    assert(fread(words,1,(size_t)length,f)==(size_t)length);fclose(f);
    PsbcCompileOptions options={.target=PSBC_TARGET_PS5,.stage=PSBC_STAGE_VERTEX,
        .entrypoint="main",.optimise=true,.address32_hi=2,.ngg=true,
        .omit_implicit_primitive_id=true,.primitive_type=4,.rasterization_samples=1,
        .vertex_attribute_count=16};
    for(unsigned i=0;i<16;++i)
        options.vertex_attributes[i]=(PsbcVertexAttribute){.location=i,.binding=15-i,
            .format=PSBC_VERTEX_FORMAT_R32G32B32A32_FLOAT,.stride=16,.alignment=1};
    for(unsigned variant=0;variant<2;++variant) {
        options.specialization_constant_count=variant;
        options.specialization_constants[0].constant_id=0;
        options.specialization_constants[0].size=4;
        PsbcShaderOutput output={0};
        assert(psbc_compile_shader(words,(size_t)length,&options,&output)==PSBC_RESULT_OK);
        assert(output.metadata.version==PSBC_SHADER_METADATA_VERSION && output.metadata.vertex_buffer_table_valid);
        assert(output.metadata.vertex_buffer_usage_mask==(variant?0x8000u:0xffffu));
        assert(!output.metadata.vertex_buffer_per_attribute);
        assert(output.metadata.vertex_buffer_table_user_data_dword<output.metadata.user_sgpr_count);
        psbc_free_output(&output);
    }
    free(words);
    puts("Vertex metadata: 16 bindings and specialization-pruned binding 15 pass");
}
