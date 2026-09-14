#include "psbc_compile.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv)
{
    assert(argc==2);
    FILE *f=fopen(argv[1],"rb");assert(f);
    assert(!fseek(f,0,SEEK_END));long bytes=ftell(f);assert(bytes>0 && !(bytes&3));
    rewind(f);uint32_t *code=malloc((size_t)bytes);assert(code);
    assert(fread(code,1,(size_t)bytes,f)==(size_t)bytes);fclose(f);
    PsbcCompileOptions options={.target=PSBC_TARGET_PS5,.stage=PSBC_STAGE_FRAGMENT,
        .entrypoint="main",.optimise=true,.address32_hi=2,
        .primitive_type=4,.rasterization_samples=1};
    /* Each live location needs a distinct attribute, including sparse
     * locations. Repeated component loads must reuse that attribute. */
    const uint32_t expected[]={15,16|(1u<<22),20|(1u<<22),24|(1u<<22)};
    for(unsigned last=0;last<2;++last) {
        options.provoking_vtx_last=last;
        PsbcShaderOutput out={0};
        assert(psbc_compile_shader(code,(size_t)bytes,&options,&out)==PSBC_RESULT_OK);
        assert(out.machine_code_size && out.metadata.input_semantic_count==4);
        assert(!(out.metadata.unresolved_fields&PSBC_UNRESOLVED_AGC_LINKAGE));
        for(unsigned i=0;i<4;++i)assert(out.metadata.input_semantics[i]==expected[i]);
        psbc_free_output(&out);
    }
    free(code);
    puts("Fragment input bases: sparse smooth/flat float/int/uint and provoking modes pass");
}
