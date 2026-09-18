/* Merged vertex+geometry metadata contract.
 *
 * A merged VS+GS program occupies the same NGG hardware slot as a vertex-only
 * one, and the AGC linked block the PS5 runtime builds carries no geometry
 * variant, so a consumer cannot tell the two apart unless the compiler says so.
 * These assertions pin the three things the native adapter needs and pin the
 * difference between "the caller supplied device facts" and "nobody knows the
 * CU/PC-line topology, so the GE allocation stays invalid instead of guessed".
 */
#include "psbc_compile.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GE_PC_ALLOC_UC_OFFSET 0x260u /* (0x030980 - 0x030000) / 4 */

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

static PsbcCompileOptions base_options(void)
{
    return (PsbcCompileOptions){
        .target = PSBC_TARGET_PS5,
        .stage = PSBC_STAGE_GEOMETRY,
        .entrypoint = "main",
        .optimise = true,
        .ngg = true,
        .primitive_type = 4,
        .rasterization_samples = 1,
        .address32_hi = 2,
    };
}

int main(int argc, char **argv)
{
    assert(argc == 3);
    size_t vertex_bytes = 0, geometry_bytes = 0;
    uint32_t *vertex = read_spirv(argv[1], &vertex_bytes);
    uint32_t *geometry = read_spirv(argv[2], &geometry_bytes);

    /* Without device facts: the merged pair is identified, its ES half is
     * described, and no GE allocation is invented. */
    PsbcCompileOptions options = base_options();
    PsbcShaderOutput merged = {0};
    assert(psbc_compile_geometry_pipeline(vertex, vertex_bytes, geometry,
                                          geometry_bytes, &options,
                                          &merged) == PSBC_RESULT_OK);
    assert(merged.metadata.version == PSBC_SHADER_METADATA_VERSION);
    assert(merged.metadata.merged_geometry);
    assert(merged.metadata.merged_es_itemsize > 0);
    assert(merged.metadata.merged_esgs_ring_itemsize > 0);
    /* The two ES-half sizes are the same quantity in the two units radv uses. */
    assert(merged.metadata.merged_esgs_ring_itemsize * 4 ==
           merged.metadata.merged_es_itemsize);
    assert(!merged.metadata.linkage_ge_pc_alloc_valid);
    assert(merged.metadata.linkage_ge_pc_alloc.offset == 0);
    assert(merged.metadata.linkage_ge_pc_alloc.value == 0);
    psbc_free_output(&merged);

    /* With the caller's SA/CU and PC-line facts: the same pair now also carries
     * the GE PC-line allocation radv programs for every NGG pipeline. */
    options.ngg_device_facts = true;
    options.ngg_pc_lines = 1024;          /* ac_gpu_info.c GFX10 value */
    options.ngg_min_good_cu_per_sa = 8;
    PsbcShaderOutput with_facts = {0};
    assert(psbc_compile_geometry_pipeline(vertex, vertex_bytes, geometry,
                                          geometry_bytes, &options,
                                          &with_facts) == PSBC_RESULT_OK);
    assert(with_facts.metadata.merged_geometry);
    assert(with_facts.metadata.linkage_ge_pc_alloc_valid);
    assert(with_facts.metadata.linkage_ge_pc_alloc.offset == GE_PC_ALLOC_UC_OFFSET);
    /* pc_lines / 4 = 256 oversubscribed lines with OVERSUB_EN set. */
    assert(with_facts.metadata.linkage_ge_pc_alloc.value == 0x1ffu);
    psbc_free_output(&with_facts);

    /* Control: the vertex shader alone is not a merged program and publishes no
     * ES-half facts, even though it does get the GE allocation. */
    options.stage = PSBC_STAGE_VERTEX;
    PsbcShaderOutput vertex_only = {0};
    assert(psbc_compile_shader(vertex, vertex_bytes, &options,
                               &vertex_only) == PSBC_RESULT_OK);
    assert(!vertex_only.metadata.merged_geometry);
    assert(vertex_only.metadata.merged_es_itemsize == 0);
    assert(vertex_only.metadata.merged_esgs_ring_itemsize == 0);
    assert(vertex_only.metadata.linkage_ge_pc_alloc_valid);
    psbc_free_output(&vertex_only);

    free(vertex);
    free(geometry);
    printf("merged geometry metadata contract holds\n");
    return 0;
}
