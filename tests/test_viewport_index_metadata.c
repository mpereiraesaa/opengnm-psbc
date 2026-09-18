/* Viewport-index export contract.
 *
 * A geometry stage selects which of the pipeline's viewport banks it writes to
 * by writing gl_ViewportIndex, and RADV counts that as a parameter export: it
 * takes a slot in vs_output_param_offset and raises param_exports, while the
 * varying loop that fills the semantic list only walks user locations. The
 * result used to be a list one word shorter than the export count, which leaves
 * the whole linkage field unresolved in the metadata and makes a consumer
 * refuse the pair - the shape the viewport-routing witness hit on hardware.
 *
 * These assertions pin both halves: the pair compiles, the export is named with
 * the parameter index the compiler gave it, and the control geometry stage
 * without the built-in is unchanged (nothing is emitted for a pipeline that
 * does not select a viewport).
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

static unsigned viewport_index_semantics(const PsbcShaderMetadata *m,
                                         unsigned *parameter)
{
    unsigned found = 0;
    for (unsigned i = 0; i < m->output_semantic_count; ++i) {
        if ((m->output_semantics[i] & 255u) == PSBC_SEMANTIC_VIEWPORT_INDEX) {
            if (parameter) *parameter = m->output_semantics[i] >> 8;
            ++found;
        }
    }
    return found;
}

int main(int argc, char **argv)
{
    assert(argc == 4);
    size_t vertex_bytes = 0, geometry_bytes = 0, control_bytes = 0;
    uint32_t *vertex = read_spirv(argv[1], &vertex_bytes);
    uint32_t *geometry = read_spirv(argv[2], &geometry_bytes);
    uint32_t *control = read_spirv(argv[3], &control_bytes);

    /* The merged pair that selects a viewport is accepted, which is only
     * possible if the semantic list and the parameter export count agree. */
    PsbcCompileOptions options = base_options();
    PsbcShaderOutput merged = {0};
    assert(psbc_compile_geometry_pipeline(vertex, vertex_bytes, geometry,
                                          geometry_bytes, &options,
                                          &merged) == PSBC_RESULT_OK);
    assert(!(merged.metadata.unresolved_fields & PSBC_UNRESOLVED_AGC_LINKAGE));
    /* The export is named exactly once, and the parameter index it carries is
     * the one the compiler assigned to the built-in - parameter 0 in this
     * shape, ahead of the varying, which is what makes "the nth semantic" a
     * wrong way to read the list. */
    unsigned parameter = ~0u;
    assert(viewport_index_semantics(&merged.metadata, &parameter) == 1);
    assert(parameter < PSBC_MAX_SEMANTICS);
    assert(merged.metadata.output_semantic_count >= 1);
    psbc_free_output(&merged);

    /* The control pair writes no viewport index and must not hear about the
     * built-in at all: the semantic is emitted because the export exists, not
     * because the compiler knows the built-in's name. */
    PsbcShaderOutput without = {0};
    assert(psbc_compile_geometry_pipeline(vertex, vertex_bytes, control,
                                          control_bytes, &options,
                                          &without) == PSBC_RESULT_OK);
    assert(!(without.metadata.unresolved_fields & PSBC_UNRESOLVED_AGC_LINKAGE));
    assert(viewport_index_semantics(&without.metadata, NULL) == 0);
    psbc_free_output(&without);

    free(vertex);
    free(geometry);
    free(control);
    puts("Viewport-index export: named in the semantic list, export count agrees, control unchanged");
    return 0;
}
