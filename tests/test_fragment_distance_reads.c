/* A pixel stage that reads the distance built-ins must say so.
 *
 * The pre-raster masks describe what the vertex half exports; these fields
 * describe what the fragment half would have to receive, so a consumer that
 * cannot route distances to the pixel stage can refuse precisely instead of
 * discovering it at draw time.
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

static void compile_fragment(const char *path, PsbcShaderMetadata *out)
{
    size_t bytes = 0;
    uint32_t *words = read_spirv(path, &bytes);
    PsbcCompileOptions options = {
        .target = PSBC_TARGET_PS5,
        .stage = PSBC_STAGE_FRAGMENT,
        .entrypoint = "main",
        .optimise = true,
        .primitive_type = 4,
        .rasterization_samples = 1,
        .address32_hi = 2,
    };
    PsbcShaderOutput output = {0};
    assert(psbc_compile_shader(words, bytes, &options, &output) == PSBC_RESULT_OK);
    *out = output.metadata;
    /* The code is not needed here; free the rest of the output. */
    output.metadata = (PsbcShaderMetadata){0};
    psbc_free_output(&output);
    free(words);
}

int main(int argc, char **argv)
{
    assert(argc == 4);
    PsbcShaderMetadata reading, plain, second;
    compile_fragment(argv[1], &reading);
    assert(reading.version == PSBC_SHADER_METADATA_VERSION);
    assert(reading.hardware_stage == PSBC_HW_STAGE_PIXEL);
    assert(reading.ps_clip_distance_reads == 2);
    assert(reading.ps_cull_distance_reads == 0);
    /* The pre-raster export masks stay zero: a read is not an export. */
    assert(!reading.clip_distance_mask && !reading.cull_distance_mask);

    compile_fragment(argv[2], &plain);
    assert(plain.hardware_stage == PSBC_HW_STAGE_PIXEL);
    assert(!plain.ps_clip_distance_reads && !plain.ps_cull_distance_reads);

    /* The register a distance read targets. A fragment that declares eight clip
     * distances and reads only gl_ClipDistance[4] must name the SECOND packed
     * register, because that is where component 4 lives; naming the first made
     * the pixel stage interpolate the wrong register, and the upstream clipping
     * leaf that reads the middle distance failed on hardware because of it. */
    compile_fragment(argv[3], &second);
    assert(second.ps_clip_distance_reads == 8 && !second.ps_cull_distance_reads);
    assert(second.input_semantic_count == 1);
    assert((second.input_semantics[0] & 255u) == PSBC_SEMANTIC_DISTANCE_REGISTER + 1u);

    printf("fragment distance reads reported: clip=%u cull=%u plain=%u\n",
           reading.ps_clip_distance_reads, reading.ps_cull_distance_reads,
           plain.ps_clip_distance_reads);
    return 0;
}
