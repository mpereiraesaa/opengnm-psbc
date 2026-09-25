#define _POSIX_C_SOURCE 200809L
#include "psbc_compile.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Ordered no-GDS streamout contract (ps5_global_streamout): the merged NGG
 * vertex+geometry capture program reserves each workgroup's range with global
 * atomics on the control block (record 4 of the streamout table) in
 * primitive order. A workgroup waits, bounded, until the ticket at +48 equals
 * its first primitive id, reserves, returns the unwritten part of its
 * reservation so the offset is the end of the last whole primitive, and hands
 * the ticket on with an atomic swap. Nothing uses GDS. The final ACO listing
 * printed under PSBC_DEBUG_DISASM is the evidence; the metadata must still
 * describe the capture. */

static uint32_t *read_spirv(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb"); assert(file);
    assert(!fseek(file, 0, SEEK_END)); long size = ftell(file);
    assert(size > 0 && !(size & 3)); rewind(file);
    uint32_t *data = malloc((size_t)size); assert(data);
    assert(fread(data, 1, (size_t)size, file) == (size_t)size);
    fclose(file); *bytes = (size_t)size; return data;
}

static unsigned count(const char *text, const char *needle)
{
    unsigned n = 0;
    for (const char *p = strstr(text, needle); p; p = strstr(p + 1, needle)) ++n;
    return n;
}

int main(int argc, char **argv)
{
    assert(argc == 3);
    size_t vertex_bytes, geometry_bytes;
    uint32_t *vertex = read_spirv(argv[1], &vertex_bytes);
    uint32_t *geometry = read_spirv(argv[2], &geometry_bytes);
    PsbcCompileOptions opts = {.target = PSBC_TARGET_PS5, .stage = PSBC_STAGE_GEOMETRY,
        .entrypoint = "main", .optimise = true, .ngg = true, .address32_hi = 2,
        .ps5_global_streamout = true, .primitive_type = 1 /* point list */};
    FILE *capture = tmpfile(); assert(capture);
    fflush(stderr);
    int saved = dup(STDERR_FILENO); assert(saved >= 0);
    assert(dup2(fileno(capture), STDERR_FILENO) >= 0);
    PsbcShaderOutput output = {0};
    PsbcResult result = psbc_compile_geometry_pipeline(vertex, vertex_bytes, geometry,
        geometry_bytes, &opts, &output);
    fflush(stderr);
    assert(dup2(saved, STDERR_FILENO) >= 0); close(saved);
    assert(result == PSBC_RESULT_OK && output.machine_code && output.machine_code_size);
    const PsbcShaderMetadata *m = &output.metadata;
    assert(m->streamout_valid && m->streamout_enabled_stream_buffers_mask == 1u);
    assert(m->streamout_strides_dwords[0] == 8u && !m->streamout_strides_dwords[1]);
    assert(m->streamout_buffer_table_user_data_dword < m->user_sgpr_count);
    long size = ftell(capture); assert(size > 0); rewind(capture);
    char *text = calloc((size_t)size + 1, 1); assert(text);
    assert(fread(text, 1, (size_t)size, capture) == (size_t)size); fclose(capture);
    /* The last listing, after lowering to hardware instructions, is the
     * final program. */
    const char *marker = "After lowering to hw instructions:";
    char *final = NULL;
    for (char *p = strstr(text, marker); p; p = strstr(p + 1, marker)) final = p;
    assert(final);
    /* No GDS ordered append or GDS counter. */
    assert(!count(final, "ds_ordered_count") && !count(final, " gds"));
    /* The ticket: one bounded wait (a returning atomic add of zero at +48,
     * an iteration bound of 0x10000) and one hand-off swap. */
    assert(count(final, "global_atomic_swap") == 1u);
    assert(count(final, "0x10000") >= 1u);
    assert(count(final, "v_mov_b32 48") >= 2u);
    printf("ordered streamout: primitive-id ticket, bounded wait, one hand-off, "
           "no GDS (%u bytes)\n", (unsigned)output.machine_code_size);
    psbc_free_output(&output);
    free(text); free(vertex); free(geometry);
    return 0;
}
