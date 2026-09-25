#define _POSIX_C_SOURCE 200809L
#include "psbc_compile.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* robustBufferAccess2 contract: with the option set, the SSBO pair
 * src.v[i - 1], src.v[i] and the UBO pair ubo.v[i - 1], ubo.v[i] stay separate
 * loads (their combined offset could wrap); without it the vectorizer merges
 * each pair into one wider load. The final ACO listing printed under
 * PSBC_DEBUG_DISASM is the evidence. */

static uint32_t *read_spirv(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb"); assert(file);
    assert(!fseek(file, 0, SEEK_END)); long size = ftell(file);
    assert(size > 0 && !(size & 3)); rewind(file);
    uint32_t *data = malloc((size_t)size); assert(data);
    assert(fread(data, 1, (size_t)size, file) == (size_t)size);
    fclose(file); *bytes = (size_t)size; return data;
}

struct counts { unsigned dword, dwordx2, dwordx4, dwordx8; };

static unsigned count(const char *text, const char *needle)
{
    unsigned n = 0;
    for (const char *p = strstr(text, needle); p; p = strstr(p + 1, needle))
        if (p[strlen(needle)] == ' ') ++n;
    return n;
}

/* Only the first ACO listing (after register allocation) is counted. */
static struct counts compile(const uint32_t *spirv, size_t bytes, bool robust)
{
    PsbcCompileOptions opts = {.target = PSBC_TARGET_PS5, .stage = PSBC_STAGE_COMPUTE,
        .entrypoint = "main", .optimise = true, .address32_hi = 2,
        .descriptor_binding_count = 3, .robust_buffer_access2 = robust};
    opts.descriptor_bindings[0] = (PsbcDescriptorBinding){.set = 0, .binding = 0,
        .type = PSBC_DESCRIPTOR_STORAGE_BUFFER, .array_size = 1, .stride = 16};
    opts.descriptor_bindings[1] = (PsbcDescriptorBinding){.set = 0, .binding = 1,
        .type = PSBC_DESCRIPTOR_STORAGE_BUFFER, .array_size = 1, .offset = 16, .stride = 16};
    opts.descriptor_bindings[2] = (PsbcDescriptorBinding){.set = 0, .binding = 2,
        .type = PSBC_DESCRIPTOR_UNIFORM_BUFFER, .array_size = 1, .offset = 32, .stride = 16};
    FILE *capture = tmpfile(); assert(capture);
    fflush(stderr);
    int saved = dup(STDERR_FILENO); assert(saved >= 0);
    assert(dup2(fileno(capture), STDERR_FILENO) >= 0);
    PsbcShaderOutput output = {0};
    PsbcResult result = psbc_compile_shader(spirv, bytes, &opts, &output);
    fflush(stderr);
    assert(dup2(saved, STDERR_FILENO) >= 0); close(saved);
    assert(result == PSBC_RESULT_OK && output.machine_code && output.machine_code_size);
    psbc_free_output(&output);
    long size = ftell(capture); assert(size > 0); rewind(capture);
    char *text = calloc((size_t)size + 1, 1); assert(text);
    assert(fread(text, 1, (size_t)size, capture) == (size_t)size); fclose(capture);
    char *first = strstr(text, "After RA:"); assert(first);
    char *stage = strstr(first, "ACO shader stage:"); assert(stage);
    char *next = strstr(stage + 1, "ACO shader stage:");
    if (next) *next = '\0';
    struct counts c = {count(first, "s_buffer_load_dword"), count(first, "s_buffer_load_dwordx2"),
                       count(first, "s_buffer_load_dwordx4"), count(first, "s_buffer_load_dwordx8")};
    free(text);
    return c;
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    size_t bytes = 0; uint32_t *spirv = read_spirv(argv[1], &bytes);
    assert(!setenv("PSBC_DEBUG_DISASM", "1", 1));
    const struct counts relaxed = compile(spirv, bytes, false);
    const struct counts robust = compile(spirv, bytes, true);
    printf("relaxed: dword=%u x2=%u x4=%u x8=%u; robust: dword=%u x2=%u x4=%u x8=%u\n",
           relaxed.dword, relaxed.dwordx2, relaxed.dwordx4, relaxed.dwordx8,
           robust.dword, robust.dwordx2, robust.dwordx4, robust.dwordx8);
    /* Positive control: without the option both pairs are merged. */
    assert(relaxed.dwordx2 == 1 && relaxed.dwordx8 == 1);
    /* With it: no merged pair; the SSBO pair is two dword loads and the UBO
     * pair two dwordx4 loads, besides the dword load of the index. */
    assert(robust.dwordx2 == 0 && robust.dwordx8 == 0);
    assert(robust.dword == relaxed.dword + 2 && robust.dwordx4 == relaxed.dwordx4 + 2);
    free(spirv);
    puts("robustBufferAccess2 keeps wrapping-offset loads separate: pass");
}
