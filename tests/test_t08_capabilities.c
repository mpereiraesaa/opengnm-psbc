#include "psbc_compile.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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

static int has_capability(const uint32_t *words, size_t bytes, uint32_t capability)
{
    size_t count = bytes / sizeof(*words);
    for (size_t at = 5; at < count;) {
        uint16_t length = words[at] >> 16;
        assert(length && length <= count - at);
        if ((words[at] & 0xffffu) == 17u && length == 2 && words[at + 1] == capability)
            return 1;
        at += length;
    }
    return 0;
}

static PsbcCompileOptions options(void)
{
    PsbcCompileOptions out = {.target = PSBC_TARGET_PS5,
        .stage = PSBC_STAGE_COMPUTE, .entrypoint = "main", .optimise = true,
        .address32_hi = 2, .force_indirect_push_constants = true,
        .descriptor_binding_count = 1};
    out.descriptor_bindings[0] = (PsbcDescriptorBinding){
        .set = 0, .binding = 0, .type = PSBC_DESCRIPTOR_STORAGE_BUFFER,
        .array_size = 1, .stride = 16};
    return out;
}

static void expect_result(const uint32_t *spirv, size_t bytes,
                          PsbcCompileOptions *opts, PsbcResult expected)
{
    PsbcShaderOutput output = {0};
    PsbcResult result = psbc_compile_shader(spirv, bytes, opts, &output);
    if (result != expected)
        fprintf(stderr, "expected %s, got %s\n", psbc_result_string(expected),
                psbc_result_string(result));
    assert(result == expected);
    if (result == PSBC_RESULT_OK) {
        assert(output.machine_code && output.machine_code_size);
        psbc_free_output(&output);
    }
}

int main(void)
{
    size_t bytes = 0;
    uint32_t *address = read_spirv("tests/t08_address.comp.spv", &bytes);
    assert(has_capability(address, bytes, 5347)); /* PhysicalStorageBufferAddresses */
    PsbcCompileOptions opts = options();
    expect_result(address, bytes, &opts, PSBC_RESULT_UNSUPPORTED_CAPABILITY);
    opts.enable_physical_storage_buffer_addresses = true;
    expect_result(address, bytes, &opts, PSBC_RESULT_OK);
    free(address);

    uint32_t *queue = read_spirv("tests/t08_memory_model_queue.comp.spv", &bytes);
    assert(has_capability(queue, bytes, 5345)); /* VulkanMemoryModel */
    assert(!has_capability(queue, bytes, 5346));
    opts = options();
    expect_result(queue, bytes, &opts, PSBC_RESULT_UNSUPPORTED_CAPABILITY);
    opts.enable_vulkan_memory_model = true;
    expect_result(queue, bytes, &opts, PSBC_RESULT_OK);
    free(queue);

    uint32_t *device = read_spirv("tests/t08_memory_model.comp.spv", &bytes);
    assert(has_capability(device, bytes, 5345) && has_capability(device, bytes, 5346));
    opts = options();
    opts.enable_vulkan_memory_model_device_scope = true;
    expect_result(device, bytes, &opts, PSBC_RESULT_INTERNAL_ERROR);
    opts.enable_vulkan_memory_model = true;
    opts.enable_vulkan_memory_model_device_scope = false;
    expect_result(device, bytes, &opts, PSBC_RESULT_UNSUPPORTED_CAPABILITY);
    opts.enable_vulkan_memory_model_device_scope = true;
    expect_result(device, bytes, &opts, PSBC_RESULT_OK);
    free(device);

    puts("T08 PSBC capability gates: pass (compiled ISA only)");
    return 0;
}
