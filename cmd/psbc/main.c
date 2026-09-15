/*
 * opengnm-psbc — SPIR-V to PS4/PS5 Shader Binary Compiler (CLI)
 *
 * Thin CLI wrapper around libpsbc. The compilation logic lives in
 * libpsbc/psbc_compile.c; this file handles argument parsing, file I/O,
 * and calling psbc_compile_shader().
 */

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "psbc_compile.h"

static const char* VERSION_STR = "0.1.0";

typedef struct {
	const char* inputfile;
	const char* previousfile;
	const char* outputfile;
	const char* metadatafile;
	const char* entrypoint;
	PsbcStage stage;
	bool showhelp;
	PsbcTarget target;
	bool optimise;
	bool verbose;
	bool ngg;
	bool ps5_global_streamout;
	bool force_accelerated_dot;
	bool provoking_vtx_last;
	bool raw;
	uint32_t primitive_type;
	uint32_t address32_hi;
	uint32_t vertex_attribute_count;
	PsbcVertexAttribute vertex_attributes[PSBC_MAX_VERTEX_ATTRIBUTES];
	uint32_t descriptor_binding_count;
	PsbcDescriptorBinding descriptor_bindings[PSBC_MAX_DESCRIPTOR_BINDINGS];
	bool parse_error;
} CmdOptions;

static PsbcVertexFormat parse_vertex_format(const char* name) {
	if (!strcmp(name, "r32_float")) return PSBC_VERTEX_FORMAT_R32_FLOAT;
	if (!strcmp(name, "r32g32_float")) return PSBC_VERTEX_FORMAT_R32G32_FLOAT;
	if (!strcmp(name, "r32g32b32_float")) return PSBC_VERTEX_FORMAT_R32G32B32_FLOAT;
	if (!strcmp(name, "r32g32b32a32_float")) return PSBC_VERTEX_FORMAT_R32G32B32A32_FLOAT;
	if (!strcmp(name, "b8g8r8a8_unorm")) return PSBC_VERTEX_FORMAT_B8G8R8A8_UNORM;
	if (!strcmp(name, "r10g10b10a2_unorm")) return PSBC_VERTEX_FORMAT_R10G10B10A2_UNORM;
	if (!strcmp(name, "b10g10r10a2_unorm")) return PSBC_VERTEX_FORMAT_B10G10R10A2_UNORM;
	if (!strcmp(name, "r10g10b10a2_snorm")) return PSBC_VERTEX_FORMAT_R10G10B10A2_SNORM;
	if (!strcmp(name, "b10g10r10a2_snorm")) return PSBC_VERTEX_FORMAT_B10G10R10A2_SNORM;
	if (!strcmp(name, "r10g10b10a2_uscaled")) return PSBC_VERTEX_FORMAT_R10G10B10A2_USCALED;
	if (!strcmp(name, "b10g10r10a2_uscaled")) return PSBC_VERTEX_FORMAT_B10G10R10A2_USCALED;
	if (!strcmp(name, "r10g10b10a2_sscaled")) return PSBC_VERTEX_FORMAT_R10G10B10A2_SSCALED;
	if (!strcmp(name, "b10g10r10a2_sscaled")) return PSBC_VERTEX_FORMAT_B10G10R10A2_SSCALED;
	if (!strcmp(name, "r32_sint")) return PSBC_VERTEX_FORMAT_R32_SINT;
	if (!strcmp(name, "r32g32_sint")) return PSBC_VERTEX_FORMAT_R32G32_SINT;
	if (!strcmp(name, "r32g32b32_sint")) return PSBC_VERTEX_FORMAT_R32G32B32_SINT;
	if (!strcmp(name, "r32g32b32a32_sint")) return PSBC_VERTEX_FORMAT_R32G32B32A32_SINT;
	if (!strcmp(name, "r32_uint")) return PSBC_VERTEX_FORMAT_R32_UINT;
	if (!strcmp(name, "r32g32_uint")) return PSBC_VERTEX_FORMAT_R32G32_UINT;
	if (!strcmp(name, "r32g32b32_uint")) return PSBC_VERTEX_FORMAT_R32G32B32_UINT;
	if (!strcmp(name, "r32g32b32a32_uint")) return PSBC_VERTEX_FORMAT_R32G32B32A32_UINT;
	if (!strcmp(name, "r8_unorm")) return PSBC_VERTEX_FORMAT_R8_UNORM;
	if (!strcmp(name, "r8_snorm")) return PSBC_VERTEX_FORMAT_R8_SNORM;
	if (!strcmp(name, "r8_uint")) return PSBC_VERTEX_FORMAT_R8_UINT;
	if (!strcmp(name, "r8_sint")) return PSBC_VERTEX_FORMAT_R8_SINT;
	if (!strcmp(name, "r8g8_unorm")) return PSBC_VERTEX_FORMAT_R8G8_UNORM;
	if (!strcmp(name, "r8g8_snorm")) return PSBC_VERTEX_FORMAT_R8G8_SNORM;
	if (!strcmp(name, "r8g8_uint")) return PSBC_VERTEX_FORMAT_R8G8_UINT;
	if (!strcmp(name, "r8g8_sint")) return PSBC_VERTEX_FORMAT_R8G8_SINT;
	if (!strcmp(name, "r8g8b8a8_unorm")) return PSBC_VERTEX_FORMAT_R8G8B8A8_UNORM;
	if (!strcmp(name, "r8g8b8a8_snorm")) return PSBC_VERTEX_FORMAT_R8G8B8A8_SNORM;
	if (!strcmp(name, "r8g8b8a8_uint")) return PSBC_VERTEX_FORMAT_R8G8B8A8_UINT;
	if (!strcmp(name, "r8g8b8a8_sint")) return PSBC_VERTEX_FORMAT_R8G8B8A8_SINT;
	if (!strcmp(name, "r16_unorm")) return PSBC_VERTEX_FORMAT_R16_UNORM;
	if (!strcmp(name, "r16_snorm")) return PSBC_VERTEX_FORMAT_R16_SNORM;
	if (!strcmp(name, "r16_uint")) return PSBC_VERTEX_FORMAT_R16_UINT;
	if (!strcmp(name, "r16_sint")) return PSBC_VERTEX_FORMAT_R16_SINT;
	if (!strcmp(name, "r16_float")) return PSBC_VERTEX_FORMAT_R16_FLOAT;
	if (!strcmp(name, "r16g16_unorm")) return PSBC_VERTEX_FORMAT_R16G16_UNORM;
	if (!strcmp(name, "r16g16_snorm")) return PSBC_VERTEX_FORMAT_R16G16_SNORM;
	if (!strcmp(name, "r16g16_uint")) return PSBC_VERTEX_FORMAT_R16G16_UINT;
	if (!strcmp(name, "r16g16_sint")) return PSBC_VERTEX_FORMAT_R16G16_SINT;
	if (!strcmp(name, "r16g16_float")) return PSBC_VERTEX_FORMAT_R16G16_FLOAT;
	if (!strcmp(name, "r16g16b16a16_unorm")) return PSBC_VERTEX_FORMAT_R16G16B16A16_UNORM;
	if (!strcmp(name, "r16g16b16a16_snorm")) return PSBC_VERTEX_FORMAT_R16G16B16A16_SNORM;
	if (!strcmp(name, "r16g16b16a16_uint")) return PSBC_VERTEX_FORMAT_R16G16B16A16_UINT;
	if (!strcmp(name, "r16g16b16a16_sint")) return PSBC_VERTEX_FORMAT_R16G16B16A16_SINT;
	if (!strcmp(name, "r16g16b16a16_float")) return PSBC_VERTEX_FORMAT_R16G16B16A16_FLOAT;
	return PSBC_VERTEX_FORMAT_NONE;
}

static PsbcDescriptorType parse_descriptor_type(const char* name) {
	if (!strcmp(name, "uniform_buffer"))
		return PSBC_DESCRIPTOR_UNIFORM_BUFFER;
	if (!strcmp(name, "uniform_texel_buffer"))
		return PSBC_DESCRIPTOR_UNIFORM_TEXEL_BUFFER;
	if (!strcmp(name, "combined_image_sampler"))
		return PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER;
	if (!strcmp(name, "storage_buffer"))
		return PSBC_DESCRIPTOR_STORAGE_BUFFER;
	return PSBC_DESCRIPTOR_NONE;
}

static uint32_t parse_primitive_type(const char* name) {
	if (!strcmp(name, "point-list")) return 1;
	if (!strcmp(name, "line-list")) return 2;
	if (!strcmp(name, "line-strip")) return 3;
	if (!strcmp(name, "triangle-list")) return 4;
	if (!strcmp(name, "triangle-fan")) return 5;
	if (!strcmp(name, "triangle-strip")) return 6;
	if (!strcmp(name, "line-list-adjacency")) return 10;
	if (!strcmp(name, "line-strip-adjacency")) return 11;
	if (!strcmp(name, "triangle-list-adjacency")) return 12;
	if (!strcmp(name, "triangle-strip-adjacency")) return 13;
	return UINT32_MAX;
}

static CmdOptions parsecmdoptions(int argc, char* argv[]) {
	CmdOptions res = {
	    .entrypoint = "main",
	    .stage = PSBC_STAGE_NONE,
	    .optimise = true,
	    .target = PSBC_TARGET_PS5,
	};

	for (int i = 0; i < argc; i += 1) {
		const char* curarg = argv[i];
		if (!strcmp(curarg, "-h")) {
			res.showhelp = true;
		} else if (!strcmp(curarg, "-f")) {
			if (i + 1 < argc) {
				res.inputfile = argv[i + 1];
			}
		} else if (!strcmp(curarg, "-o")) {
			if (i + 1 < argc) {
				res.outputfile = argv[i + 1];
			}
		} else if (!strcmp(curarg, "--previous")) {
			if (i + 1 < argc) {
				res.previousfile = argv[i + 1];
			}
		} else if (!strcmp(curarg, "--metadata")) {
			if (i + 1 < argc) {
				res.metadatafile = argv[i + 1];
			}
		} else if (!strcmp(curarg, "-e")) {
			if (i + 1 < argc) {
				res.entrypoint = argv[i + 1];
			}
		} else if (!strcmp(curarg, "-s")) {
			if (i + 1 < argc) {
				res.stage = psbc_stage_from_name(argv[i + 1]);
			}
		} else if (!strcmp(curarg, "-4")) {
			res.target = PSBC_TARGET_PS4_BASE;
		} else if (!strcmp(curarg, "-n")) {
			res.target = PSBC_TARGET_PS4_NEO;
		} else if (!strcmp(curarg, "-g")) {
			res.target = PSBC_TARGET_PS5;
		} else if (!strcmp(curarg, "-Od")) {
			res.optimise = false;
		} else if (!strcmp(curarg, "-vv")) {
			res.verbose = true;
		} else if (!strcmp(curarg, "--ngg")) {
			res.ngg = true;
		} else if (!strcmp(curarg, "--ps5-global-streamout")) {
			res.ps5_global_streamout = true;
		} else if (!strcmp(curarg, "--force-accelerated-dot")) {
			res.force_accelerated_dot = true;
		} else if (!strcmp(curarg, "--provoking-vertex-last")) {
			res.provoking_vtx_last = true;
		} else if (!strcmp(curarg, "--raw")) {
			res.raw = true;
		} else if (!strcmp(curarg, "--primitive-type")) {
			if (i + 1 >= argc) {
				res.parse_error = true;
				continue;
			}
			res.primitive_type = parse_primitive_type(argv[++i]);
			if (res.primitive_type == UINT32_MAX)
				res.parse_error = true;
		} else if (!strcmp(curarg, "--address32-hi")) {
			if (i + 1 < argc) {
				char* end = NULL;
				unsigned long value = strtoul(argv[++i], &end, 0);
				if (!end || *end || value > UINT32_MAX)
					res.parse_error = true;
				else
					res.address32_hi = (uint32_t)value;
			}
		} else if (!strcmp(curarg, "--vertex-attribute")) {
			if (i + 1 >= argc ||
			    res.vertex_attribute_count >= PSBC_MAX_VERTEX_ATTRIBUTES) {
				res.parse_error = true;
				continue;
			}
			unsigned location, binding, offset, stride, alignment;
			unsigned instance_divisor = 0;
			char format_name[32];
			const int fields = sscanf(argv[++i],
			                          "%u:%31[^:]:%u:%u:%u:%u:%u",
			                          &location, format_name, &binding, &offset,
			                          &stride, &alignment, &instance_divisor);
			if (fields != 6 && fields != 7) {
				res.parse_error = true;
				continue;
			}
			PsbcVertexAttribute* attribute =
			    &res.vertex_attributes[res.vertex_attribute_count++];
			attribute->location = (uint8_t)location;
			attribute->binding = (uint8_t)binding;
			attribute->format = parse_vertex_format(format_name);
			attribute->offset = offset;
			attribute->stride = stride;
			attribute->alignment = alignment;
			attribute->instance_divisor = instance_divisor;
			if (location >= PSBC_MAX_VERTEX_ATTRIBUTES || binding >= 32 ||
			    attribute->format == PSBC_VERTEX_FORMAT_NONE)
				res.parse_error = true;
		} else if (!strcmp(curarg, "--descriptor-binding")) {
			if (i + 1 >= argc ||
			    res.descriptor_binding_count >= PSBC_MAX_DESCRIPTOR_BINDINGS) {
				res.parse_error = true;
				continue;
			}
			unsigned set, binding, array_size, offset, stride;
			char type_name[32];
			if (sscanf(argv[++i], "%u:%u:%31[^:]:%u:%u:%u", &set,
			           &binding, type_name, &array_size, &offset, &stride) != 6) {
				res.parse_error = true;
				continue;
			}
			PsbcDescriptorBinding* descriptor =
			    &res.descriptor_bindings[res.descriptor_binding_count++];
			descriptor->set = (uint8_t)set;
			descriptor->binding = (uint8_t)binding;
			descriptor->type = parse_descriptor_type(type_name);
			descriptor->array_size = array_size;
			descriptor->offset = offset;
			descriptor->stride = stride;
			if (set != descriptor->set || binding != descriptor->binding ||
			    descriptor->type == PSBC_DESCRIPTOR_NONE)
				res.parse_error = true;
		}
	}

	return res;
}

static inline void showhelp(void) {
	printf(
	    "opengnm-psbc %s\n"
	    "Usage:\n"
	    "\t-f [path] -- The input SPIRV file to compile\n"
	    "\t-o [path] -- The output compiled GCN shader file\n"
	    "\t--metadata [path] -- Write typed hardware metadata as JSON\n"
	    "\t--previous [path] -- Vertex SPIR-V for a merged NGG geometry pipeline\n"
	    "\t-e [entrypoint] -- The entrypoint's name (default: \"main\")\n"
	    "\t-s [stage] -- The shader's stage name\n"
	    "\t-Od -- Disable some optimisations\n"
	    "\t-4 -- Target PS4 base (GFX7) instead of PS5\n"
	    "\t-n -- Target PS4 Pro (NEO/GFX8) instead of PS5\n"
	    "\t-g -- Target PS5 (RDNA2/GFX10.3) [default]\n"
	    "\t--ngg -- Experimental PS5 vertex NGG lowering (VS -> FS)\n"
	    "\t--ps5-global-streamout -- Ordered no-GDS merged GS streamout\n"
	    "\t--force-accelerated-dot -- Diagnostic only: emit native packed-dot ISA\n"
	    "\t--provoking-vertex-last -- Select the final flat-shaded primitive vertex\n"
	    "\t--raw -- Write raw ACO machine code instead of a GNM wrapper\n"
	    "\t--primitive-type [point-list|line-list|line-strip|triangle-list|triangle-fan|triangle-strip|line-list-adjacency|line-strip-adjacency|triangle-list-adjacency|triangle-strip-adjacency]\n"
	    "\t--address32-hi [value] -- Upper 32 bits for 32-bit GPU pointers\n"
	    "\t--vertex-attribute loc:format:binding:offset:stride:alignment[:divisor]\n"
	    "\t--descriptor-binding set:binding:type:array:offset:stride\n"
	    "\t-vv -- Enable verbose messages output\n"
	    "\t-h -- Show this help message\n",
	    VERSION_STR
	);
}

static inline _Noreturn void fatal(const char* msg) {
	fprintf(stderr, "%s\n", msg);
	exit(EXIT_FAILURE);
}
static inline _Noreturn void fatalf(const char* fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	putc('\n', stderr);
	exit(EXIT_FAILURE);
}

static void write_registers(FILE* h, const PsbcRegisterWrite* registers,
	                        uint32_t count) {
	for (uint32_t i = 0; i < count; ++i) {
		fprintf(h, "%s    {\"offset\": %u, \"value\": %u}",
		        i ? ",\n" : "", registers[i].offset, registers[i].value);
	}
}

static void write_semantics(FILE* h, const uint32_t* semantics,
	                        uint32_t count) {
	for (uint32_t i = 0; i < count; ++i)
		fprintf(h, "%s    %u", i ? ",\n" : "", semantics[i]);
}

static void write_metadata(const char* path, const PsbcShaderOutput* output) {
	FILE* h = fopen(path, "wb");
	if (!h)
		fatalf("Failed to open metadata file with %s", strerror(errno));

	const PsbcShaderMetadata* metadata = &output->metadata;
	fprintf(h,
	        "{\n"
	        "  \"version\": %u,\n"
	        "  \"target\": %u,\n"
	        "  \"source_stage\": %u,\n"
	        "  \"hardware_stage\": %u,\n"
	        "  \"machine_code_size\": %zu,\n"
	        "  \"unresolved_fields\": %u,\n",
	        metadata->version, metadata->target, metadata->source_stage,
	        metadata->hardware_stage, output->machine_code_size,
	        metadata->unresolved_fields);
	if (metadata->linkage_valid) {
		fprintf(h,
		        "  \"linkage\": {\n"
		        "    \"ge_cntl\": {\"offset\": %u, \"value\": %u},\n"
		        "    \"stages_en\": {\"offset\": %u, \"value\": %u},\n"
		        "    \"user_vgpr_en\": {\"offset\": %u, \"value\": %u}\n"
		        "  },\n",
		        metadata->linkage_ge_cntl.offset,
		        metadata->linkage_ge_cntl.value,
		        metadata->linkage_stages_en.offset,
		        metadata->linkage_stages_en.value,
		        metadata->linkage_user_vgpr_en.offset,
		        metadata->linkage_user_vgpr_en.value);
	} else {
		fprintf(h, "  \"linkage\": null,\n");
	}
	fprintf(h, "  \"context_registers\": [\n");
	write_registers(h, metadata->context_registers,
	                metadata->context_register_count);
	fprintf(h, "\n  ],\n  \"shader_registers\": [\n");
	write_registers(h, metadata->shader_registers,
	                metadata->shader_register_count);
	fprintf(h, "\n  ],\n  \"input_semantics\": [\n");
	write_semantics(h, metadata->input_semantics,
	                metadata->input_semantic_count);
	fprintf(h, "\n  ],\n  \"output_semantics\": [\n");
	write_semantics(h, metadata->output_semantics,
	                metadata->output_semantic_count);
	fprintf(h,
	        "\n  ],\n"
	        "  \"address32_hi\": %u,\n"
	        "  \"user_sgpr_count\": %u,\n"
	        "  \"vertex_buffer_table_user_data_dword\": ",
	        metadata->address32_hi, metadata->user_sgpr_count);
	if (metadata->vertex_buffer_table_valid)
		fprintf(h, "%u", metadata->vertex_buffer_table_user_data_dword);
	else
		fprintf(h, "null");
	fprintf(h, ",\n  \"vertex_buffer_usage_mask\": %u,\n"
	           "  \"vertex_buffer_per_attribute\": %s",
	        metadata->vertex_buffer_usage_mask,
	        metadata->vertex_buffer_per_attribute ? "true" : "false");
	fprintf(h, ",\n  \"descriptor_set0_user_data_dword\": ");
	if (metadata->descriptor_set0_valid)
		fprintf(h, "%u", metadata->descriptor_set0_user_data_dword);
	else
		fprintf(h, "null");
	fprintf(h, ",\n  \"descriptor_set_user_data_dwords\": [");
	for (uint32_t set = 0; set < PSBC_MAX_DESCRIPTOR_SETS; ++set) {
		if (set)
			fprintf(h, ", ");
		if (metadata->descriptor_set_valid[set])
			fprintf(h, "%u", metadata->descriptor_set_user_data_dword[set]);
		else
			fprintf(h, "null");
	}
	fprintf(h, "]");
	fprintf(h, ",\n  \"base_vertex_user_data_dword\": ");
	if (metadata->base_vertex_valid)
		fprintf(h, "%u", metadata->base_vertex_user_data_dword);
	else
		fprintf(h, "null");
	fprintf(h, ",\n  \"draw_id_user_data_dword\": ");
	if (metadata->draw_id_valid)
		fprintf(h, "%u", metadata->draw_id_user_data_dword);
	else
		fprintf(h, "null");
	fprintf(h, ",\n  \"start_instance_user_data_dword\": ");
	if (metadata->start_instance_valid)
		fprintf(h, "%u", metadata->start_instance_user_data_dword);
	else
		fprintf(h, "null");
	fprintf(h, ",\n  \"ngg_lds_layout\": ");
	if (metadata->ngg_lds_layout_valid)
		fprintf(h, "{\"user_data_dword\": %u, \"value\": %u}",
		        metadata->ngg_lds_layout_user_data_dword, metadata->ngg_lds_layout);
	else
		fprintf(h, "null");
	fprintf(h, ",\n  \"streamout\": ");
	if (metadata->streamout_valid) {
		fprintf(h,
		        "{\n"
		        "    \"buffer_table_user_data_dword\": %u,\n"
		        "    \"enabled_stream_buffers_mask\": %u,\n"
		        "    \"strides_dwords\": [%u, %u, %u, %u],\n"
		        "    \"config_sgpr\": %u,\n"
		        "    \"write_index_sgpr\": %u,\n"
		        "    \"offset_sgprs\": [",
		        metadata->streamout_buffer_table_user_data_dword,
		        metadata->streamout_enabled_stream_buffers_mask,
		        metadata->streamout_strides_dwords[0],
		        metadata->streamout_strides_dwords[1],
		        metadata->streamout_strides_dwords[2],
		        metadata->streamout_strides_dwords[3],
		        metadata->streamout_config_sgpr,
		        metadata->streamout_write_index_sgpr);
		for (uint32_t i = 0; i < 4; ++i) {
			if (i)
				fprintf(h, ", ");
			if (metadata->streamout_strides_dwords[i])
				fprintf(h, "%u", metadata->streamout_offset_sgprs[i]);
			else
				fprintf(h, "null");
		}
		fprintf(h, "]\n  }");
	} else {
		fprintf(h, "null");
	}
	fprintf(h, ",\n  \"descriptor_bindings\": [\n");
	for (uint32_t i = 0; i < metadata->descriptor_binding_count; ++i) {
		const PsbcDescriptorBinding* binding =
		    &metadata->descriptor_bindings[i];
		fprintf(h,
		        "%s    {\"set\": %u, \"binding\": %u, \"type\": %u, "
		        "\"array_size\": %u, \"offset\": %u, \"stride\": %u}",
		        i ? ",\n" : "", binding->set, binding->binding,
		        binding->type, binding->array_size, binding->offset,
		        binding->stride);
	}
	fprintf(h, "\n  ]\n}\n");
	if (fclose(h) != 0)
		fatalf("Failed to write metadata file with %s", strerror(errno));
}

int main(int argc, char** argv) {
	const CmdOptions opts = parsecmdoptions(argc, argv);

	if (opts.showhelp) {
		showhelp();
		return EXIT_SUCCESS;
	}

	if (!opts.inputfile) {
		fatal("Please pass an input file");
	}
	if (opts.parse_error) {
		fatal("Invalid compiler option");
	}
	if (!opts.outputfile) {
		fatal("Please pass an output file");
	}
	if (opts.stage == PSBC_STAGE_NONE) {
		fatal("Please pass a valid shader stage");
	}

	FILE* inputhandle = fopen(opts.inputfile, "rb");
	if (!inputhandle) {
		fatalf("Failed to open input file with %s", strerror(errno));
	}

	fseek(inputhandle, 0, SEEK_END);
	size_t inputlen = ftell(inputhandle);
	fseek(inputhandle, 0, SEEK_SET);

	void* input = malloc(inputlen);
	if (!input) {
		abort();
	}

	if (fread(input, 1, inputlen, inputhandle) != inputlen) {
		fatalf("Failed to read input file with %s", strerror(errno));
	}

	fclose(inputhandle);

	void* previous = NULL;
	size_t previouslen = 0;
	if (opts.previousfile) {
		FILE* previoushandle = fopen(opts.previousfile, "rb");
		if (!previoushandle)
			fatalf("Failed to open previous-stage input with %s", strerror(errno));
		fseek(previoushandle, 0, SEEK_END);
		previouslen = ftell(previoushandle);
		fseek(previoushandle, 0, SEEK_SET);
		previous = malloc(previouslen);
		if (!previous)
			abort();
		if (fread(previous, 1, previouslen, previoushandle) != previouslen)
			fatalf("Failed to read previous-stage input with %s", strerror(errno));
		fclose(previoushandle);
	}

	/* Compile via libpsbc */
	PsbcCompileOptions compile_opts = {
	    .target = opts.target,
	    .stage = opts.stage,
	    .entrypoint = opts.entrypoint,
	    .optimise = opts.optimise,
	    .ngg = opts.ngg,
	    .ps5_global_streamout = opts.ps5_global_streamout,
	    .force_accelerated_dot = opts.force_accelerated_dot,
	    .primitive_type = opts.primitive_type,
	    .provoking_vtx_last = opts.provoking_vtx_last,
	    .address32_hi = opts.address32_hi,
	    .vertex_attribute_count = opts.vertex_attribute_count,
	    .descriptor_binding_count = opts.descriptor_binding_count,
	};
	memcpy(compile_opts.vertex_attributes, opts.vertex_attributes,
	       sizeof(compile_opts.vertex_attributes));
	memcpy(compile_opts.descriptor_bindings, opts.descriptor_bindings,
	       sizeof(compile_opts.descriptor_bindings));

	PsbcShaderOutput output = {0};
	PsbcResult result = opts.previousfile
	    ? psbc_compile_geometry_pipeline(
	          (const uint32_t*)previous, previouslen,
	          (const uint32_t*)input, inputlen, &compile_opts, &output)
	    : psbc_compile_shader(
	          (const uint32_t*)input, inputlen, &compile_opts, &output);

	free(input);
	free(previous);

	if (result != PSBC_RESULT_OK) {
		fatalf("Shader compilation failed: %s", psbc_result_string(result));
	}

	/* NGG has no valid legacy GNM wrapper; always emit raw code for it. */
	const bool write_raw = opts.raw || opts.ngg;
	const void* write_data = write_raw ? output.machine_code : output.data;
	const size_t write_size = write_raw ? output.machine_code_size : output.size;

	FILE* h = fopen(opts.outputfile, "wb");
	if (!h) {
		fatalf("Failed to open output file with %s", strerror(errno));
	}

	if (fwrite(write_data, 1, write_size, h) != write_size) {
		psbc_free_output(&output);
		fclose(h);
		fatalf("Failed to write output file with %s", strerror(errno));
	}

	fclose(h);
	if (opts.metadatafile)
		write_metadata(opts.metadatafile, &output);

	if (opts.verbose) {
		printf("Compiled %s (%s) -> %s (%zu bytes%s)\n",
		       opts.inputfile, opts.entrypoint, opts.outputfile, write_size,
		       write_raw ? ", raw" : "");
	}

	psbc_free_output(&output);

	return EXIT_SUCCESS;
}
