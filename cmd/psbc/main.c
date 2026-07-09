/*
 * opengnm-psbc — SPIR-V to PS4/PS5 Shader Binary Compiler
 *
 * Ported from the old psbc (Mesa ~22.x) to Mesa 26.2.0 APIs.
 * Uses radv_shader_nir_to_asm() for the full ACO compile pipeline.
 * Supports GFX7 (PS4 base), GFX8 (PS4 NEO), and GFX10.3 (PS5 RDNA2).
 */

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <pssl_types.h>

#include "aco_interface.h"
#include "ac_gpu_info.h"
#include "ac_shader_util.h"
#include "ac_binary.h"
#include "amd_family.h"
#include "nir/radv_nir.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "radv_shader_info.h"
#include "radv_aco_shader_info.h"
#include "radv_pipeline.h"
#include "sid.h"

#include "crc32_sb.h"

static const char* VERSION_STR = "0.1.0";

typedef enum {
	GFX_TARGET_PS4_BASE = 0,
	GFX_TARGET_PS4_NEO  = 1,
	GFX_TARGET_PS5      = 2,
} GfxTarget;

typedef struct {
	const char* inputfile;
	const char* outputfile;
	const char* entrypoint;
	mesa_shader_stage stage;
	bool showhelp;
	GfxTarget target;
	bool optimise;
	bool verbose;
} CmdOptions;

static inline mesa_shader_stage findstage(const char* name) {
	if (!strcmp(name, "vertex")) {
		return MESA_SHADER_VERTEX;
	} else if (!strcmp(name, "tess-ctrl")) {
		return MESA_SHADER_TESS_CTRL;
	} else if (!strcmp(name, "tess-eval")) {
		return MESA_SHADER_TESS_EVAL;
	} else if (!strcmp(name, "geometry")) {
		return MESA_SHADER_GEOMETRY;
	} else if (!strcmp(name, "fragment")) {
		return MESA_SHADER_FRAGMENT;
	} else if (!strcmp(name, "compute")) {
		return MESA_SHADER_COMPUTE;
	} else if (!strcmp(name, "task")) {
		return MESA_SHADER_TASK;
	}
	return MESA_SHADER_NONE;
}

static CmdOptions parsecmdoptions(int argc, char* argv[]) {
	CmdOptions res = {
	    .entrypoint = "main",
	    .stage = MESA_SHADER_NONE,
	    .optimise = true,
	    .target = GFX_TARGET_PS5,
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
		} else if (!strcmp(curarg, "-e")) {
			if (i + 1 < argc) {
				res.entrypoint = argv[i + 1];
			}
		} else if (!strcmp(curarg, "-s")) {
			if (i + 1 < argc) {
				res.stage = findstage(argv[i + 1]);
			}
		} else if (!strcmp(curarg, "-4")) {
		res.target = GFX_TARGET_PS4_BASE;
	} else if (!strcmp(curarg, "-n")) {
			res.target = GFX_TARGET_PS4_NEO;
		} else if (!strcmp(curarg, "-g")) {
			res.target = GFX_TARGET_PS5;
		} else if (!strcmp(curarg, "-Od")) {
			res.optimise = false;
		} else if (!strcmp(curarg, "-vv")) {
			res.verbose = true;
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
	    "\t-e [entrypoint] -- The entrypoint's name (default: \"main\")\n"
	    "\t-s [stage] -- The shader's stage name\n"
	    "\t-Od -- Disable some optimisations\n"
	    "\t-4 -- Target PS4 base (GFX7) instead of PS5\n"
	    "\t-n -- Target PS4 Pro (NEO/GFX8) instead of PS5\n"
	    "\t-g -- Target PS5 (RDNA2/GFX10.3) [default]\n"
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

static inline PsslShaderType psbshtype(mesa_shader_stage s) {
	switch (s) {
	case MESA_SHADER_VERTEX:
		return PSSL_SHADER_VS;
	case MESA_SHADER_FRAGMENT:
		return PSSL_SHADER_FS;
	case MESA_SHADER_COMPUTE:
		return PSSL_SHADER_CS;
	default:
		fatalf("Unhandled shader type %u", s);
	}
}
static inline GnmShaderType gnmshtype(mesa_shader_stage s) {
	switch (s) {
	case MESA_SHADER_VERTEX:
		return GNM_SHADER_VERTEX;
	case MESA_SHADER_FRAGMENT:
		return GNM_SHADER_PIXEL;
	case MESA_SHADER_COMPUTE:
		return GNM_SHADER_COMPUTE;
	default:
		fatalf("Unhandled shader type %u", s);
	}
}
static inline GnmShaderBinaryType shbintype(mesa_shader_stage s) {
	switch (s) {
	case MESA_SHADER_VERTEX:
		return GNM_SHB_VS_VS;
	case MESA_SHADER_FRAGMENT:
		return GNM_SHB_PS;
	case MESA_SHADER_COMPUTE:
		return GNM_SHB_CS;
	default:
		fatalf("Unhandled shader type %u", s);
	}
}
static inline uint32_t headershsize(mesa_shader_stage s) {
	switch (s) {
	case MESA_SHADER_VERTEX:
		return sizeof(GnmVsShader);
	case MESA_SHADER_FRAGMENT:
		return sizeof(GnmPsShader);
	default:
		fatalf("Unhandled shader type %u", s);
	}
}

static inline uint32_t hashsb(
    const void* code, uint32_t codesize, const GnmShaderBinaryInfo* sb
) {
	static const uint8_t padbytes[8] = {0};
	const uint32_t numpadbytes =
	    (sb->length & 0x7) ? 8 - (sb->length & 0x7) : 0;
	assert(numpadbytes <= sizeof(padbytes));

	uint32_t hash = crc32_sb(code, codesize, crc32_sb_begin());
	hash = crc32_sb(padbytes, numpadbytes, hash);
	hash = crc32_sb(sb, sizeof(*sb) - sizeof(sb->crc32), hash);
	return crc32_sb_end(hash);
}

/*
 * Compute the number of position exports from shader info.
 * Ported from radv_get_num_pos_exports() which is static in radv_shader.c.
 */
static unsigned get_num_pos_exports(const struct radv_shader_info *info,
                                    unsigned *clip_dist_mask_out,
                                    unsigned *cull_dist_mask_out) {
	unsigned num = 1;

	if (info->outinfo.writes_pointsize || info->outinfo.writes_viewport_index ||
	    info->outinfo.writes_layer || info->outinfo.writes_primitive_shading_rate)
		num++;

	unsigned num_clip_dist_comps = util_bitcount(info->outinfo.clip_dist_mask);
	unsigned num_cull_dist_comps = info->has_ngg_culling ? 0 : util_bitcount(info->outinfo.cull_dist_mask);
	unsigned clip_cull_mask = BITFIELD_MASK(num_clip_dist_comps + num_cull_dist_comps);

	if (clip_cull_mask & 0x0f)
		num++;
	if (clip_cull_mask & 0xf0)
		num++;

	if (clip_dist_mask_out)
		*clip_dist_mask_out = BITFIELD_MASK(num_clip_dist_comps);
	if (cull_dist_mask_out)
		*cull_dist_mask_out = BITFIELD_RANGE(num_clip_dist_comps, num_cull_dist_comps);
	return num;
}

/*
 * Compute db_shader_control register value for PS.
 * Ported from the inline computation in radv_precompute_registers_hw_ps().
 */
static uint32_t compute_db_shader_control(const struct radv_shader_info *info) {
	unsigned conservative_z_export = V_02880C_EXPORT_ANY_Z;
	if (info->ps.depth_layout == FRAG_DEPTH_LAYOUT_GREATER)
		conservative_z_export = V_02880C_EXPORT_GREATER_THAN_Z;
	else if (info->ps.depth_layout == FRAG_DEPTH_LAYOUT_LESS)
		conservative_z_export = V_02880C_EXPORT_LESS_THAN_Z;

	const unsigned z_order =
		info->ps.early_fragment_test || !info->ps.writes_memory
			? V_02880C_EARLY_Z_THEN_LATE_Z : V_02880C_LATE_Z;

	return S_02880C_Z_EXPORT_ENABLE(info->ps.writes_z) |
	       S_02880C_STENCIL_TEST_VAL_EXPORT_ENABLE(info->ps.writes_stencil) |
	       S_02880C_KILL_ENABLE(info->ps.can_discard) |
	       S_02880C_MASK_EXPORT_ENABLE(info->ps.writes_sample_mask) |
	       S_02880C_CONSERVATIVE_Z_EXPORT(conservative_z_export) |
	       S_02880C_Z_ORDER(z_order) |
	       S_02880C_DEPTH_BEFORE_SHADER(info->ps.early_fragment_test) |
	       S_02880C_PRE_SHADER_DEPTH_COVERAGE_ENABLE(info->ps.post_depth_coverage) |
	       S_02880C_EXEC_ON_HIER_FAIL(info->ps.writes_memory) |
	       S_02880C_EXEC_ON_NOOP(info->ps.writes_memory) |
	       S_02880C_PRIMITIVE_ORDERED_PIXEL_SHADER(info->ps.pops);
}

typedef struct {
	const char* outpath;
	const struct nir_shader* nir;
	const struct radv_shader_info* rinfo;
	const struct radv_shader_args* rargs;
	const struct ac_shader_config* config;
	enum amd_gfx_level gfx_level;
	enum radeon_family family;
	mesa_shader_stage stage;
	bool neo;
} BuildContext;

static void buildshaderbinary(
    const BuildContext* ctx,
    const uint32_t* code, uint32_t code_dw
) {
	FILE* h = fopen(ctx->outpath, "w");
	if (!h) {
		fatalf("Failed to open output file with %s", strerror(errno));
	}

	uint32_t codesize = code_dw * sizeof(uint32_t);
	if (ctx->nir->info.stage == MESA_SHADER_VERTEX &&
	    ctx->rinfo->vs.has_prolog) {
		codesize += sizeof(uint32_t);
	}

	uint32_t* newcode = malloc(codesize);
	uint32_t* nextcode = newcode;

	/* prepend vertex shaders that use vertex input with essentially a
	 * jump instruction to a fetch "shader" */
	if (ctx->nir->info.stage == MESA_SHADER_VERTEX &&
	    ctx->rinfo->vs.has_prolog) {
		/* s_swappc_b64 s[0:1], s[0:1] */
		newcode[0] = 0xbe802100;
		nextcode += 1;
	}

	memcpy(nextcode, code, code_dw * sizeof(uint32_t));

	uint32_t numinputslots = 0;
	if (ctx->rargs->prolog_inputs.used) {
		numinputslots += 1;
	}
	if (ctx->rargs->ac.vertex_buffers.used) {
		numinputslots += 1;
	}
	if (ctx->rargs->descriptors[0].used) {
		numinputslots += 1;
	}

	uint32_t shspecificsize = headershsize(ctx->stage);
	shspecificsize += numinputslots * sizeof(GnmInputUsageSlot);

	/* exclude position as it doesn't count as an export semantic in PSB */
	const uint32_t outputswritten =
	    util_bitcount64(ctx->nir->info.outputs_written & ~VARYING_BIT_POS);

	switch (ctx->stage) {
	case MESA_SHADER_VERTEX:
		shspecificsize += util_bitcount64(ctx->nir->info.inputs_read) *
				  sizeof(GnmVertexInputSemantic);
		shspecificsize +=
		    outputswritten * sizeof(GnmVertexExportSemantic);
		break;
	case MESA_SHADER_FRAGMENT:
		shspecificsize += util_bitcount64(ctx->nir->info.inputs_read) *
				  sizeof(GnmPixelInputSemantic);
		break;
	default:
		break;
	}

	/* GCN bytecode that comes after the headers must be 4 byte aligned */
	const uint32_t alignbytes =
	    ((shspecificsize + 3) & (-4)) - shspecificsize;
	shspecificsize += alignbytes;

	const PsslBinaryHeader psbh = {
	    .vermajor = 0,
	    .verminor = 4,
	    .shadertype = psbshtype(ctx->stage),
	    .codetype = PSSL_CODE_ISA,
	    .compilertype = PSSL_COMPILER_UNSPECIFIED,
	    .codesize = sizeof(GnmShaderFileHeader) + shspecificsize +
			codesize + sizeof(GnmShaderBinaryInfo),
	};
	fwrite(&psbh, 1, sizeof(psbh), h);

	const GnmShaderFileHeader gsfh = {
	    .magic = GNM_SHADER_FILE_HEADER_ID,
	    .vermajor = 7,
	    .verminor = 2,
	    .type = gnmshtype(ctx->stage),
	    .headersizedwords = shspecificsize / 4,
	    .targetgpumodes =
		(ctx->gfx_level >= GFX10_3) ? GNM_TARGETGPUMODE_NEO :
		(ctx->neo ? GNM_TARGETGPUMODE_NEO : GNM_TARGETGPUMODE_BASE),
	};
	fwrite(&gsfh, 1, sizeof(gsfh), h);

	switch (ctx->stage) {
	case MESA_SHADER_VERTEX: {
		const uint32_t nparams =
		    MAX2(ctx->rinfo->outinfo.param_exports, 1);
		unsigned clip_dist_mask = 0, cull_dist_mask = 0;
		const unsigned num_pos_exports =
		    get_num_pos_exports(ctx->rinfo, &clip_dist_mask, &cull_dist_mask);
		const uint32_t total_mask = clip_dist_mask | cull_dist_mask;
		const bool misc_vec_ena =
		    ctx->rinfo->outinfo.writes_pointsize ||
		    ctx->rinfo->outinfo.writes_layer ||
		    ctx->rinfo->outinfo.writes_viewport_index ||
		    ctx->rinfo->outinfo.writes_primitive_shading_rate;

		const GnmVsShader vsh = {
		    .common =
			{
			    .shadersize =
				codesize + sizeof(GnmShaderBinaryInfo),
			    .numinputusageslots = numinputslots,
			},
		    .registers =
			{
			    .spishaderpgmlovs = shspecificsize,
			    .spishaderpgmhivs = 0xffffffff,
			    .spishaderpgmrsrc1vs = ctx->config->rsrc1,
			    .spishaderpgmrsrc2vs = ctx->config->rsrc2,
			    .spivsoutconfig =
				S_0286C4_VS_EXPORT_COUNT(nparams - 1),
			    .spishaderposformat =
				S_02870C_POS0_EXPORT_FORMAT(
				    V_02870C_SPI_SHADER_4COMP
				) |
				S_02870C_POS1_EXPORT_FORMAT(
				    num_pos_exports > 1
					? V_02870C_SPI_SHADER_4COMP
					: V_02870C_SPI_SHADER_NONE
				) |
				S_02870C_POS2_EXPORT_FORMAT(
				    num_pos_exports > 2
					? V_02870C_SPI_SHADER_4COMP
					: V_02870C_SPI_SHADER_NONE
				) |
				S_02870C_POS3_EXPORT_FORMAT(
				    num_pos_exports > 3
					? V_02870C_SPI_SHADER_4COMP
					: V_02870C_SPI_SHADER_NONE
				),
			    .paclvsoutcntl =
				S_02881C_USE_VTX_POINT_SIZE(
				    ctx->rinfo->outinfo.writes_pointsize
				) |
				S_02881C_USE_VTX_RENDER_TARGET_INDX(
				    ctx->rinfo->outinfo.writes_layer
				) |
				S_02881C_USE_VTX_VIEWPORT_INDX(
				    ctx->rinfo->outinfo.writes_viewport_index
				) |
				S_02881C_USE_VTX_VRS_RATE(
				    ctx->rinfo->outinfo
					.writes_primitive_shading_rate
				) |
				S_02881C_VS_OUT_MISC_VEC_ENA(misc_vec_ena) |
				S_02881C_VS_OUT_MISC_SIDE_BUS_ENA(
				    misc_vec_ena ||
				    (ctx->gfx_level >= GFX10_3 &&
				     num_pos_exports > 1)
				) |
				S_02881C_VS_OUT_CCDIST0_VEC_ENA(
				    (total_mask & 0x0f) != 0
				) |
				S_02881C_VS_OUT_CCDIST1_VEC_ENA(
				    (total_mask & 0xf0) != 0
				) |
				total_mask << 8 |
				clip_dist_mask,
			},
		    .numinputsemantics =
			util_bitcount64(ctx->nir->info.inputs_read),
		    .numexportsemantics = outputswritten,
		};
		fwrite(&vsh, 1, sizeof(vsh), h);
		break;
	}
	case MESA_SHADER_FRAGMENT: {
		const bool param_gen = ctx->gfx_level >= GFX11 &&
				       !ctx->rinfo->ps.num_inputs &&
				       ctx->config->lds_size;

		const GnmPsShader psh = {
		    .common =
			{
			    .shadersize =
				codesize + sizeof(GnmShaderBinaryInfo),
			    .numinputusageslots = numinputslots,
			},
		    .registers =
			{
			    .spishaderpgmlops = shspecificsize,
			    .spishaderpgmhips = 0xffffffff,
			    .spishaderpgmrsrc1ps = ctx->config->rsrc1,
			    .spishaderpgmrsrc2ps = ctx->config->rsrc2,
			    .spishaderzformat = ac_get_spi_shader_z_format(
				ctx->rinfo->ps.writes_z,
				ctx->rinfo->ps.writes_stencil,
				ctx->rinfo->ps.writes_sample_mask,
				ctx->rinfo->ps.writes_mrt0_alpha
			    ),
			    .spishadercolformat =
				ctx->rinfo->ps.spi_shader_col_format,
			    .spipsinputena = ctx->config->spi_ps_input_ena,
			    .spipsinputaddr = ctx->config->spi_ps_input_addr,
			    .spipsincontrol =
				/* On GFX10.3, NUM_INTERP is not precomputed because
				 * per-primitive attributes are tracked separately
				 * in NUM_PRIM_INTERP. */
				(ctx->gfx_level != GFX10_3
				     ? S_0286D8_NUM_INTERP(ctx->rinfo->ps.num_inputs)
				     : 0) |
				S_0286D8_PS_W32_EN(
				    ctx->rinfo->wave_size == 32
				) |
				S_0286D8_PARAM_GEN(param_gen),
			    .spibaryccntl = S_0286E0_FRONT_FACE_ALL_BITS(1),
			    .dbshadercontrol = compute_db_shader_control(ctx->rinfo),
			    .cbshadermask = ac_get_cb_shader_mask(
				ctx->rinfo->ps.spi_shader_col_format
			    ),
			},
		    .numinputsemantics =
			util_bitcount64(ctx->nir->info.inputs_read),
		};
		fwrite(&psh, 1, sizeof(psh), h);
		break;
	}
	default:
		fatalf("Unhandled shader type %u", ctx->stage);
	}

	/* write shader common data: input usage slots */
	if (ctx->rargs->prolog_inputs.used) {
		const GnmInputUsageSlot s = {
		    .usagetype = GNM_SHINPUTUSAGE_SUBPTR_FETCHSHADER,
		    .startregister =
			ctx->rargs->ac.args[ctx->rargs->prolog_inputs.arg_index]
			    .offset,
		};
		fwrite(&s, 1, sizeof(s), h);
	}
	if (ctx->rargs->ac.vertex_buffers.used) {
		const GnmInputUsageSlot s = {
		    .usagetype = GNM_SHINPUTUSAGE_PTR_VERTEXBUFFERTABLE,
		    .startregister =
			ctx->rargs->ac
			    .args[ctx->rargs->ac.vertex_buffers.arg_index]
			    .offset,
		};
		fwrite(&s, 1, sizeof(s), h);
	}
	if (ctx->rargs->descriptors[0].used) {
		const GnmInputUsageSlot s = {
		    .usagetype = GNM_SHINPUTUSAGE_PTR_INDIRECTRESOURCETABLE,
		    .startregister =
			ctx->rargs->ac
			    .args[ctx->rargs->descriptors[0].arg_index]
			    .offset,
		};
		fwrite(&s, 1, sizeof(s), h);
	}

	/* write input/export semantics */
	switch (ctx->stage) {
	case MESA_SHADER_VERTEX:
		for (uint32_t i = 0;
		     i < util_bitcount64(ctx->nir->info.inputs_read); i += 1) {
			const GnmVertexInputSemantic input = {
			    .semantic = i,
			    .vgpr =
				ctx->rargs->ac
				    .args[ctx->rargs->vs_inputs[i].arg_index]
				    .offset,
			    .sizeinelements =
				ctx->rargs->ac
				    .args[ctx->rargs->vs_inputs[i].arg_index]
				    .size,
			};
			fwrite(&input, 1, sizeof(input), h);
		}
		for (uint32_t i = 0; i < outputswritten; i += 1) {
			const GnmVertexExportSemantic out = {
			    .semantic = 15 + i,
			    .outindex = i,
			    .exportf16 = 0,
			};
			fwrite(&out, 1, sizeof(out), h);
		}
		break;
	case MESA_SHADER_FRAGMENT:
		for (uint32_t i = 0;
		     i < util_bitcount64(ctx->nir->info.inputs_read); i += 1) {
			const GnmPixelInputSemantic input = {
			    .semantic = 15 + i,
			};
			fwrite(&input, 1, sizeof(input), h);
		}
		break;
	default:
		break;
	}

	/* alignment padding */
	for (uint32_t i = 0; i < alignbytes; i += 1) {
		uint8_t val = 0;
		fwrite(&val, 1, sizeof(val), h);
	}

	/* shader code */
	fwrite(newcode, 1, codesize, h);

	GnmShaderBinaryInfo bininfo = {
	    .signature = GNM_SHADER_BINARY_INFO_MAGIC,
	    .version = 7,
	    .ispsslcg = 1,
	    .type = shbintype(ctx->stage),
	    .length = codesize,
	};
	bininfo.crc32 = hashsb(newcode, codesize, &bininfo);
	fwrite(&bininfo, 1, sizeof(bininfo), h);

	const PsslBinaryParamInfo paraminfo = {0};
	fwrite(&paraminfo, 1, sizeof(paraminfo), h);

	free(newcode);
	fclose(h);
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
	if (!opts.outputfile) {
		fatal("Please pass an output file");
	}
	if (opts.stage == MESA_SHADER_NONE) {
		fatal("Please pass a valid shader stage");
	}

	FILE* inputhandle = fopen(opts.inputfile, "r");
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

	/* Select GFX level and family based on target */
	enum amd_gfx_level gfxlevel;
	enum radeon_family chipfamily;
	bool neo = false;

	switch (opts.target) {
	case GFX_TARGET_PS5:
		/* PS5 GPU is RDNA2 (GFX10.3).
		 * Use CHIP_NAVI21 as the closest family match.
		 * The family value mainly affects shader compilation heuristics
		 * and should not significantly alter the generated ISA. */
		gfxlevel = GFX10_3;
		chipfamily = CHIP_NAVI21;
		break;
	case GFX_TARGET_PS4_NEO:
		/* PS4 Pro is GFX8 (Volcanic Islands/Polaris) */
		gfxlevel = GFX8;
		chipfamily = CHIP_TONGA;
		neo = true;
		break;
	case GFX_TARGET_PS4_BASE:
	default:
		/* PS4 base is GFX7 (Sea Islands) */
		gfxlevel = GFX7;
		chipfamily = CHIP_KAVERI;
		break;
	}

	glsl_type_singleton_init_or_ref();

	/* Construct ac_compiler_info for the target GPU */
	struct ac_compiler_info ac_info = {0};
	ac_info.gfx_level = gfxlevel;

	/* Fill in reasonable defaults for the target */
	if (gfxlevel >= GFX10_3) {
		ac_info.max_waves_per_simd = 16;
		ac_info.num_physical_sgprs_per_simd = 128 * 16;
		ac_info.num_physical_wave64_vgprs_per_simd = 256;
		ac_info.num_simd_per_compute_unit = 2;
		ac_info.min_sgpr_alloc = 1;
		ac_info.max_sgpr_alloc = 128;
		ac_info.sgpr_alloc_granularity = 1;
		ac_info.min_wave64_vgpr_alloc = 4;
		ac_info.max_vgpr_alloc = 256;
		ac_info.wave64_vgpr_alloc_granularity = 4;
		ac_info.has_packed_math_16bit = true;
		ac_info.has_fma_mix = true;
	} else if (gfxlevel == GFX10) {
		ac_info.max_waves_per_simd = 20;
		ac_info.num_physical_sgprs_per_simd = 128 * 20;
		ac_info.num_physical_wave64_vgprs_per_simd = 256;
		ac_info.num_simd_per_compute_unit = 2;
		ac_info.min_sgpr_alloc = 1;
		ac_info.max_sgpr_alloc = 128;
		ac_info.sgpr_alloc_granularity = 1;
		ac_info.min_wave64_vgpr_alloc = 4;
		ac_info.max_vgpr_alloc = 256;
		ac_info.wave64_vgpr_alloc_granularity = 4;
		ac_info.has_packed_math_16bit = true;
	} else if (gfxlevel == GFX8) {
		ac_info.max_waves_per_simd = 10;
		ac_info.num_physical_sgprs_per_simd = 16 * 10;
		ac_info.num_physical_wave64_vgprs_per_simd = 128;
		ac_info.num_simd_per_compute_unit = 1;
		ac_info.min_sgpr_alloc = 1;
		ac_info.max_sgpr_alloc = 16;
		ac_info.sgpr_alloc_granularity = 1;
		ac_info.min_wave64_vgpr_alloc = 4;
		ac_info.max_vgpr_alloc = 128;
		ac_info.wave64_vgpr_alloc_granularity = 4;
	} else {
		/* GFX7 */
		ac_info.max_waves_per_simd = 10;
		ac_info.num_physical_sgprs_per_simd = 12 * 10;
		ac_info.num_physical_wave64_vgprs_per_simd = 64;
		ac_info.num_simd_per_compute_unit = 1;
		ac_info.min_sgpr_alloc = 1;
		ac_info.max_sgpr_alloc = 12;
		ac_info.sgpr_alloc_granularity = 1;
		ac_info.min_wave64_vgpr_alloc = 4;
		ac_info.max_vgpr_alloc = 64;
		ac_info.wave64_vgpr_alloc_granularity = 4;
	}

	/* Construct radv_compiler_info */
	struct radv_compiler_info compiler_info = {0};
	compiler_info.ac = &ac_info;
	compiler_info.hw.address32_hi = 0;
	compiler_info.key.ge_wave_size = 64;
	compiler_info.key.ps_wave_size = (gfxlevel >= GFX10_3) ? 32 : 64;
	compiler_info.key.cs_wave_size = (gfxlevel >= GFX10_3) ? 32 : 64;
	compiler_info.key.rt_wave_size = 64;
	compiler_info.key.family = chipfamily;
	compiler_info.key.load_grid_size_from_user_sgpr = (gfxlevel >= GFX10_3);

	/* Initialize NIR options for all stages */
	radv_get_nir_options(&compiler_info);

	/* Construct radv_shader_stage */
	struct radv_shader_stage stage = {0};
	stage.stage = opts.stage;
	stage.next_stage = (opts.stage == MESA_SHADER_VERTEX) ? MESA_SHADER_FRAGMENT : MESA_SHADER_NONE;
	stage.spirv.data = (const char*)input;
	stage.spirv.size = inputlen;
	stage.entrypoint = opts.entrypoint;
	stage.key.optimisations_disabled = !opts.optimise;

	/* SPIR-V to NIR */
	const struct radv_spirv_to_nir_options spirv_options = {
	    .lower_view_index_to_zero = true,
	    .lower_view_index_to_device_index = false,
	};

	nir_shader* nir = radv_shader_spirv_to_nir(
	    &compiler_info, &stage, &spirv_options, false
	);

	if (!nir) {
		fatal("SPIRV to NIR compilation failed");
	}

	if (opts.verbose) {
		puts("========================");
		puts("BEFORE NIR OPTIMISATIONS");
		puts("========================");
		nir_print_shader(nir, stdout);
	}

	/* Optimize NIR */
	radv_optimize_nir(nir, !opts.optimise);

	/* Gather info again — outputs_read can be out-of-date */
	nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
	radv_nir_lower_io(nir);

	if (opts.verbose) {
		puts("=======================");
		puts("AFTER NIR OPTIMISATIONS");
		puts("=======================");
		nir_print_shader(nir, stdout);
	}

	/* Shader info + args + postprocess */
	radv_nir_shader_info_init(stage.stage, stage.next_stage, &stage.info);

	struct radv_shader_layout layout = {0};
	struct radv_graphics_state_key gfx_state = {0};
	gfx_state.ps.epilog.spi_shader_col_format = V_028714_SPI_SHADER_FP16_ABGR;
	gfx_state.ps.epilog.color_is_int8 = 0xff;
	gfx_state.ps.has_epilog = false;

	radv_nir_shader_info_pass(
	    &compiler_info, nir, &layout, &stage.key, &gfx_state,
	    RADV_PIPELINE_GRAPHICS, false, &stage.info
	);

	radv_declare_shader_args(
	    &compiler_info, &gfx_state, &stage, MESA_SHADER_NONE, NULL
	);

	stage.nir = nir;
	stage.info.user_sgprs_locs = stage.args.user_sgprs_locs;
	stage.info.inline_push_constant_mask = stage.args.ac.inline_push_const_mask;

	radv_postprocess_nir(
	    &compiler_info, &gfx_state, &stage
	);

	if (opts.verbose) {
		puts("==========================");
		puts("AFTER RADV POSTPROCESS NIR");
		puts("==========================");
		nir_print_shader(nir, stdout);
	}

	/* Compile NIR to GCN ISA via ACO */
	struct radv_shader_binary* binary = radv_shader_nir_to_asm(
	    &compiler_info, &stage, &nir, 1, &gfx_state
	);

	if (!binary) {
		fatal("ACO shader compilation failed");
	}

	/* Extract code from radv_shader_binary_legacy */
	struct radv_shader_binary_legacy* legacy =
	    (struct radv_shader_binary_legacy*)binary;

	/* The data layout in radv_shader_binary_legacy is:
	 * [stats | code | ir | disasm | debug_info]
	 * Code starts at offset stats_size. */
	const uint32_t* code = (const uint32_t*)(legacy->data + legacy->stats_size);
	uint32_t code_dw = legacy->code_size / sizeof(uint32_t);

	/* Build the PS4/PS5 shader binary file */
	const BuildContext buildctx = {
	    .outpath = opts.outputfile,
	    .rinfo = &binary->info,
	    .rargs = &stage.args,
	    .config = &binary->config,
	    .nir = nir,
	    .gfx_level = gfxlevel,
	    .family = chipfamily,
	    .stage = opts.stage,
	    .neo = neo,
	};
	buildshaderbinary(&buildctx, code, code_dw);

	free(binary);
	glsl_type_singleton_decref();

	return EXIT_SUCCESS;
}
