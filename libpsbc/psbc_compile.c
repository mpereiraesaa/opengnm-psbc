/*
 * libpsbc — SPIR-V to PS4/PS5 Shader Binary Compiler Library
 *
 * Extracted from cmd/psbc/main.c. Contains the full compilation pipeline:
 *   SPIR-V → NIR (via radv_shader_spirv_to_nir)
 *   NIR optimization (radv_optimize_nir, radv_nir_lower_io)
 *   NIR postprocessing (radv_postprocess_nir)
 *   NIR → GCN ISA (via ACO: radv_shader_nir_to_asm)
 *   GCN ISA → GnmShaderFileHeader (buildshaderbinary)
 */

#include "psbc_compile.h"

#include <errno.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#include <pssl_types.h>

#include "aco_interface.h"
#include "ac_gpu_info.h"
#include "ac_shader_util.h"
#include "ac_binary.h"
#include "ac_nir.h"
#include "amd_family.h"
#include "nir/nir_builder.h"
#include "nir/radv_nir.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "radv_shader_info.h"
#include "radv_descriptor_set.h"
#include "radv_aco_shader_info.h"
#include "radv_pipeline.h"
#include "sid.h"
#include "spirv/spirv.h"

#include "crc32_sb.h"

/* Keep standalone PSBC ABI-compatible with the pinned Mesa NIR build. */
_Static_assert(sizeof(nir_instr_type) == 1, "NIR enums must be packed");
_Static_assert(sizeof(nir_intrinsic_op) == 4, "unexpected NIR intrinsic enum size");
_Static_assert(offsetof(nir_intrinsic_instr, intrinsic) == 56,
               "PSBC/Mesa NIR layout mismatch");

/* ACO packs color exports into consecutive MRT slots. Match RADV's final
 * register emission, but keep CB_SHADER_MASK in logical attachment order. */
static uint32_t compact_spi_shader_col_format(uint32_t formats) {
    uint32_t compacted = 0;
    unsigned slot = 0;
    for (unsigned i = 0; i < 8; ++i) {
        uint32_t format = (formats >> (4 * i)) & 0xf;
        if (format)
            compacted |= format << (4 * slot++);
    }
    return compacted;
}

static void debug_shader_io(const char* label, const nir_shader* nir,
                            const struct radv_shader_info* info) {
    if (!getenv("PSBC_DEBUG_IO"))
        return;
    fprintf(stderr,
            "PSBC IO %s: inputs=0x%016" PRIx64
            " outputs=0x%016" PRIx64
            " ps_inputs=%u ps_mask=0x%08x params=%u prim_params=%u"
            " linked=%u/%u merged_separate=%u slots=%u/%u\n",
            label, nir->info.inputs_read, nir->info.outputs_written,
            info ? info->ps.num_inputs : 0,
            info ? info->ps.input_mask : 0,
            info ? info->outinfo.param_exports : 0,
            info ? info->outinfo.prim_param_exports : 0,
            info ? info->inputs_linked : 0,
            info ? info->outputs_linked : 0,
            info ? info->merged_shader_compiled_separately : 0,
            info && info->stage == MESA_SHADER_VERTEX
                ? info->vs.num_linked_outputs : 0,
            info && info->stage == MESA_SHADER_GEOMETRY
                ? info->gs.num_linked_inputs : 0);
}

static void debug_stage(const char* label) {
    if (!getenv("PSBC_DEBUG_STAGE"))
        return;
    fprintf(stderr, "PSBC stage %s\n", label);
    fflush(stderr);
}

/* Gallium NIR names constant buffers with a scalar slot. RADV's descriptor
 * lowering expects the Vulkan resource-index tuple produced by SPIR-V import.
 * Convert constant scalar slots at the standalone API boundary so both input
 * forms use the same descriptor pipeline. */
static bool lower_gallium_ubo_index(nir_builder* b, nir_instr* instruction,
                                    void* data) {
    nir_intrinsic_instr* intrinsic;
    nir_src slot_src;
    uint64_t slot;
    nir_def* resource;

    (void)data;
    if (instruction->type != nir_instr_type_intrinsic)
        return false;
    intrinsic = nir_instr_as_intrinsic(instruction);
    if (intrinsic->intrinsic != nir_intrinsic_load_ubo ||
        intrinsic->src[0].ssa->num_components != 1)
        return false;

    slot_src = intrinsic->src[0];
    if (!nir_src_is_const(slot_src))
        return false;
    slot = nir_src_as_uint(slot_src);
    if (slot >= PSBC_MAX_DESCRIPTOR_BINDINGS -
                    PSBC_GALLIUM_UBO_BINDING_BASE)
        return false;

    b->cursor = nir_before_instr(instruction);
    resource = nir_vulkan_resource_index(
        b, 3, 32, nir_imm_int(b, 0),
        .desc_set = 0,
        .binding = PSBC_GALLIUM_UBO_BINDING_BASE + (unsigned)slot,
        .desc_type = nir_descriptor_type_uniform_buffer,
        .resource_type = nir_resource_type_uniform_buffer);
    nir_src_rewrite(&intrinsic->src[0], resource);
    return true;
}

/* === Mesa stage mapping === */

static mesa_shader_stage psbc_to_mesa_stage(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:     return MESA_SHADER_VERTEX;
    case PSBC_STAGE_TESS_CTRL:  return MESA_SHADER_TESS_CTRL;
    case PSBC_STAGE_TESS_EVAL:  return MESA_SHADER_TESS_EVAL;
    case PSBC_STAGE_GEOMETRY:   return MESA_SHADER_GEOMETRY;
    case PSBC_STAGE_FRAGMENT:   return MESA_SHADER_FRAGMENT;
    case PSBC_STAGE_COMPUTE:    return MESA_SHADER_COMPUTE;
    case PSBC_STAGE_TASK:       return MESA_SHADER_TASK;
    case PSBC_STAGE_EXPORT:     return MESA_SHADER_VERTEX;  /* ES is a VS variant */
    case PSBC_STAGE_LOCAL:      return MESA_SHADER_VERTEX;  /* LS is a VS variant */
    default:                    return MESA_SHADER_NONE;
    }
}

/* === Init/shutdown (refcounted) === */

static int g_init_refcount = 0;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

void psbc_init(void) {
    pthread_mutex_lock(&g_init_mutex);
    if (g_init_refcount++ == 0) {
        glsl_type_singleton_init_or_ref();
    }
    pthread_mutex_unlock(&g_init_mutex);
}

void psbc_shutdown(void) {
    pthread_mutex_lock(&g_init_mutex);
    if (g_init_refcount > 0 && --g_init_refcount == 0) {
        glsl_type_singleton_decref();
    }
    pthread_mutex_unlock(&g_init_mutex);
}

/* === PSSL/GNM type mapping === */

static PsslShaderType psbshtype(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:      return PSSL_SHADER_VS;
    case PSBC_STAGE_TESS_EVAL:   return PSSL_SHADER_VS;  /* DS outputs vertices */
    case PSBC_STAGE_EXPORT:      return PSSL_SHADER_VS;  /* ES is a VS variant */
    case PSBC_STAGE_LOCAL:       return PSSL_SHADER_VS;  /* LS is a VS variant */
    case PSBC_STAGE_FRAGMENT:    return PSSL_SHADER_FS;
    case PSBC_STAGE_COMPUTE:     return PSSL_SHADER_CS;
    case PSBC_STAGE_GEOMETRY:    return PSSL_SHADER_VS;  /* no PSSL GS type; use VS */
    case PSBC_STAGE_TESS_CTRL:   return PSSL_SHADER_VS;  /* no PSSL HS type; use VS */
    default:                      return PSSL_SHADER_VS;
    }
}

static GnmShaderType gnmshtype(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:      return GNM_SHADER_VERTEX;
    case PSBC_STAGE_TESS_EVAL:   return GNM_SHADER_VERTEX;  /* DS outputs vertices */
    case PSBC_STAGE_EXPORT:      return GNM_SHADER_VERTEX;  /* ES is a VS variant */
    case PSBC_STAGE_LOCAL:       return GNM_SHADER_VERTEX;  /* LS is a VS variant */
    case PSBC_STAGE_FRAGMENT:    return GNM_SHADER_PIXEL;
    case PSBC_STAGE_COMPUTE:     return GNM_SHADER_COMPUTE;
    case PSBC_STAGE_GEOMETRY:    return GNM_SHADER_GEOMETRY;
    case PSBC_STAGE_TESS_CTRL:   return GNM_SHADER_HULL;
    default:                      return GNM_SHADER_VERTEX;
    }
}

static GnmShaderBinaryType shbintype(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:      return GNM_SHB_VS_VS;
    case PSBC_STAGE_TESS_EVAL:   return GNM_SHB_DS_VS;
    case PSBC_STAGE_EXPORT:      return GNM_SHB_VS_ES;
    case PSBC_STAGE_LOCAL:       return GNM_SHB_VS_LS;
    case PSBC_STAGE_FRAGMENT:    return GNM_SHB_PS;
    case PSBC_STAGE_COMPUTE:     return GNM_SHB_CS;
    case PSBC_STAGE_GEOMETRY:    return GNM_SHB_GS;
    case PSBC_STAGE_TESS_CTRL:   return GNM_SHB_HS;
    default:                      return GNM_SHB_VS_VS;
    }
}

static uint32_t headershsize(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:      return sizeof(GnmVsShader);
    case PSBC_STAGE_TESS_EVAL:   return sizeof(GnmVsShader);  /* DS uses VS struct */
    case PSBC_STAGE_EXPORT:      return sizeof(GnmEsShader);
    case PSBC_STAGE_LOCAL:       return sizeof(GnmLsShader);
    case PSBC_STAGE_FRAGMENT:    return sizeof(GnmPsShader);
    case PSBC_STAGE_COMPUTE:     return sizeof(GnmCsShader);
    case PSBC_STAGE_GEOMETRY:    return sizeof(GnmGsShader);
    case PSBC_STAGE_TESS_CTRL:   return sizeof(GnmHsShader);
    default:                      return sizeof(GnmVsShader);
    }
}

static uint32_t hashsb(
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

/* === Shader binary builder === */

typedef struct {
    bool valid;
    uint32_t count;
    uint32_t words[PSBC_MAX_SEMANTICS];
} PsbcInputSemantics;

typedef struct {
    const struct nir_shader* nir;
    const struct radv_shader_info* rinfo;
    const struct radv_shader_args* rargs;
    const struct ac_shader_config* config;
    enum amd_gfx_level gfx_level;
    enum radeon_family family;
    mesa_shader_stage stage;
    PsbcStage psbc_stage;
    const uint32_t* spirv_data;
    size_t spirv_size;
    PsbcTarget target;
    bool ngg;
    bool neo;
    uint32_t address32_hi;
    const PsbcCompileOptions* options;
    const PsbcInputSemantics* input_semantics;
    /* Bindings the compiled stage statically uses, per set; zero when the caller
     * did not request static descriptor use. */
    uint64_t descriptor_used_binding_mask[PSBC_MAX_DESCRIPTOR_SETS];
} BuildContext;

static unsigned ps5_last_provoking_vertex(uint32_t primitive_type) {
    switch (primitive_type) {
    case 1: /* point list */
        return 0;
    case 2: /* line list */
    case 3: /* line strip */
        return 1;
    case 4: /* triangle list */
    case 5: /* triangle fan */
    case 6: /* triangle strip */
        return 2;
    default:
        return 0;
    }
}

static bool lower_flat_input_vertex(nir_builder* builder,
                                    nir_intrinsic_instr* intrinsic,
                                    void* data) {
    const PsbcCompileOptions* options = data;
    const unsigned vertex_id = ps5_last_provoking_vertex(options->primitive_type);
    if (intrinsic->intrinsic != nir_intrinsic_load_input)
        return false;
    /* An implicit NGG PrimitiveID has one value per primitive, not P0/P1/P2.
     * Explicit GS outputs still obey the ordinary provoking-vertex rule. */
    if (options->primitive_id_per_primitive &&
        nir_intrinsic_io_semantics(intrinsic).location == VARYING_SLOT_PRIMITIVE_ID)
        return false;

    builder->cursor = nir_before_instr(&intrinsic->instr);
    nir_def* replacement = nir_load_input_vertex(
        builder, intrinsic->def.num_components, intrinsic->def.bit_size,
        nir_imm_int(builder, vertex_id), intrinsic->src[0].ssa,
        .base = nir_intrinsic_base(intrinsic),
        .component = nir_intrinsic_component(intrinsic),
        .dest_type = nir_intrinsic_dest_type(intrinsic),
        .io_semantics = nir_intrinsic_io_semantics(intrinsic));
    nir_def_replace(&intrinsic->def, replacement);
    return true;
}

static unsigned ps5_input_interpolation_mode(
    const nir_intrinsic_instr* intrinsic) {
    if (intrinsic->intrinsic == nir_intrinsic_load_interpolated_input)
        return 2;
    if (intrinsic->intrinsic == nir_intrinsic_load_input ||
        intrinsic->intrinsic == nir_intrinsic_load_input_vertex)
        return 1;
    return 0;
}

typedef struct {
    uint8_t modes[PSBC_MAX_SEMANTICS];
    uint8_t flat_attribute[PSBC_MAX_SEMANTICS];
    uint8_t primitive_id_attribute;
} Ps5MixedInputState;

static bool classify_ps5_input(nir_builder* builder,
                               nir_instr* instruction, void* data) {
    (void)builder;
    if (instruction->type != nir_instr_type_intrinsic)
        return false;
    nir_intrinsic_instr* intrinsic = nir_instr_as_intrinsic(instruction);
    const unsigned mode = ps5_input_interpolation_mode(intrinsic);
    if (!mode)
        return false;
    const nir_io_semantics io = nir_intrinsic_io_semantics(intrinsic);
    if (io.location < VARYING_SLOT_VAR0)
        return false;
    for (unsigned slot = 0; slot < io.num_slots; ++slot) {
        const unsigned location = io.location - VARYING_SLOT_VAR0 + slot;
        if (location < PSBC_MAX_SEMANTICS)
            ((Ps5MixedInputState*)data)->modes[location] |= mode;
    }
    return false;
}

static bool remap_ps5_flat_input(nir_builder* builder,
                                 nir_instr* instruction, void* data) {
    (void)builder;
    if (instruction->type != nir_instr_type_intrinsic)
        return false;
    nir_intrinsic_instr* intrinsic = nir_instr_as_intrinsic(instruction);
    if (ps5_input_interpolation_mode(intrinsic) != 1)
        return false;
    const nir_io_semantics io = nir_intrinsic_io_semantics(intrinsic);
    const Ps5MixedInputState* state = data;
    if (io.location == VARYING_SLOT_PRIMITIVE_ID &&
        state->primitive_id_attribute != UINT8_MAX) {
        nir_intrinsic_set_base(intrinsic, state->primitive_id_attribute);
        return true;
    }
    if (io.location < VARYING_SLOT_VAR0 || io.num_slots != 1)
        return false;
    const unsigned location = io.location - VARYING_SLOT_VAR0;
    if (location >= PSBC_MAX_SEMANTICS || state->modes[location] != 3)
        return false;
    nir_intrinsic_set_base(intrinsic, state->flat_attribute[location]);
    return true;
}

static bool split_ps5_mixed_inputs(nir_shader* nir,
                                   struct radv_shader_info* info,
                                   bool primitive_id_per_primitive) {
    Ps5MixedInputState state = {.primitive_id_attribute = UINT8_MAX};
    memset(state.flat_attribute, UINT8_MAX, sizeof(state.flat_attribute));
    nir_shader_instructions_pass(nir, classify_ps5_input,
                                 nir_metadata_all, &state);
    const bool per_primitive_id = primitive_id_per_primitive && info->ps.prim_id_input;
    unsigned next = info->ps.num_inputs - per_primitive_id;
    for (unsigned location = 0; location < PSBC_MAX_SEMANTICS; ++location) {
        if (state.modes[location] != 3)
            continue;
        if (next >= PSBC_MAX_SEMANTICS)
            return false;
        state.flat_attribute[location] = next++;
    }
    /* GFX10.3 requires per-primitive inputs after every per-vertex input,
     * including the extra flat aliases created above. */
    if (per_primitive_id) {
        if (next >= PSBC_MAX_SEMANTICS)
            return false;
        state.primitive_id_attribute = next++;
    }
    if (next == info->ps.num_inputs)
        return true;
    nir_shader_instructions_pass(nir, remap_ps5_flat_input,
                                 nir_metadata_all, &state);
    info->ps.num_inputs = next;
    return true;
}

#define PSBC_CX_OFFSET(reg) ((uint16_t)(((reg) - SI_CONTEXT_REG_OFFSET) / 4))
#define PSBC_SH_OFFSET(reg) ((uint16_t)(((reg) - SI_SH_REG_OFFSET) / 4))
#define PSBC_UC_OFFSET(reg) ((uint16_t)(((reg) - CIK_UCONFIG_REG_OFFSET) / 4))

static void metadata_add_register(PsbcRegisterWrite* registers,
                                  uint32_t* count, uint32_t limit,
                                  uint16_t offset, uint32_t value) {
    if (*count >= limit)
        return;
    registers[*count] = (PsbcRegisterWrite) {
        .offset = offset,
        .value = value,
    };
    *count += 1;
}

/*
 * AGC semantic words use the low byte as the producer/consumer match key.
 * Generic user varyings start at semantic 15, matching both Sony-produced
 * AGC packages and the legacy GNM/PSSL metadata convention.
 * A producer additionally stores its parameter-export index in bits 8..12.
 * Pixel consumer bit 22 selects flat shading in PS5 AGC semantics.  This is
 * the PS5 counterpart of GnmPixelInputSemantic.isflatshaded and is consumed
 * by sceAgcLinkShaders when it builds SPI_PS_INPUT_CNTL_i.
 */
static bool build_input_semantics(const nir_shader* nir,
                                  const struct radv_shader_info* info,
                                  bool primitive_id_per_primitive,
                                  PsbcInputSemantics* result) {
    memset(result, 0, sizeof(*result));
    const uint32_t generic_count = info->ps.num_inputs;
    if (generic_count > PSBC_MAX_SEMANTICS)
        return false;
    if (info->ps.input_per_primitive_mask ||
        info->ps.explicit_shaded_mask || info->ps.explicit_strict_shaded_mask ||
        info->ps.float16_shaded_mask || info->ps.float16_hi_shaded_mask)
        return false;

    uint32_t words_by_attribute[PSBC_MAX_SEMANTICS] = {0};
    bool seen[PSBC_MAX_SEMANTICS] = {false};
    nir_foreach_function_impl(impl, nir) {
        nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
                if (instr->type != nir_instr_type_intrinsic)
                    continue;
                const nir_intrinsic_instr* intrin = nir_instr_as_intrinsic(instr);
                const unsigned mode = ps5_input_interpolation_mode(intrin);
                if (!mode)
                    continue;
                const nir_io_semantics io = nir_intrinsic_io_semantics(intrin);
                const bool primitive_id = io.location == VARYING_SLOT_PRIMITIVE_ID;
                if (primitive_id && (io.num_slots != 1 || mode != 1))
                    return false;
                if (!primitive_id && io.location < VARYING_SLOT_VAR0)
                    continue;
                for (uint32_t slot = 0; slot < io.num_slots; ++slot) {
                    const uint32_t location =
                        primitive_id ? 0 : io.location - VARYING_SLOT_VAR0 + slot;
                    const uint32_t attribute =
                        nir_intrinsic_base(intrin) + slot;
                    if (location >= PSBC_MAX_SEMANTICS ||
                        attribute >= generic_count)
                        return false;
                    uint32_t word = primitive_id ? PSBC_SEMANTIC_PRIMITIVE_ID : 15u + location;
                    if (mode == 1 && !(primitive_id && primitive_id_per_primitive))
                        word |= BITFIELD_BIT(22);
                    if (seen[attribute] &&
                        words_by_attribute[attribute] != word)
                        return false;
                    words_by_attribute[attribute] = word;
                    seen[attribute] = true;
                }
            }
        }
    }

    for (uint32_t attribute = 0; attribute < generic_count; ++attribute) {
        if (!seen[attribute])
            return false;
        result->words[result->count++] = words_by_attribute[attribute];
    }
    result->valid = true;
    return true;
}

static bool fill_output_semantics(const struct radv_shader_info* info,
                                  PsbcShaderMetadata* metadata) {
    for (unsigned semantic = 0; semantic < PSBC_MAX_SEMANTICS; ++semantic) {
        const uint8_t parameter =
            info->outinfo.vs_output_param_offset[VARYING_SLOT_VAR0 + semantic];
        if (parameter < PSBC_MAX_SEMANTICS) {
            metadata->output_semantics[metadata->output_semantic_count++] =
                (15u + semantic) | ((uint32_t)parameter << 8);
        }
    }
    const uint8_t primitive_id =
        info->outinfo.vs_output_param_offset[VARYING_SLOT_PRIMITIVE_ID];
    if (primitive_id < PSBC_MAX_SEMANTICS) {
        if (metadata->output_semantic_count >= PSBC_MAX_SEMANTICS)
            return false;
        metadata->output_semantics[metadata->output_semantic_count++] =
            PSBC_SEMANTIC_PRIMITIVE_ID | ((uint32_t)primitive_id << 8);
    }
    return metadata->output_semantic_count ==
        info->outinfo.param_exports + info->outinfo.prim_param_exports;
}

static uint32_t build_pa_cl_vs_out_cntl(const BuildContext* ctx,
                                        unsigned num_pos_exports) {
    unsigned clip_dist_mask = 0, cull_dist_mask = 0;
    get_num_pos_exports(ctx->rinfo, &clip_dist_mask, &cull_dist_mask);
    const uint32_t total_mask = clip_dist_mask | cull_dist_mask;
    const bool misc_vec_ena =
        ctx->rinfo->outinfo.writes_pointsize ||
        ctx->rinfo->outinfo.writes_layer ||
        ctx->rinfo->outinfo.writes_viewport_index ||
        ctx->rinfo->outinfo.writes_primitive_shading_rate;

    return S_02881C_USE_VTX_POINT_SIZE(ctx->rinfo->outinfo.writes_pointsize) |
           S_02881C_USE_VTX_RENDER_TARGET_INDX(ctx->rinfo->outinfo.writes_layer) |
           S_02881C_USE_VTX_VIEWPORT_INDX(ctx->rinfo->outinfo.writes_viewport_index) |
           S_02881C_USE_VTX_VRS_RATE(ctx->rinfo->outinfo.writes_primitive_shading_rate) |
           S_02881C_VS_OUT_MISC_VEC_ENA(misc_vec_ena) |
           S_02881C_VS_OUT_MISC_SIDE_BUS_ENA(
               misc_vec_ena || (ctx->gfx_level >= GFX10_3 && num_pos_exports > 1)) |
           S_02881C_VS_OUT_CCDIST0_VEC_ENA((total_mask & 0x0f) != 0) |
           S_02881C_VS_OUT_CCDIST1_VEC_ENA((total_mask & 0xf0) != 0) |
           total_mask << 8 | clip_dist_mask;
}

static uint32_t build_spi_shader_pos_format(unsigned num_pos_exports) {
    return S_02870C_POS0_EXPORT_FORMAT(V_02870C_SPI_SHADER_4COMP) |
           S_02870C_POS1_EXPORT_FORMAT(num_pos_exports > 1
                                          ? V_02870C_SPI_SHADER_4COMP
                                          : V_02870C_SPI_SHADER_NONE) |
           S_02870C_POS2_EXPORT_FORMAT(num_pos_exports > 2
                                          ? V_02870C_SPI_SHADER_4COMP
                                          : V_02870C_SPI_SHADER_NONE) |
           S_02870C_POS3_EXPORT_FORMAT(num_pos_exports > 3
                                          ? V_02870C_SPI_SHADER_4COMP
                                          : V_02870C_SPI_SHADER_NONE);
}

/* Record one statically used descriptor binding. Sets and binding numbers
 * outside the metadata ABI cannot be represented, so they are reported through
 * the caller-visible set mask instead (see gather_static_descriptor_use). */
static void record_descriptor_use(uint32_t set, uint32_t binding,
                                  uint32_t *set_mask,
                                  uint64_t binding_mask[PSBC_MAX_DESCRIPTOR_SETS])
{
    if (set >= PSBC_MAX_DESCRIPTOR_SETS)
        return;
    *set_mask |= 1u << set;
    if (binding < 64u)
        binding_mask[set] |= 1ull << binding;
}

/* Walk the optimized NIR that ACO consumes and record the descriptor bindings
 * the stage really dereferences. RADV's own info pass records the same sources
 * as a set mask; the standalone ABI also needs the binding identity so a
 * consumer can tell a required descriptor from a declared-but-unused one. */
static void gather_static_descriptor_use(
    const nir_shader *nir, uint32_t *set_mask,
    uint64_t binding_mask[PSBC_MAX_DESCRIPTOR_SETS])
{
    nir_foreach_function_impl(impl, nir) {
        nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
                if (instr->type == nir_instr_type_intrinsic) {
                    nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
                    switch (intrin->intrinsic) {
                    case nir_intrinsic_vulkan_resource_index:
                        record_descriptor_use(nir_intrinsic_desc_set(intrin),
                                              nir_intrinsic_binding(intrin),
                                              set_mask, binding_mask);
                        break;
                    case nir_intrinsic_image_deref_load:
                    case nir_intrinsic_image_deref_sparse_load:
                    case nir_intrinsic_image_deref_store:
                    case nir_intrinsic_image_deref_atomic:
                    case nir_intrinsic_image_deref_atomic_swap:
                    case nir_intrinsic_image_deref_size:
                    case nir_intrinsic_image_deref_samples: {
                        const nir_variable *var = nir_deref_instr_get_variable(
                            nir_def_as_deref(intrin->src[0].ssa));
                        if (var)
                            record_descriptor_use(var->data.descriptor_set,
                                                  var->data.binding,
                                                  set_mask, binding_mask);
                        break;
                    }
                    default:
                        break;
                    }
                } else if (instr->type == nir_instr_type_tex) {
                    nir_tex_instr *tex = nir_instr_as_tex(instr);
                    for (unsigned i = 0; i < tex->num_srcs; ++i) {
                        if (tex->src[i].src_type != nir_tex_src_texture_deref &&
                            tex->src[i].src_type != nir_tex_src_sampler_deref)
                            continue;
                        const nir_variable *var = nir_deref_instr_get_variable(
                            nir_src_as_deref(tex->src[i].src));
                        if (var)
                            record_descriptor_use(var->data.descriptor_set,
                                                  var->data.binding,
                                                  set_mask, binding_mask);
                    }
                }
            }
        }
    }
}

/* VGT_TF_PARAM for a tessellation stage, derived from the interface that stage
 * declares: the primitive mode (domain), the spacing/partitioning and the
 * output topology.  The legacy GNM packaging and the PS5 metadata both use this
 * one derivation so the two can never disagree. */
static uint32_t tess_tf_param(const nir_shader* nir) {
    uint32_t tf_type;
    switch (nir->info.tess._primitive_mode) {
    case TESS_PRIMITIVE_ISOLINES:   tf_type = V_028B6C_TESS_ISOLINE;  break;
    case TESS_PRIMITIVE_TRIANGLES:  tf_type = V_028B6C_TESS_TRIANGLE; break;
    case TESS_PRIMITIVE_QUADS:      tf_type = V_028B6C_TESS_QUAD;     break;
    default:                        tf_type = V_028B6C_TESS_TRIANGLE; break;
    }

    uint32_t tf_partition;
    switch (nir->info.tess.spacing) {
    case TESS_SPACING_EQUAL:           tf_partition = V_028B6C_PART_INTEGER;    break;
    case TESS_SPACING_FRACTIONAL_ODD:  tf_partition = V_028B6C_PART_FRAC_ODD;   break;
    case TESS_SPACING_FRACTIONAL_EVEN: tf_partition = V_028B6C_PART_FRAC_EVEN;  break;
    default:                           tf_partition = V_028B6C_PART_INTEGER;    break;
    }

    uint32_t tf_topology;
    if (nir->info.tess.point_mode) {
        tf_topology = V_028B6C_OUTPUT_POINT;
    } else if (nir->info.tess.ccw) {
        tf_topology = V_028B6C_OUTPUT_TRIANGLE_CCW;
    } else {
        tf_topology = V_028B6C_OUTPUT_TRIANGLE_CW;
    }

    return S_028B6C_TYPE(tf_type) |
           S_028B6C_PARTITIONING(tf_partition) |
           S_028B6C_TOPOLOGY(tf_topology);
}

static void fill_shader_metadata(const BuildContext* ctx,
                                 PsbcShaderMetadata* metadata) {
    memset(metadata, 0, sizeof(*metadata));
    metadata->version = PSBC_SHADER_METADATA_VERSION;
    metadata->target = ctx->target;
    metadata->source_stage = ctx->psbc_stage;
    metadata->unresolved_fields = PSBC_UNRESOLVED_PROGRAM_CHECKSUM;
    metadata->clip_distance_mask = ctx->rinfo->outinfo.clip_dist_mask;
    metadata->cull_distance_mask = ctx->rinfo->outinfo.cull_dist_mask;
    metadata->address32_hi = ctx->address32_hi;
    metadata->user_sgpr_count = ctx->rargs->num_user_sgprs;
    if (ctx->rinfo->loads_push_constants &&
        ctx->rargs->ac.push_constants.used) {
        metadata->push_constants_valid = true;
        metadata->push_constants_user_data_dword =
            ctx->rargs->user_sgprs_locs
                .shader_data[AC_UD_PUSH_CONSTANTS].sgpr_idx;
        metadata->push_constant_size = ctx->rinfo->push_constant_size;
    }
    if (ctx->ngg && ctx->rargs->ngg_lds_layout.used) {
        metadata->ngg_lds_layout_valid = true;
        metadata->ngg_lds_layout_user_data_dword =
            ctx->rargs->user_sgprs_locs.shader_data[AC_UD_NGG_LDS_LAYOUT].sgpr_idx;
        metadata->ngg_lds_layout = ctx->rinfo->ngg_info.esgs_ring_size;
    }
    if (ctx->config->scratch_bytes_per_wave) {
        metadata->scratch_valid = true;
        metadata->scratch_bytes_per_wave =
            ctx->config->scratch_bytes_per_wave;
        metadata->scratch_size_per_thread = ctx->nir->scratch_size;
        metadata->scratch_buffer_table_user_data_dword =
            ctx->rargs->user_sgprs_locs
                .shader_data[AC_UD_SCRATCH_RING_OFFSETS].sgpr_idx;
    }
    for (uint32_t set = 0; set < PSBC_MAX_DESCRIPTOR_SETS; ++set) {
        if (ctx->rargs->descriptors[set].used &&
            ctx->rargs->user_sgprs_locs.descriptor_sets[set].sgpr_idx !=
                UINT32_MAX) {
            metadata->descriptor_set_valid[set] = true;
            metadata->descriptor_set_user_data_dword[set] =
                ctx->rargs->user_sgprs_locs.descriptor_sets[set].sgpr_idx;
        }
    }
    metadata->descriptor_set0_valid = metadata->descriptor_set_valid[0];
    metadata->descriptor_set0_user_data_dword =
        metadata->descriptor_set_user_data_dword[0];
    if ((ctx->stage == MESA_SHADER_VERTEX ||
         (ctx->stage == MESA_SHADER_GEOMETRY && ctx->ngg)) &&
        ctx->rargs->ac.vertex_buffers.used) {
        metadata->vertex_buffer_table_valid = true;
        metadata->vertex_buffer_usage_mask = ctx->rinfo->vs.vb_desc_usage_mask;
        metadata->vertex_buffer_per_attribute = ctx->rinfo->vs.use_per_attribute_vb_descs;
        metadata->vertex_buffer_table_user_data_dword =
            ctx->rargs->user_sgprs_locs
                .shader_data[AC_UD_VS_VERTEX_BUFFERS].sgpr_idx;
    }
    if ((ctx->stage == MESA_SHADER_VERTEX ||
         (ctx->stage == MESA_SHADER_GEOMETRY && ctx->ngg)) &&
        ctx->rargs->ac.base_vertex.used) {
        metadata->base_vertex_valid = true;
        metadata->base_vertex_user_data_dword =
            ctx->rargs->user_sgprs_locs
                .shader_data[AC_UD_VS_BASE_VERTEX_START_INSTANCE].sgpr_idx;
    }
    if ((ctx->stage == MESA_SHADER_VERTEX ||
         (ctx->stage == MESA_SHADER_GEOMETRY && ctx->ngg)) &&
        ctx->rargs->ac.base_vertex.used && ctx->rargs->ac.draw_id.used) {
        /* DrawIndex lands in the same user-data block as the base vertex, so
         * the slot is that arg's dword offset relative to base_vertex. The
         * base_vertex argument is required for the arithmetic to mean
         * anything: if it were absent the pair is left invalid rather than
         * reported at a guessed offset. */
        const struct ac_shader_args* args = &ctx->rargs->ac;
        metadata->draw_id_valid = true;
        metadata->draw_id_user_data_dword =
            metadata->base_vertex_user_data_dword +
            args->args[args->draw_id.arg_index].offset -
            args->args[args->base_vertex.arg_index].offset;
    }
    if ((ctx->stage == MESA_SHADER_VERTEX ||
         (ctx->stage == MESA_SHADER_GEOMETRY && ctx->ngg)) &&
        ctx->rargs->ac.start_instance.used) {
        const struct ac_shader_args* args = &ctx->rargs->ac;
        metadata->start_instance_valid = true;
        metadata->start_instance_user_data_dword =
            metadata->base_vertex_user_data_dword +
            args->args[args->start_instance.arg_index].offset -
            args->args[args->base_vertex.arg_index].offset;
    }
    if ((ctx->stage == MESA_SHADER_VERTEX || ctx->stage == MESA_SHADER_FRAGMENT) &&
        ctx->rargs->ac.view_index.used) {
        /* ViewIndex is declared as its own user-data location (AC_UD_VIEW_INDEX,
         * added by RADV's vertex argument setup when the stage reads
         * SYSTEM_VALUE_VIEW_INDEX), so its slot is that location's SGPR index in
         * the block - not an offset inside another argument's dwords, which is
         * what separates it from draw_id and start_instance. The stage check
         * matches the lowering: vertex and PS5 fragment stages read the
         * argument. A stage that does not read it reports no slot. */
        metadata->view_index_valid = true;
        metadata->view_index_user_data_dword =
            ctx->rargs->user_sgprs_locs.shader_data[AC_UD_VIEW_INDEX].sgpr_idx;
    }
    if (ctx->rinfo->so.enabled_stream_buffers_mask &&
        ctx->rargs->streamout_buffers.used &&
        (ctx->ngg || (ctx->rargs->ac.streamout_config.used &&
                      ctx->rargs->ac.streamout_write_index.used))) {
        const struct ac_shader_args* args = &ctx->rargs->ac;
        metadata->streamout_valid = true;
        metadata->streamout_buffer_table_user_data_dword =
            ctx->rargs->user_sgprs_locs
                .shader_data[AC_UD_STREAMOUT_BUFFERS].sgpr_idx;
        metadata->streamout_enabled_stream_buffers_mask =
            ctx->rinfo->so.enabled_stream_buffers_mask;
        if (!ctx->ngg) {
            metadata->streamout_config_sgpr =
                args->args[args->streamout_config.arg_index].offset;
            metadata->streamout_write_index_sgpr =
                args->args[args->streamout_write_index.arg_index].offset;
        }
        for (unsigned i = 0; i < 4; ++i) {
            metadata->streamout_strides_dwords[i] =
                ctx->rinfo->so.strides[i];
            if (!ctx->ngg && args->streamout_offset[i].used)
                metadata->streamout_offset_sgprs[i] =
                    args->args[args->streamout_offset[i].arg_index].offset;
        }
    }
    metadata->descriptor_binding_count =
        ctx->options->descriptor_binding_count;
    memcpy(metadata->descriptor_bindings, ctx->options->descriptor_bindings,
           metadata->descriptor_binding_count *
               sizeof(metadata->descriptor_bindings[0]));
    /* The per-binding use mask backs the "required" side of the runtime
     * delivery contract. With static descriptor use it comes from the optimized
     * NIR; a set RADV reports as used whose binding the walk could not name
     * falls back to every declared binding of that set so an unknown entry can
     * never look unused. Without the option the whole declaration is reported,
     * which is the conservative legacy behaviour. */
    if (ctx->options->descriptor_binding_count &&
        ctx->options->static_descriptor_use) {
        memcpy(metadata->descriptor_used_binding_mask,
               ctx->descriptor_used_binding_mask,
               sizeof(metadata->descriptor_used_binding_mask));
    } else {
        for (uint32_t i = 0; i < metadata->descriptor_binding_count; ++i) {
            const PsbcDescriptorBinding* declared =
                &metadata->descriptor_bindings[i];
            if (declared->set >= PSBC_MAX_DESCRIPTOR_SETS ||
                declared->binding >= 64u)
                continue;
            metadata->descriptor_used_binding_mask[declared->set] |=
                1ull << declared->binding;
        }
    }

    PsbcRegisterWrite* cx = metadata->context_registers;
    PsbcRegisterWrite* sh = metadata->shader_registers;
    uint32_t* cx_count = &metadata->context_register_count;
    uint32_t* sh_count = &metadata->shader_register_count;

    if (ctx->stage == MESA_SHADER_FRAGMENT) {
        const unsigned per_primitive_inputs = ctx->target == PSBC_TARGET_PS5 &&
            ctx->options->primitive_id_per_primitive && ctx->rinfo->ps.prim_id_input;
        const bool param_gen = ctx->gfx_level >= GFX11 &&
                               !ctx->rinfo->ps.num_inputs &&
                               ctx->config->lds_size;
        metadata->hardware_stage = PSBC_HW_STAGE_PIXEL;
        if (!ctx->input_semantics || !ctx->input_semantics->valid)
            metadata->unresolved_fields |= PSBC_UNRESOLVED_AGC_LINKAGE;
        else {
            metadata->input_semantic_count = ctx->input_semantics->count;
            memcpy(metadata->input_semantics, ctx->input_semantics->words,
                   metadata->input_semantic_count *
                       sizeof(metadata->input_semantics[0]));
        }
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028710_SPI_SHADER_Z_FORMAT),
            ac_get_spi_shader_z_format(ctx->rinfo->ps.writes_z,
                ctx->rinfo->ps.writes_stencil,
                ctx->rinfo->ps.writes_sample_mask,
                ctx->rinfo->ps.writes_mrt0_alpha));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028714_SPI_SHADER_COL_FORMAT),
            compact_spi_shader_col_format(ctx->rinfo->ps.spi_shader_col_format));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_0286CC_SPI_PS_INPUT_ENA),
            ctx->config->spi_ps_input_ena);
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_0286D0_SPI_PS_INPUT_ADDR),
            ctx->config->spi_ps_input_addr);
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_0286D8_SPI_PS_IN_CONTROL),
            /* RADV normally leaves NUM_INTERP for its GFX10.3 pipeline linker
             * because per-primitive inputs use a separate count.  AGC's
             * standalone linker does not fill the per-vertex count for our
             * generated package, so carry the compiler-proven count here. */
            S_0286D8_NUM_INTERP(ctx->rinfo->ps.num_inputs - per_primitive_inputs) |
            S_0286D8_NUM_PRIM_INTERP(per_primitive_inputs) |
            S_0286D8_PS_W32_EN(ctx->rinfo->wave_size == 32) |
            S_0286D8_PARAM_GEN(param_gen));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_0286E0_SPI_BARYC_CNTL),
            S_0286E0_FRONT_FACE_ALL_BITS(0));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_02880C_DB_SHADER_CONTROL),
            compute_db_shader_control(ctx->rinfo));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_02823C_CB_SHADER_MASK),
            ac_get_cb_shader_mask(ctx->rinfo->ps.spi_shader_col_format));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028C40_PA_SC_SHADER_CONTROL), 0);

        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B020_SPI_SHADER_PGM_LO_PS), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B024_SPI_SHADER_PGM_HI_PS), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B028_SPI_SHADER_PGM_RSRC1_PS),
            ctx->config->rsrc1);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B02C_SPI_SHADER_PGM_RSRC2_PS),
            ctx->config->rsrc2);
        return;
    }

    if ((ctx->stage == MESA_SHADER_VERTEX ||
         ctx->stage == MESA_SHADER_GEOMETRY) && ctx->ngg) {
        const bool has_geometry = ctx->stage == MESA_SHADER_GEOMETRY;
        const uint32_t nparams = MAX2(ctx->rinfo->outinfo.param_exports, 1);
        const bool no_pc_export = ctx->rinfo->outinfo.param_exports == 0 &&
                                  ctx->rinfo->outinfo.prim_param_exports == 0;
        const unsigned num_prim_params = ctx->rinfo->outinfo.prim_param_exports;
        const unsigned num_pos_exports = get_num_pos_exports(ctx->rinfo, NULL, NULL);
        const uint32_t gs_num_invocations =
            ctx->rinfo->stage == MESA_SHADER_GEOMETRY
                ? ctx->rinfo->gs.invocations : 1;

        metadata->hardware_stage = PSBC_HW_STAGE_NGG;
        metadata->unresolved_fields |=
            PSBC_UNRESOLVED_NGG_ESGS_RING_ITEMSIZE;
        if (!fill_output_semantics(ctx->rinfo, metadata))
            metadata->unresolved_fields |= PSBC_UNRESOLVED_AGC_LINKAGE;

        metadata->linkage_valid = true;
        metadata->linkage_ge_cntl = (PsbcRegisterWrite) {
            .offset = PSBC_UC_OFFSET(R_03096C_GE_CNTL),
            .value = S_03096C_PRIM_GRP_SIZE_GFX10(
                         ctx->rinfo->ngg_info.max_gsprims) |
                     S_03096C_VERT_GRP_SIZE(
                         ctx->rinfo->ngg_info.hw_max_esverts),
        };
        metadata->linkage_stages_en = (PsbcRegisterWrite) {
            .offset = PSBC_CX_OFFSET(R_028B54_VGT_SHADER_STAGES_EN),
            /* Match RADV's GFX10 VGT shader-stage programming.  NGG runs
             * through the ES/GS hardware path even without an API geometry
             * shader.  A real GS must enable GS and cannot use NGG
             * passthrough; otherwise its emitted primitives are skipped. */
            .value = S_028B54_ES_EN(V_028B54_ES_STAGE_REAL) |
                     S_028B54_GS_EN(has_geometry) |
                     S_028B54_PRIMGEN_EN(1) |
                     S_028B54_MAX_PRIMGRP_IN_WAVE(2) |
                     S_028B54_GS_W32_EN(ctx->rinfo->wave_size == 32) |
                     S_028B54_VS_W32_EN(!has_geometry &&
                                        ctx->rinfo->wave_size == 32) |
                     S_028B54_NGG_WAVE_ID_EN(
                         ctx->rinfo->ngg_wave_id_en) |
                     S_028B54_PRIMGEN_PASSTHRU_EN(
                         ctx->rinfo->is_ngg_passthrough),
        };
        metadata->linkage_user_vgpr_en = (PsbcRegisterWrite) {
            .offset = PSBC_UC_OFFSET(R_030988_GE_USER_VGPR_EN),
            .value = 0,
        };
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_0286C4_SPI_VS_OUT_CONFIG),
            S_0286C4_VS_EXPORT_COUNT(nparams - 1) |
            S_0286C4_PRIM_EXPORT_COUNT(num_prim_params) |
            S_0286C4_NO_PC_EXPORT(no_pc_export));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028A84_VGT_PRIMITIVEID_EN),
            S_028A84_NGG_DISABLE_PROVOK_REUSE(
                ctx->stage == MESA_SHADER_VERTEX && ctx->rinfo->outinfo.export_prim_id));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_02870C_SPI_SHADER_POS_FORMAT),
            build_spi_shader_pos_format(num_pos_exports));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028708_SPI_SHADER_IDX_FORMAT),
            S_028708_IDX0_EXPORT_FORMAT(V_028708_SPI_SHADER_1COMP));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_02881C_PA_CL_VS_OUT_CNTL),
            build_pa_cl_vs_out_cntl(ctx, num_pos_exports));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028B4C_GE_NGG_SUBGRP_CNTL),
            S_028B4C_PRIM_AMP_FACTOR(ctx->rinfo->ngg_info.prim_amp_factor) |
            S_028B4C_THDS_PER_SUBGRP(0));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028B90_VGT_GS_INSTANCE_CNT),
            S_028B90_CNT(gs_num_invocations) |
            S_028B90_ENABLE(gs_num_invocations > 1) |
            S_028B90_EN_MAX_VERT_OUT_PER_GS_INSTANCE(
                ctx->rinfo->ngg_info.max_vert_out_per_gs_instance));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028A44_VGT_GS_ONCHIP_CNTL),
            S_028A44_ES_VERTS_PER_SUBGRP(ctx->rinfo->ngg_info.hw_max_esverts) |
            S_028A44_GS_PRIMS_PER_SUBGRP(ctx->rinfo->ngg_info.max_gsprims) |
            S_028A44_GS_INST_PRIMS_IN_SUBGRP(
                ctx->rinfo->ngg_info.max_gsprims * gs_num_invocations));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_0287FC_GE_MAX_OUTPUT_PER_SUBGROUP),
            S_0287FC_MAX_VERTS_PER_SUBGROUP(
                ctx->rinfo->ngg_info.max_out_verts));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028AAC_VGT_ESGS_RING_ITEMSIZE),
            ctx->rinfo->ngg_info.vgt_esgs_ring_itemsize);
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028B38_VGT_GS_MAX_VERT_OUT),
            ctx->rinfo->gs.vertices_out);
        if (has_geometry) {
            uint32_t output_primitive = V_028A6C_TRISTRIP;

            if (ctx->rinfo->gs.output_prim == MESA_PRIM_POINTS)
                output_primitive = V_028A6C_POINTLIST;
            else if (ctx->rinfo->gs.output_prim == MESA_PRIM_LINE_STRIP)
                output_primitive = V_028A6C_LINESTRIP;

            metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
                PSBC_CX_OFFSET(R_028A6C_VGT_GS_OUT_PRIM_TYPE),
                S_028A6C_OUTPRIM_TYPE(output_primitive));
        }

        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B320_SPI_SHADER_PGM_LO_ES), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B324_SPI_SHADER_PGM_HI_ES), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B228_SPI_SHADER_PGM_RSRC1_GS),
            ctx->config->rsrc1);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B22C_SPI_SHADER_PGM_RSRC2_GS),
            ctx->config->rsrc2);
        if (ctx->target == PSBC_TARGET_PS5) {
            /* GFX10 programs these for every NGG shader.  Keep late
             * allocation disabled until the console CU topology is known;
             * this is the conservative ac_compute_late_alloc fallback. */
            metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
                PSBC_SH_OFFSET(R_00B21C_SPI_SHADER_PGM_RSRC3_GS),
                S_00B21C_CU_EN(0xffff) | S_00B21C_WAVE_LIMIT(0x3f));
            metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
                PSBC_SH_OFFSET(R_00B204_SPI_SHADER_PGM_RSRC4_GS),
                S_00B204_CU_EN_GFX10(0xffff) |
                S_00B204_SPI_SHADER_LATE_ALLOC_GS_GFX10(0));
        }
        return;
    }

    if (ctx->stage == MESA_SHADER_GEOMETRY) {
        /* A standalone legacy GS proves NIR/ACO support, but it is not a
         * complete GFX9+ pipeline: the pre-raster stage must be merged and
         * the legacy path also needs a copy shader.  Do not mislabel this
         * diagnostic artifact as a directly loadable vertex package. */
        metadata->hardware_stage = PSBC_HW_STAGE_UNKNOWN;
        metadata->unresolved_fields |= PSBC_UNRESOLVED_AGC_LINKAGE;
        return;
    }

    if (ctx->stage == MESA_SHADER_VERTEX) {
        const uint32_t nparams = MAX2(ctx->rinfo->outinfo.param_exports, 1);
        const unsigned num_pos_exports =
            get_num_pos_exports(ctx->rinfo, NULL, NULL);

        metadata->hardware_stage = PSBC_HW_STAGE_VERTEX;
        metadata->linkage_valid = true;
        metadata->linkage_ge_cntl = (PsbcRegisterWrite) {
            .offset = PSBC_UC_OFFSET(R_03096C_GE_CNTL),
            .value = S_03096C_PRIM_GRP_SIZE_GFX10(128) |
                     S_03096C_VERT_GRP_SIZE(256),
        };
        metadata->linkage_stages_en = (PsbcRegisterWrite) {
            .offset = PSBC_CX_OFFSET(R_028B54_VGT_SHADER_STAGES_EN),
            .value = S_028B54_MAX_PRIMGRP_IN_WAVE(2) |
                     S_028B54_VS_W32_EN(ctx->rinfo->wave_size == 32),
        };
        metadata->linkage_user_vgpr_en = (PsbcRegisterWrite) {
            .offset = PSBC_UC_OFFSET(R_030988_GE_USER_VGPR_EN),
            .value = 0,
        };

        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_0286C4_SPI_VS_OUT_CONFIG),
            S_0286C4_VS_EXPORT_COUNT(nparams - 1) |
            S_0286C4_NO_PC_EXPORT(ctx->rinfo->outinfo.param_exports == 0));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_02870C_SPI_SHADER_POS_FORMAT),
            build_spi_shader_pos_format(num_pos_exports));
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_02881C_PA_CL_VS_OUT_CNTL),
            build_pa_cl_vs_out_cntl(ctx, num_pos_exports));

        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B120_SPI_SHADER_PGM_LO_VS), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B124_SPI_SHADER_PGM_HI_VS), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B128_SPI_SHADER_PGM_RSRC1_VS),
            ctx->config->rsrc1);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B12C_SPI_SHADER_PGM_RSRC2_VS),
            ctx->config->rsrc2);
        return;
    }

    if (ctx->stage == MESA_SHADER_TESS_CTRL) {
        /* The hull half. On GFX10 the hull stage is two program counters: the
         * LS program (the vertex half, R_00B520/R_00B528) and the HS program
         * (this tessellation-control half, R_00B420/R_00B428), whose
         * RSRC1/RSRC2 radv combines with radv_shader_combine_cfg_vs_tcs().
         * Publish this half's program and resource registers from the same
         * config the legacy GNM packaging already writes; everything else -
         * the LS half's program, the combined RSRC pair, and the hull state the
         * driver owns (VGT_SHADER_STAGES_EN LS_EN/HS_EN, VGT_LS_HS_CONFIG,
         * VGT_TF_RING_SIZE, VGT_HS_OFFCHIP_PARAM) - stays explicitly
         * unresolved, so a consumer still cannot mistake this for a loadable
         * hull package. */
        metadata->unresolved_fields |= PSBC_UNRESOLVED_TESS_PIPELINE;
        /* The hull/domain interface state this stage fully determines: the
         * domain, the partitioning and the output topology.  The rest of the
         * hull state (LS_HS_CONFIG, the TF ring, the offchip parameter and the
         * stage enables) needs the pipeline's patch control points and the
         * driver's draw state, so it stays out of the package on purpose. */
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028B6C_VGT_TF_PARAM), tess_tf_param(ctx->nir));
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B420_SPI_SHADER_PGM_LO_HS), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B424_SPI_SHADER_PGM_HI_HS), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B428_SPI_SHADER_PGM_RSRC1_HS),
            ctx->config->rsrc1);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B42C_SPI_SHADER_PGM_RSRC2_HS),
            ctx->config->rsrc2);
        return;
    }

    if (ctx->stage == MESA_SHADER_TESS_EVAL) {
        /* The domain half has no package state at all yet: radv runs it in the
         * ES/GS hardware path with the tessellation-specific configuration, and
         * nothing here describes that. Keep the explicit unresolved bit and
         * publish no registers rather than an incomplete set. */
        metadata->unresolved_fields |= PSBC_UNRESOLVED_TESS_PIPELINE;
        return;
    }
}

/* Build the shader binary into a memory buffer instead of a file. */
static PsbcResult buildshaderbinary(
    const BuildContext* ctx,
    const uint32_t* code, uint32_t code_dw,
    uint8_t** out_data, size_t* out_size
) {
    uint32_t codesize = code_dw * sizeof(uint32_t);
    if (ctx->nir->info.stage == MESA_SHADER_VERTEX &&
        ctx->rinfo->vs.has_prolog) {
        codesize += sizeof(uint32_t);
    }

    uint32_t* newcode = malloc(codesize);
    if (!newcode)
        return PSBC_RESULT_OUT_OF_MEMORY;
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

    /* Count input usage slots.
     * prolog_inputs (SUBPTR_FETCHSHADER) and vertex_buffers
     * (PTR_VERTEXBUFFERTABLE) are VS-specific — they belong to the VS
     * portion of a merged pipeline. For standalone HS/GS compilation,
     * these args are declared because previous_stage=VERTEX, but the
     * slots should not be emitted in the HS/GS shader binary.
     * The descriptor table (PTR_INDIRECTRESOURCETABLE) is shared by
     * all stages and should always be emitted when used. */
    const bool is_vs_family =
        ctx->stage == MESA_SHADER_VERTEX ||
        ctx->psbc_stage == PSBC_STAGE_EXPORT ||
        ctx->psbc_stage == PSBC_STAGE_LOCAL;

    uint32_t numinputslots = 0;
    if (is_vs_family && ctx->rargs->prolog_inputs.used) {
        numinputslots += 1;
    }
    if (is_vs_family && ctx->rargs->ac.vertex_buffers.used) {
        numinputslots += 1;
    }
    if (ctx->rargs->descriptors[0].used) {
        numinputslots += 1;
    }

    uint32_t shspecificsize = headershsize(ctx->psbc_stage);
    shspecificsize += numinputslots * sizeof(GnmInputUsageSlot);

    /* exclude position as it doesn't count as an export semantic in PSB */
    const uint32_t outputswritten =
        util_bitcount64(ctx->nir->info.outputs_written & ~VARYING_BIT_POS);

    switch (ctx->stage) {
    case MESA_SHADER_VERTEX:
    case MESA_SHADER_TESS_EVAL:  /* DS outputs vertices like VS */
        shspecificsize += util_bitcount64(ctx->nir->info.inputs_read) *
                          sizeof(GnmVertexInputSemantic);
        shspecificsize +=
            outputswritten * sizeof(GnmVertexExportSemantic);
        break;
    case MESA_SHADER_FRAGMENT:
        shspecificsize += ctx->input_semantics->count *
                          sizeof(GnmPixelInputSemantic);
        break;
    case MESA_SHADER_GEOMETRY:
        /* GS has input/export semantics like VS */
        shspecificsize += util_bitcount64(ctx->nir->info.inputs_read) *
                          sizeof(GnmVertexInputSemantic);
        shspecificsize +=
            outputswritten * sizeof(GnmVertexExportSemantic);
        break;
    case MESA_SHADER_TESS_CTRL:
        /* HS has input semantics (from VS outputs) but no export semantics */
        shspecificsize += util_bitcount64(ctx->nir->info.inputs_read) *
                          sizeof(GnmVertexInputSemantic);
        break;
    default:
        break;
    }

    /* GCN bytecode that comes after the headers must be 4 byte aligned */
    const uint32_t alignbytes =
        ((shspecificsize + 3) & (-4)) - shspecificsize;
    shspecificsize += alignbytes;

    /* Calculate total size */
    const size_t total_size = sizeof(PsslBinaryHeader) +
                              sizeof(GnmShaderFileHeader) +
                              shspecificsize +
                              codesize +
                              sizeof(GnmShaderBinaryInfo) +
                              sizeof(PsslBinaryParamInfo);

    uint8_t* buf = malloc(total_size);
    if (!buf) {
        free(newcode);
        return PSBC_RESULT_OUT_OF_MEMORY;
    }
    size_t offset = 0;

    /* PSSL binary header */
    const PsslBinaryHeader psbh = {
        .vermajor = 0,
        .verminor = 4,
        .shadertype = psbshtype(ctx->psbc_stage),
        .codetype = PSSL_CODE_ISA,
        .compilertype = PSSL_COMPILER_UNSPECIFIED,
        .codesize = sizeof(GnmShaderFileHeader) + shspecificsize +
                    codesize + sizeof(GnmShaderBinaryInfo),
    };
    memcpy(buf + offset, &psbh, sizeof(psbh));
    offset += sizeof(psbh);

    /* GNM shader file header */
    const GnmShaderFileHeader gsfh = {
        .magic = GNM_SHADER_FILE_HEADER_ID,
        .vermajor = 7,
        .verminor = 2,
        .type = gnmshtype(ctx->psbc_stage),
        .headersizedwords = shspecificsize / 4,
        .targetgpumodes =
            (ctx->gfx_level >= GFX10_3) ? GNM_TARGETGPUMODE_NEO :
            (ctx->neo ? GNM_TARGETGPUMODE_NEO : GNM_TARGETGPUMODE_BASE),
    };
    memcpy(buf + offset, &gsfh, sizeof(gsfh));
    offset += sizeof(gsfh);

    /* Shader-specific header (VS/PS/GS/HS/ES/LS registers) */
    switch (ctx->psbc_stage) {
    case PSBC_STAGE_VERTEX:
    case PSBC_STAGE_TESS_EVAL: {
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
                ctx->input_semantics->count,
            .numexportsemantics = outputswritten,
        };
        memcpy(buf + offset, &vsh, sizeof(vsh));
        offset += sizeof(vsh);
        break;
    }
    case PSBC_STAGE_FRAGMENT: {
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
                        compact_spi_shader_col_format(ctx->rinfo->ps.spi_shader_col_format),
                    .spipsinputena = ctx->config->spi_ps_input_ena,
                    .spipsinputaddr = ctx->config->spi_ps_input_addr,
                    .spipsincontrol =
                        S_0286D8_NUM_INTERP(ctx->rinfo->ps.num_inputs) |
                        S_0286D8_PS_W32_EN(
                            ctx->rinfo->wave_size == 32
                        ) |
                        S_0286D8_PARAM_GEN(param_gen),
                    .spibaryccntl = S_0286E0_FRONT_FACE_ALL_BITS(0),
                    .dbshadercontrol = compute_db_shader_control(ctx->rinfo),
                    .cbshadermask = ac_get_cb_shader_mask(
                        ctx->rinfo->ps.spi_shader_col_format
                    ),
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
        };
        memcpy(buf + offset, &psh, sizeof(psh));
        offset += sizeof(psh);
        break;
    }
    case PSBC_STAGE_COMPUTE: {
        const GnmCsShader csh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .computepgmlo = shspecificsize,
                    .computepgmhi = 0,
                    .computepgmrsrc1 = ctx->config->rsrc1,
                    .computepgmrsrc2 = ctx->config->rsrc2,
                    /* Thread group size from NIR compute shader info */
                    .computenumthreadx =
                        ctx->nir->info.workgroup_size[0] ?
                        ctx->nir->info.workgroup_size[0] : 1,
                    .computenumthready =
                        ctx->nir->info.workgroup_size[1] ?
                        ctx->nir->info.workgroup_size[1] : 1,
                    .computenumthreadz =
                        ctx->nir->info.workgroup_size[2] ?
                        ctx->nir->info.workgroup_size[2] : 1,
                },
        };
        memcpy(buf + offset, &csh, sizeof(csh));
        offset += sizeof(csh);
        break;
    }
    case PSBC_STAGE_GEOMETRY: {
        /* Map mesa_prim to VGT_GS_OUT_PRIM_TYPE values */
        uint32_t gs_out_prim;
        switch (ctx->nir->info.gs.output_primitive) {
        case MESA_PRIM_POINTS:       gs_out_prim = V_028A6C_POINTLIST; break;
        case MESA_PRIM_LINE_STRIP:     gs_out_prim = V_028A6C_LINESTRIP; break;
        case MESA_PRIM_TRIANGLE_STRIP: gs_out_prim = V_028A6C_TRISTRIP;  break;
        default:                     gs_out_prim = V_028A6C_TRISTRIP;  break;
        }

        const GnmGsShader gsh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmlogs = shspecificsize,
                    .spishaderpgmhigs = 0xffffffff,
                    .spishaderpgmrsrc1gs = ctx->config->rsrc1,
                    .spishaderpgmrsrc2gs = ctx->config->rsrc2,
                    .vgtstrmoutconfig = 0,
                    .vgtgsoutprimtype = gs_out_prim,
                    .vgtgsinstancecnt =
                        S_028B90_CNT(MIN2(ctx->rinfo->gs.invocations, 127)) |
                        S_028B90_ENABLE(ctx->rinfo->gs.invocations > 0),
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
            .numexportsemantics = outputswritten,
        };
        memcpy(buf + offset, &gsh, sizeof(gsh));
        offset += sizeof(gsh);
        break;
    }
    case PSBC_STAGE_TESS_CTRL: {
        /* One derivation, shared with the PS5 metadata path. */
        const uint32_t tf_param = tess_tf_param(ctx->nir);

        const GnmHsShader hsh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmlohs = shspecificsize,
                    .spishaderpgmhihs = 0xffffffff,
                    .spishaderpgmrsrc1hs = ctx->config->rsrc1,
                    .spishaderpgmrsrc2hs = ctx->config->rsrc2,
                    .vgttfparam = tf_param,
                    /* Tessellation levels — use defaults if not available */
                    .vgthosmaxtesslevel = 64,
                    .vgthosmintesslevel = 1,
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
        };
        memcpy(buf + offset, &hsh, sizeof(hsh));
        offset += sizeof(hsh);
        break;
    }
    case PSBC_STAGE_EXPORT: {
        /* ES (Export Shader) — VS variant with ES registers.
         * Runs before GS in a geometry pipeline. */
        const GnmEsShader esh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmloes = shspecificsize,
                    .spishaderpgmhies = 0xffffffff,
                    .spishaderpgmrsrc1es = ctx->config->rsrc1,
                    .spishaderpgmrsrc2es = ctx->config->rsrc2,
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
            .numexportsemantics = outputswritten,
        };
        memcpy(buf + offset, &esh, sizeof(esh));
        offset += sizeof(esh);
        break;
    }
    case PSBC_STAGE_LOCAL: {
        /* LS (Local Shader) — VS variant with LS registers.
         * Runs before HS in a tessellation pipeline. */
        const GnmLsShader lsh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmlols = shspecificsize,
                    .spishaderpgmhils = 0xffffffff,
                    .spishaderpgmrsrc1ls = ctx->config->rsrc1,
                    .spishaderpgmrsrc2ls = ctx->config->rsrc2,
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
            .numexportsemantics = outputswritten,
        };
        memcpy(buf + offset, &lsh, sizeof(lsh));
        offset += sizeof(lsh);
        break;
    }
    default:
        free(newcode);
        free(buf);
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    }

    /* write shader common data: input usage slots */
    if (is_vs_family && ctx->rargs->prolog_inputs.used) {
        const GnmInputUsageSlot s = {
            .usagetype = GNM_SHINPUTUSAGE_SUBPTR_FETCHSHADER,
            .startregister =
                ctx->rargs->ac.args[ctx->rargs->prolog_inputs.arg_index]
                    .offset,
        };
        memcpy(buf + offset, &s, sizeof(s));
        offset += sizeof(s);
    }
    if (is_vs_family && ctx->rargs->ac.vertex_buffers.used) {
        const GnmInputUsageSlot s = {
            .usagetype = GNM_SHINPUTUSAGE_PTR_VERTEXBUFFERTABLE,
            .startregister =
                ctx->rargs->ac
                    .args[ctx->rargs->ac.vertex_buffers.arg_index]
                    .offset,
        };
        memcpy(buf + offset, &s, sizeof(s));
        offset += sizeof(s);
    }
    if (ctx->rargs->descriptors[0].used) {
        const GnmInputUsageSlot s = {
            .usagetype = GNM_SHINPUTUSAGE_PTR_INDIRECTRESOURCETABLE,
            .startregister =
                ctx->rargs->ac
                    .args[ctx->rargs->descriptors[0].arg_index]
                    .offset,
        };
        memcpy(buf + offset, &s, sizeof(s));
        offset += sizeof(s);
    }

    /* write input/export semantics */
    switch (ctx->stage) {
    case MESA_SHADER_VERTEX:
    case MESA_SHADER_TESS_EVAL:  /* DS uses same semantics as VS */
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
            memcpy(buf + offset, &input, sizeof(input));
            offset += sizeof(input);
        }
        for (uint32_t i = 0; i < outputswritten; i += 1) {
            const GnmVertexExportSemantic out = {
                .semantic = 15 + i,
                .outindex = i,
                .exportf16 = 0,
            };
            memcpy(buf + offset, &out, sizeof(out));
            offset += sizeof(out);
        }
        break;
    case MESA_SHADER_FRAGMENT:
        for (uint32_t i = 0; i < ctx->input_semantics->count; i += 1) {
            const uint32_t word = ctx->input_semantics->words[i];
            const GnmPixelInputSemantic input = {
                .semantic = word & 0xffu,
                .isflatshaded = (word >> 22) & 1u,
            };
            memcpy(buf + offset, &input, sizeof(input));
            offset += sizeof(input);
        }
        break;
    case MESA_SHADER_GEOMETRY:
        /* GS input/export semantics use the same vertex semantic format */
        for (uint32_t i = 0;
             i < util_bitcount64(ctx->nir->info.inputs_read); i += 1) {
            const GnmVertexInputSemantic input = {
                .semantic = i,
            };
            memcpy(buf + offset, &input, sizeof(input));
            offset += sizeof(input);
        }
        for (uint32_t i = 0; i < outputswritten; i += 1) {
            const GnmVertexExportSemantic out = {
                .semantic = 15 + i,
                .outindex = i,
                .exportf16 = 0,
            };
            memcpy(buf + offset, &out, sizeof(out));
            offset += sizeof(out);
        }
        break;
    case MESA_SHADER_TESS_CTRL:
        /* HS input semantics describe which VS outputs are read.
         * No export semantics — HS outputs go to LDS for DS. */
        for (uint32_t i = 0;
             i < util_bitcount64(ctx->nir->info.inputs_read); i += 1) {
            const GnmVertexInputSemantic input = {
                .semantic = i,
            };
            memcpy(buf + offset, &input, sizeof(input));
            offset += sizeof(input);
        }
        break;
    default:
        break;
    }

    /* alignment padding */
    memset(buf + offset, 0, alignbytes);
    offset += alignbytes;

    /* shader code */
    memcpy(buf + offset, newcode, codesize);
    offset += codesize;

    /* shader binary info */
    /* Compute chunkusagebaseoffsetdwords: offset from OrbShdr back to the
     * input usage slot table, in dwords.
     * Input usage slots are at: PSSL_HEADER_SIZE + GNM_FILE_HEADER_SIZE + headershsize
     * OrbShdr is at: current offset (= PSSL_HEADER_SIZE + GNM_FILE_HEADER_SIZE + shspecificsize + codesize)
     * Distance = offset - (PSSL_HEADER_SIZE + GNM_FILE_HEADER_SIZE + headershsize)
     * But the comment says "starts at ((uint32_t*)&ShaderBinaryInfo) - chunkusagebaseoffsetdwords"
     * so chunkusagebaseoffsetdwords = distance / 4
     * Note: if numinputslots == 0, the offset is 0 (no table to point to). */
    const size_t inputslots_offset =
        sizeof(PsslBinaryHeader) + sizeof(GnmShaderFileHeader) +
        headershsize(ctx->psbc_stage);
    const size_t orbshdr_offset = offset;
    const uint32_t chunkusageoffset =
        numinputslots > 0
            ? (uint32_t)((orbshdr_offset - inputslots_offset) / 4)
            : 0;

    /* Compute shader hash from SPIR-V data.
     * The PS4 uses this for shader cache identification.
     * We use a simple hash of the SPIR-V binary. */
    uint64_t shaderhash = 0;
    if (ctx->spirv_data && ctx->spirv_size > 0) {
        /* FNV-1a hash of the SPIR-V bytes */
        const uint8_t* sp = (const uint8_t*)ctx->spirv_data;
        shaderhash = 0xcbf29ce484222325ULL;
        for (size_t i = 0; i < ctx->spirv_size; i += 1) {
            shaderhash ^= sp[i];
            shaderhash *= 0x100000001b3ULL;
        }
    }

    GnmShaderBinaryInfo bininfo = {
        .signature = GNM_SHADER_BINARY_INFO_MAGIC,
        .version = 7,
        .ispsslcg = 1,
        .type = shbintype(ctx->psbc_stage),
        .length = codesize,
        .chunkusagebaseoffsetdwords = chunkusageoffset,
        .numinputusageslots = numinputslots,
        .shaderhash0 = (uint32_t)(shaderhash & 0xffffffff),
        .shaderhash1 = (uint32_t)(shaderhash >> 32),
    };
    bininfo.crc32 = hashsb(newcode, codesize, &bininfo);
    memcpy(buf + offset, &bininfo, sizeof(bininfo));
    offset += sizeof(bininfo);

    /* param info */
    const PsslBinaryParamInfo paraminfo = {0};
    memcpy(buf + offset, &paraminfo, sizeof(paraminfo));
    offset += sizeof(paraminfo);

    free(newcode);

    *out_data = buf;
    *out_size = offset;
    return PSBC_RESULT_OK;
}

/* === GPU target setup === */

static void setup_ac_info(struct ac_compiler_info* ac_info, enum amd_gfx_level gfxlevel) {
    ac_info->gfx_level = gfxlevel;
    ac_info->lds_size_per_workgroup = gfxlevel >= GFX7 ? 64 * 1024 : 32 * 1024;

    if (gfxlevel >= GFX10_3) {
        ac_info->max_waves_per_simd = 16;
        ac_info->num_physical_sgprs_per_simd = 108 * 16;
        ac_info->num_physical_wave64_vgprs_per_simd = 512;
        ac_info->num_simd_per_compute_unit = 2;
        ac_info->min_sgpr_alloc = 108;
        ac_info->max_sgpr_alloc = 108;
        ac_info->sgpr_alloc_granularity = 108;
        ac_info->min_wave64_vgpr_alloc = 8;
        ac_info->max_vgpr_alloc = 256;
        ac_info->wave64_vgpr_alloc_granularity = 8;
        ac_info->has_packed_math_16bit = true;
        /* PS5/GFX1013 does not implement accelerated integer dot products. */
        ac_info->has_fma_mix = true;
    } else if (gfxlevel == GFX10) {
        ac_info->max_waves_per_simd = 20;
        ac_info->num_physical_sgprs_per_simd = 128 * 20;
        ac_info->num_physical_wave64_vgprs_per_simd = 256;
        ac_info->num_simd_per_compute_unit = 2;
        ac_info->min_sgpr_alloc = 1;
        ac_info->max_sgpr_alloc = 128;
        ac_info->sgpr_alloc_granularity = 1;
        ac_info->min_wave64_vgpr_alloc = 4;
        ac_info->max_vgpr_alloc = 256;
        ac_info->wave64_vgpr_alloc_granularity = 4;
        ac_info->has_packed_math_16bit = true;
    } else if (gfxlevel == GFX8) {
        ac_info->max_waves_per_simd = 10;
        ac_info->num_physical_sgprs_per_simd = 16 * 10;
        ac_info->num_physical_wave64_vgprs_per_simd = 128;
        ac_info->num_simd_per_compute_unit = 1;
        ac_info->min_sgpr_alloc = 1;
        ac_info->max_sgpr_alloc = 16;
        ac_info->sgpr_alloc_granularity = 1;
        ac_info->min_wave64_vgpr_alloc = 4;
        ac_info->max_vgpr_alloc = 128;
        ac_info->wave64_vgpr_alloc_granularity = 4;
    } else {
        /* GFX7 */
        ac_info->max_waves_per_simd = 10;
        ac_info->num_physical_sgprs_per_simd = 12 * 10;
        ac_info->num_physical_wave64_vgprs_per_simd = 64;
        ac_info->num_simd_per_compute_unit = 1;
        ac_info->min_sgpr_alloc = 1;
        ac_info->max_sgpr_alloc = 12;
        ac_info->sgpr_alloc_granularity = 1;
        ac_info->min_wave64_vgpr_alloc = 4;
        ac_info->max_vgpr_alloc = 64;
        ac_info->wave64_vgpr_alloc_granularity = 4;
    }
}

static void setup_target(PsbcTarget target,
                         enum amd_gfx_level* gfxlevel,
                         enum radeon_family* chipfamily,
                         bool* neo) {
    switch (target) {
    case PSBC_TARGET_PS5:
        *gfxlevel = GFX10_3;
        *chipfamily = CHIP_NAVI21;
        *neo = false;
        break;
    case PSBC_TARGET_PS4_NEO:
        *gfxlevel = GFX8;
        *chipfamily = CHIP_TONGA;
        *neo = true;
        break;
    case PSBC_TARGET_PS4_BASE:
    default:
        *gfxlevel = GFX7;
        *chipfamily = CHIP_KAVERI;
        *neo = false;
        break;
    }
}

static pthread_once_t g_ps5_nir_options_once = PTHREAD_ONCE_INIT;
static struct ac_compiler_info g_ps5_nir_ac_info;
static struct radv_compiler_info g_ps5_nir_compiler_info;

static void init_ps5_nir_options(void) {
    setup_ac_info(&g_ps5_nir_ac_info, GFX10_3);
    g_ps5_nir_compiler_info.ac = &g_ps5_nir_ac_info;
    g_ps5_nir_compiler_info.key.family = CHIP_NAVI21;
    g_ps5_nir_compiler_info.key.ge_wave_size = 64;
    g_ps5_nir_compiler_info.key.ps_wave_size = 32;
    g_ps5_nir_compiler_info.key.cs_wave_size = 32;
    g_ps5_nir_compiler_info.key.rt_wave_size = 64;
    radv_get_nir_options(&g_ps5_nir_compiler_info);
}

const struct nir_shader_compiler_options*
psbc_get_nir_options(PsbcStage stage) {
    const mesa_shader_stage mesa_stage = psbc_to_mesa_stage(stage);

    if (mesa_stage == MESA_SHADER_NONE ||
        mesa_stage >= MESA_VULKAN_SHADER_STAGES)
        return NULL;
    pthread_once(&g_ps5_nir_options_once, init_ps5_nir_options);
    return &g_ps5_nir_compiler_info.nir_options[mesa_stage];
}

/* === Main compilation function === */

static bool psbc_licm_speculatable(nir_instr* instr, nir_loop* loop,
                                   bool instr_block_dominates_exit) {
    (void)loop;
    (void)instr_block_dominates_exit;
    return nir_instr_can_speculate(instr);
}

static nir_shader* prepare_stage_nir(
    const struct radv_compiler_info* compiler_info,
    struct radv_shader_stage* stage,
    const uint32_t* spirv,
    size_t spirv_size,
    const nir_shader* input_nir,
    const PsbcCompileOptions* opts
) {
    stage->spirv.data = (const char*)spirv;
    stage->spirv.size = spirv_size;
    stage->entrypoint = opts->entrypoint ? opts->entrypoint : "main";
    stage->key.optimisations_disabled = !opts->optimise;

    VkSpecializationMapEntry
        spec_entries[PSBC_MAX_SPECIALIZATION_CONSTANTS] = {0};
    uint8_t spec_data[PSBC_MAX_SPECIALIZATION_CONSTANTS *
                      PSBC_MAX_SPECIALIZATION_BYTES] = {0};
    VkSpecializationInfo spec_info = {0};
    if (opts->specialization_constant_count) {
        size_t data_size = 0;
        for (uint32_t i = 0; i < opts->specialization_constant_count; ++i) {
            const PsbcSpecializationConstant* source =
                &opts->specialization_constants[i];
            spec_entries[i].constantID = source->constant_id;
            spec_entries[i].offset = (uint32_t)data_size;
            spec_entries[i].size = source->size;
            memcpy(spec_data + data_size, source->data, source->size);
            data_size += source->size;
        }
        spec_info.mapEntryCount = opts->specialization_constant_count;
        spec_info.pMapEntries = spec_entries;
        spec_info.dataSize = data_size;
        spec_info.pData = spec_data;
        stage->spec_info = &spec_info;
    }

    if (input_nir)
        stage->internal_nir = (nir_shader*)input_nir;

    /* Preserve the per-view value for both stages of PS5 replayed multiview.
     * The fragment argument is declared explicitly instead of using LayerID:
     * each replay already rebases its attachment to the selected array layer.
     * Other targets keep their existing fragment lowering. */
    const bool deliver_view_index = stage->stage == MESA_SHADER_VERTEX ||
        (opts->target == PSBC_TARGET_PS5 && stage->stage == MESA_SHADER_FRAGMENT);
    const struct radv_spirv_to_nir_options spirv_options = {
        .lower_view_index_to_zero = !deliver_view_index,
        .lower_view_index_to_device_index = false,
    };
    nir_shader* nir = radv_shader_spirv_to_nir(
        compiler_info, stage, &spirv_options, false
    );
    if (!nir)
        return NULL;

    if (input_nir) {
        if (opts->descriptor_binding_count)
            nir_shader_instructions_pass(nir, lower_gallium_ubo_index,
                                         nir_metadata_control_flow, NULL);

        /* radv_shader_spirv_to_nir() normalizes these only on its SPIR-V
         * path.  Gallium supplies internal NIR, while ACO accepts only the
         * normalized 2*pi operations. */
        NIR_PASS(_, nir, nir_normalize_sin_cos);
    }

    radv_optimize_nir(nir, !opts->optimise);

    bool indirect_derefs_lowered = false;
    NIR_PASS(indirect_derefs_lowered, nir, ac_nir_lower_indirect_derefs);
    NIR_PASS(_, nir, nir_lower_vars_to_ssa);
    if (indirect_derefs_lowered) {
        NIR_PASS(_, nir, nir_lower_undef_to_zero, NULL);
        const nir_opt_peephole_select_options flatten_indirect_arrays = {
            .limit = 10,
            .indirect_load_ok = true,
            .expensive_alu_ok = true,
            .discard_ok = true,
        };
        NIR_PASS(_, nir, nir_opt_peephole_select,
                 &flatten_indirect_arrays);
        if (opts->optimise)
            radv_optimize_nir(nir, false);
    }
    nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
    radv_nir_lower_io(nir);
    /* Standalone SPIR-V has not gone through graphics-pipeline IO linking.
     * Its variables can all have driver_location zero. Canonicalize fragment
     * input bases from semantic locations before RADV gathers input masks and
     * ACO assigns interpolation attributes; otherwise distinct varyings alias.
     * Keep VS input bases alone: vertex descriptor locations are externally
     * specified. Later PS5 mixed-interpolation alias splitting still applies. */
    if (nir->info.stage == MESA_SHADER_FRAGMENT)
        NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_in);
    stage->nir = nir;
    return nir;
}

static PsbcResult validate_spirv_capabilities(
    const uint32_t* spirv,
    size_t spirv_size,
    const PsbcCompileOptions* opts
) {
    if (!spirv || spirv_size < 5 * sizeof(uint32_t) ||
        spirv_size % sizeof(uint32_t) || spirv[0] != UINT32_C(0x07230203))
        return PSBC_RESULT_INVALID_SPIRV;

    const size_t word_count = spirv_size / sizeof(uint32_t);
    for (size_t offset = 5; offset < word_count;) {
        const uint32_t instruction = spirv[offset];
        const uint16_t words = instruction >> 16;
        const uint16_t opcode = instruction & UINT16_MAX;
        if (!words || words > word_count - offset)
            return PSBC_RESULT_INVALID_SPIRV;
        if (opcode == SpvOpCapability && words == 2) {
            const SpvCapability capability = (SpvCapability)spirv[offset + 1];
            bool enabled = true;
            switch (capability) {
            case SpvCapabilityInt8:
                enabled = opts->enable_int8;
                break;
            case SpvCapabilityInt16:
                enabled = opts->enable_int16;
                break;
            case SpvCapabilityStorageBuffer8BitAccess:
                enabled = opts->enable_storage_buffer_8bit_access;
                break;
            case SpvCapabilityUniformAndStorageBuffer8BitAccess:
                enabled = opts->enable_uniform_and_storage_buffer_8bit_access;
                break;
            case SpvCapabilityStorageBuffer16BitAccess:
                enabled = opts->enable_storage_buffer_16bit_access;
                break;
            case SpvCapabilityUniformAndStorageBuffer16BitAccess:
                enabled = opts->enable_uniform_and_storage_buffer_16bit_access;
                break;
            default:
                break;
            }
            if (!enabled)
                return PSBC_RESULT_UNSUPPORTED_CAPABILITY;
        }
        offset += words;
    }
    return PSBC_RESULT_OK;
}

static PsbcResult psbc_compile_impl(
    const uint32_t*       spirv,
    size_t                spirv_size,
    const nir_shader*     input_nir,
    const uint32_t*       previous_spirv,
    size_t                previous_spirv_size,
    const nir_shader*     previous_input_nir,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput*     out
) {
    if ((!spirv && !input_nir) || !opts || !out)
        return PSBC_RESULT_INTERNAL_ERROR;

    memset(out, 0, sizeof(*out));
    if ((opts->enable_uniform_and_storage_buffer_8bit_access &&
         !opts->enable_storage_buffer_8bit_access) ||
        (opts->enable_uniform_and_storage_buffer_16bit_access &&
         !opts->enable_storage_buffer_16bit_access))
        return PSBC_RESULT_INTERNAL_ERROR;

    /* Validate SPIR-V structure and opt-in capabilities before Mesa lowering.
     * Mesa's generic reader warns and continues when a capability is absent
     * from its mask; a reusable Vulkan compiler must fail closed instead. */
    if (!input_nir) {
        const PsbcResult validation =
            validate_spirv_capabilities(spirv, spirv_size, opts);
        if (validation != PSBC_RESULT_OK)
            return validation;
    }

    const bool paired_geometry = previous_spirv || previous_input_nir;
    mesa_shader_stage mesa_stage = psbc_to_mesa_stage(opts->stage);
    if (mesa_stage == MESA_SHADER_NONE)
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (paired_geometry &&
        (mesa_stage != MESA_SHADER_GEOMETRY || !opts->ngg ||
         opts->target != PSBC_TARGET_PS5))
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (opts->ps5_global_streamout && !paired_geometry)
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (opts->primitive_id_per_primitive &&
        (opts->target != PSBC_TARGET_PS5 || mesa_stage != MESA_SHADER_FRAGMENT))
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (previous_input_nir &&
        previous_input_nir->info.stage != MESA_SHADER_VERTEX)
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (previous_spirv) {
        const PsbcResult validation =
            validate_spirv_capabilities(previous_spirv, previous_spirv_size,
                                        opts);
        if (validation != PSBC_RESULT_OK)
            return validation;
    }
    if (opts->ngg &&
        (opts->target != PSBC_TARGET_PS5 ||
         (mesa_stage != MESA_SHADER_VERTEX && !paired_geometry)))
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (opts->descriptor_binding_count > PSBC_MAX_DESCRIPTOR_BINDINGS)
        return PSBC_RESULT_INTERNAL_ERROR;
    if (opts->specialization_constant_count >
        PSBC_MAX_SPECIALIZATION_CONSTANTS)
        return PSBC_RESULT_INTERNAL_ERROR;
    for (uint32_t i = 0; i < opts->specialization_constant_count; ++i) {
        const PsbcSpecializationConstant* spec =
            &opts->specialization_constants[i];
        if (!spec->size || spec->size > PSBC_MAX_SPECIALIZATION_BYTES)
            return PSBC_RESULT_INTERNAL_ERROR;
        for (uint32_t j = 0; j < i; ++j)
            if (opts->specialization_constants[j].constant_id ==
                spec->constant_id)
                return PSBC_RESULT_INTERNAL_ERROR;
    }
    if ((opts->color_is_int8 | opts->color_is_int10) & ~UINT32_C(0xff))
        return PSBC_RESULT_INTERNAL_ERROR;
    for (unsigned i = 0; i < 8; ++i)
        if (((opts->spi_shader_col_format >> (4 * i)) & 0xf) >
            V_028714_SPI_SHADER_32_ABGR)
            return PSBC_RESULT_INTERNAL_ERROR;
    for (uint32_t i = 0; i < opts->descriptor_binding_count; ++i) {
        const PsbcDescriptorBinding* binding = &opts->descriptor_bindings[i];
        const bool valid_type =
            binding->type == PSBC_DESCRIPTOR_UNIFORM_BUFFER ||
            binding->type == PSBC_DESCRIPTOR_UNIFORM_TEXEL_BUFFER ||
            binding->type == PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER ||
            binding->type == PSBC_DESCRIPTOR_STORAGE_BUFFER ||
            binding->type == PSBC_DESCRIPTOR_INPUT_ATTACHMENT;
        /* The driver lays every record out in whole DWORDs: sixteen bytes for
         * the buffer SRDs, forty-eight for a combined T#/S# pair and
         * thirty-two for the resource-only image record an input attachment
         * is read through, which never carries the two sampler DWORDs. A
         * record declared with any other stride - a combined T#/S# presented
         * as an input attachment in particular - is refused instead of being
         * reinterpreted, so the caller can never hand the GPU a slot whose
         * width disagrees with its type. */
        const uint32_t expected_stride =
            binding->type == PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER ? 48u
            : binding->type == PSBC_DESCRIPTOR_INPUT_ATTACHMENT ? 32u : 16u;
        if (binding->set >= PSBC_MAX_DESCRIPTOR_SETS ||
            binding->binding >= PSBC_MAX_DESCRIPTOR_BINDINGS ||
            !valid_type || !binding->array_size ||
            binding->stride != expected_stride ||
            (binding->offset & 15u) ||
            (uint64_t)binding->offset +
                    (uint64_t)binding->array_size * binding->stride >
                UINT32_MAX)
            return PSBC_RESULT_INTERNAL_ERROR;
        for (uint32_t j = 0; j < i; ++j)
            if (opts->descriptor_bindings[j].set == binding->set &&
                opts->descriptor_bindings[j].binding == binding->binding)
                return PSBC_RESULT_INTERNAL_ERROR;
    }

    /* Ensure library is initialized */
    psbc_init();

    /* Select GFX level and family based on target */
    enum amd_gfx_level gfxlevel;
    enum radeon_family chipfamily;
    bool neo = false;
    setup_target(opts->target, &gfxlevel, &chipfamily, &neo);

    /* Construct ac_compiler_info for the target GPU */
    struct ac_compiler_info ac_info = {0};
    setup_ac_info(&ac_info, gfxlevel);
    if (opts->force_accelerated_dot)
        ac_info.has_accelerated_dot_product = true;

    /* Construct radv_compiler_info */
    struct radv_compiler_info compiler_info = {0};
    compiler_info.ac = &ac_info;
    compiler_info.spirv_caps.Shader = true;
    compiler_info.spirv_caps.MultiView = opts->target == PSBC_TARGET_PS5 &&
        (mesa_stage == MESA_SHADER_VERTEX || mesa_stage == MESA_SHADER_FRAGMENT);
    compiler_info.spirv_caps.Geometry = true;
    compiler_info.spirv_caps.TransformFeedback = true;
    compiler_info.spirv_caps.DotProduct = true;
    compiler_info.spirv_caps.DotProductInput4x8BitPacked = true;
    compiler_info.spirv_caps.Int8 = opts->enable_int8;
    compiler_info.spirv_caps.Int16 = opts->enable_int16;
    compiler_info.spirv_caps.StorageBuffer8BitAccess =
        opts->enable_storage_buffer_8bit_access;
    compiler_info.spirv_caps.UniformAndStorageBuffer8BitAccess =
        opts->enable_uniform_and_storage_buffer_8bit_access;
    compiler_info.spirv_caps.StorageUniformBufferBlock16 =
        opts->enable_storage_buffer_16bit_access;
    compiler_info.spirv_caps.StorageUniform16 =
        opts->enable_uniform_and_storage_buffer_16bit_access;
    if (opts->force_accelerated_dot) {
        compiler_info.spirv_caps.Int16 = true;
        compiler_info.spirv_caps.DotProductInputAll = true;
    }
    compiler_info.hw.address32_hi = opts->address32_hi;
    compiler_info.sampled_image_desc_size = 32;
    compiler_info.combined_image_sampler_desc_size = 48;
    compiler_info.combined_image_sampler_offset = 32;
    compiler_info.sampler_descriptor_size = 16;
    compiler_info.sampler_descriptor_alignment = 16;
    compiler_info.image_descriptor_size = 32;
    compiler_info.image_descriptor_alignment = 16;
    compiler_info.buffer_descriptor_size = 16;
    compiler_info.buffer_descriptor_alignment = 16;
    compiler_info.key.ge_wave_size = 64;
    compiler_info.key.ps_wave_size = (gfxlevel >= GFX10_3) ? 32 : 64;
    compiler_info.key.cs_wave_size = (gfxlevel >= GFX10_3) ? 32 : 64;
    compiler_info.key.rt_wave_size = 64;
    compiler_info.key.family = chipfamily;
    compiler_info.key.load_grid_size_from_user_sgpr = (gfxlevel >= GFX10_3);
    compiler_info.key.use_ngg = opts->ngg;
    compiler_info.key.ps5_global_streamout = opts->ps5_global_streamout;
    compiler_info.key.ps5_fragment_view_index = opts->target == PSBC_TARGET_PS5;
    /* ACO uses debug.family for disassembly and init_program assertion */
    compiler_info.debug.family = chipfamily;

    /* Initialize NIR options for all stages */
    radv_get_nir_options(&compiler_info);

    /* Construct radv_shader_stage */
    struct radv_shader_stage stage = {0};
    stage.stage = mesa_stage;
    stage.key.keep_executable_info = getenv("PSBC_DEBUG_DISASM") != NULL;
    /* Set next_stage based on the pipeline graph.
     * For standalone compilation we assume the simplest pipeline:
     *   VS → FS, VS → HS → DS → FS, VS → GS → FS
     */
    switch (mesa_stage) {
    case MESA_SHADER_VERTEX:    stage.next_stage = MESA_SHADER_FRAGMENT;     break;
    case MESA_SHADER_TESS_CTRL: stage.next_stage = MESA_SHADER_TESS_EVAL;    break;
    case MESA_SHADER_TESS_EVAL: stage.next_stage = MESA_SHADER_FRAGMENT;     break;
    case MESA_SHADER_GEOMETRY:  stage.next_stage = MESA_SHADER_FRAGMENT;     break;
    default:                    stage.next_stage = MESA_SHADER_NONE;         break;
    }
    if (input_nir && input_nir->info.stage != mesa_stage) {
        psbc_shutdown();
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    }
    debug_stage(input_nir ? "import-nir-begin" : "import-spirv-begin");
    nir_shader* nir = prepare_stage_nir(
        &compiler_info, &stage, spirv, spirv_size, input_nir, opts
    );
    if (!nir) {
        psbc_shutdown();
        return PSBC_RESULT_COMPILE_NIR;
    }
    debug_stage(input_nir ? "import-nir-end" : "import-spirv-end");
    debug_shader_io(input_nir ? "input-NIR" : "SPIR-V", nir, NULL);
    debug_shader_io("lowered-io", nir, NULL);

    const bool dual_source_blend =
        mesa_stage == MESA_SHADER_FRAGMENT &&
        (nir->info.outputs_written &
         BITFIELD64_BIT(FRAG_RESULT_DUAL_SRC_BLEND));
    struct radv_shader_stage previous = {0};
    nir_shader* previous_nir = NULL;
    if (paired_geometry) {
        previous.stage = MESA_SHADER_VERTEX;
        previous.next_stage = MESA_SHADER_GEOMETRY;
        previous_nir = prepare_stage_nir(
            &compiler_info, &previous, previous_spirv, previous_spirv_size,
            previous_input_nir, opts
        );
        if (!previous_nir) {
            ralloc_free(nir);
            psbc_shutdown();
            return PSBC_RESULT_COMPILE_NIR;
        }
        debug_shader_io(previous_input_nir ? "previous-NIR" :
                                               "previous-SPIR-V",
                        previous_nir, NULL);
    }

    /* Shader info + args + postprocess */
    stage.nir = nir;
    radv_nir_shader_info_init(stage.stage, stage.next_stage, &stage.info);
    stage.info.is_ngg = opts->ngg;
    if (paired_geometry) {
        radv_nir_shader_info_init(previous.stage, previous.next_stage,
                                  &previous.info);
        previous.info.is_ngg = true;
        NIR_PASS(_, nir, nir_lower_gs_intrinsics,
                 nir_lower_gs_intrinsics_per_stream |
                 nir_lower_gs_intrinsics_count_primitives |
                 nir_lower_gs_intrinsics_count_vertices_per_primitive |
                 nir_lower_gs_intrinsics_overwrite_incomplete);
        NIR_PASS(_, nir, nir_lower_vars_to_ssa);

        /* Both shaders are passed to ACO as one merged program.  Mark their
         * existing shared driver-location interface accordingly; otherwise
         * RADV selects the ABI for independently compiled merged halves. */
        const unsigned linked_slots = util_bitcount64(nir->info.inputs_read);
        previous.info.vs.num_linked_outputs = linked_slots;
        previous.info.outputs_linked = true;
        stage.info.gs.num_linked_inputs = linked_slots;
        stage.info.inputs_linked = true;
    }

    struct radv_shader_layout layout = {0};
    _Alignas(struct radv_descriptor_set_layout)
        uint8_t descriptor_set_storage[PSBC_MAX_DESCRIPTOR_SETS][
            sizeof(struct radv_descriptor_set_layout) +
            PSBC_MAX_DESCRIPTOR_BINDINGS *
                sizeof(struct radv_descriptor_set_binding_layout)] = {0};
    if (opts->descriptor_binding_count) {
        uint32_t set_count = 0;
        for (uint32_t i = 0; i < opts->descriptor_binding_count; ++i) {
            const PsbcDescriptorBinding* source =
                &opts->descriptor_bindings[i];
            struct radv_descriptor_set_layout* set_layout =
                (struct radv_descriptor_set_layout*)descriptor_set_storage[source->set];
            struct radv_descriptor_set_binding_layout* target =
                &set_layout->binding[source->binding];
            target->type = source->type == PSBC_DESCRIPTOR_UNIFORM_BUFFER
                               ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                           : source->type == PSBC_DESCRIPTOR_UNIFORM_TEXEL_BUFFER
                               ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                           : source->type == PSBC_DESCRIPTOR_STORAGE_BUFFER
                               ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                           : source->type == PSBC_DESCRIPTOR_INPUT_ATTACHMENT
                               ? VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT
                               : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            target->array_size = source->array_size;
            target->offset = source->offset;
            target->size = source->stride;
            set_layout->binding_count =
                MAX2(set_layout->binding_count, source->binding + 1u);
            set_layout->size = MAX2(set_layout->size, source->offset +
                source->array_size * source->stride);
            set_count = MAX2(set_count, source->set + 1u);
        }
        layout.num_sets = set_count;
        for (uint32_t set = 0; set < set_count; ++set)
            layout.set[set].layout =
                (struct radv_descriptor_set_layout*)descriptor_set_storage[set];
    }
    stage.layout = layout;
    if (paired_geometry)
        previous.layout = layout;
    struct radv_graphics_state_key gfx_state = {0};
    gfx_state.rs.provoking_vtx_last = opts->provoking_vtx_last;
    if (opts->rasterization_samples != 0 &&
        opts->rasterization_samples != 1 &&
        opts->rasterization_samples != 2 &&
        opts->rasterization_samples != 4 &&
        opts->rasterization_samples != 8) {
        if (previous_nir)
            ralloc_free(previous_nir);
        ralloc_free(nir);
        psbc_shutdown();
        return PSBC_RESULT_INTERNAL_ERROR;
    }
    gfx_state.ms.rasterization_samples = opts->rasterization_samples;
    switch (opts->primitive_type) {
    case 0:
        break;
    case 1:
        gfx_state.ia.topology = V_008958_DI_PT_POINTLIST;
        break;
    case 2:
        gfx_state.ia.topology = V_008958_DI_PT_LINELIST;
        break;
    case 3:
        gfx_state.ia.topology = V_008958_DI_PT_LINESTRIP;
        break;
    case 4:
        gfx_state.ia.topology = V_008958_DI_PT_TRILIST;
        break;
    case 5:
        gfx_state.ia.topology = V_008958_DI_PT_TRIFAN;
        break;
    case 6:
        gfx_state.ia.topology = V_008958_DI_PT_TRISTRIP;
        break;
    case 10:
        gfx_state.ia.topology = V_008958_DI_PT_LINELIST_ADJ;
        break;
    case 11:
        gfx_state.ia.topology = V_008958_DI_PT_LINESTRIP_ADJ;
        break;
    case 12:
        gfx_state.ia.topology = V_008958_DI_PT_TRILIST_ADJ;
        break;
    case 13:
        gfx_state.ia.topology = V_008958_DI_PT_TRISTRIP_ADJ;
        break;
    default:
        if (previous_nir)
            ralloc_free(previous_nir);
        ralloc_free(nir);
        psbc_shutdown();
        return PSBC_RESULT_INTERNAL_ERROR;
    }
    /* The frontend keys these exports by framebuffer format. Keep the legacy
     * defaults for standalone callers that do not provide framebuffer state. */
    gfx_state.ps.epilog.spi_shader_col_format = opts->spi_shader_col_format
        ? opts->spi_shader_col_format : UINT32_C(0x99999999);
    gfx_state.ps.epilog.color_is_int8 = opts->spi_shader_col_format
        ? opts->color_is_int8 : 0xff;
    gfx_state.ps.epilog.color_is_int10 = opts->spi_shader_col_format
        ? opts->color_is_int10 : 0;
    gfx_state.ps.has_epilog = false;
    if (dual_source_blend) {
        gfx_state.ps.epilog.mrt0_is_dual_src = true;
        /* Both sources feed MRT0, including its precision and clamp rules. */
        unsigned format = opts->spi_shader_col_format
            ? opts->spi_shader_col_format & 0xf : V_028714_SPI_SHADER_FP16_ABGR;
        gfx_state.ps.epilog.spi_shader_col_format = format * 0x11;
        gfx_state.ps.epilog.color_is_int8 = opts->spi_shader_col_format
            ? (opts->color_is_int8 & 1) * 3 : 0;
        gfx_state.ps.epilog.color_is_int10 = opts->spi_shader_col_format
            ? (opts->color_is_int10 & 1) * 3 : 0;
    }
    for (uint32_t i = 0; i < opts->vertex_attribute_count; ++i) {
        const PsbcVertexAttribute* attribute = &opts->vertex_attributes[i];
        enum pipe_format format = PIPE_FORMAT_NONE;
        switch (attribute->format) {
        case PSBC_VERTEX_FORMAT_R32_FLOAT:
            format = PIPE_FORMAT_R32_FLOAT;
            break;
        case PSBC_VERTEX_FORMAT_R32G32_FLOAT:
            format = PIPE_FORMAT_R32G32_FLOAT;
            break;
        case PSBC_VERTEX_FORMAT_R32G32B32_FLOAT:
            format = PIPE_FORMAT_R32G32B32_FLOAT;
            break;
        case PSBC_VERTEX_FORMAT_R32G32B32A32_FLOAT:
            format = PIPE_FORMAT_R32G32B32A32_FLOAT;
            break;
        case PSBC_VERTEX_FORMAT_B8G8R8A8_UNORM:
            format = PIPE_FORMAT_B8G8R8A8_UNORM;
            break;
        case PSBC_VERTEX_FORMAT_R8G8B8A8_UNORM:
            format = PIPE_FORMAT_R8G8B8A8_UNORM;
            break;
        case PSBC_VERTEX_FORMAT_R10G10B10A2_UNORM:
            format = PIPE_FORMAT_R10G10B10A2_UNORM;
            break;
        case PSBC_VERTEX_FORMAT_B10G10R10A2_UNORM:
            format = PIPE_FORMAT_B10G10R10A2_UNORM;
            break;
        case PSBC_VERTEX_FORMAT_R10G10B10A2_SNORM:
            format = PIPE_FORMAT_R10G10B10A2_SNORM;
            break;
        case PSBC_VERTEX_FORMAT_B10G10R10A2_SNORM:
            format = PIPE_FORMAT_B10G10R10A2_SNORM;
            break;
        case PSBC_VERTEX_FORMAT_R10G10B10A2_USCALED:
            format = PIPE_FORMAT_R10G10B10A2_USCALED;
            break;
        case PSBC_VERTEX_FORMAT_B10G10R10A2_USCALED:
            format = PIPE_FORMAT_B10G10R10A2_USCALED;
            break;
        case PSBC_VERTEX_FORMAT_R10G10B10A2_SSCALED:
            format = PIPE_FORMAT_R10G10B10A2_SSCALED;
            break;
        case PSBC_VERTEX_FORMAT_B10G10R10A2_SSCALED:
            format = PIPE_FORMAT_B10G10R10A2_SSCALED;
            break;
        case PSBC_VERTEX_FORMAT_R32_SINT:
            format = PIPE_FORMAT_R32_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R32G32_SINT:
            format = PIPE_FORMAT_R32G32_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R32G32B32_SINT:
            format = PIPE_FORMAT_R32G32B32_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R32G32B32A32_SINT:
            format = PIPE_FORMAT_R32G32B32A32_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R32_UINT:
            format = PIPE_FORMAT_R32_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R32G32_UINT:
            format = PIPE_FORMAT_R32G32_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R32G32B32_UINT:
            format = PIPE_FORMAT_R32G32B32_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R32G32B32A32_UINT:
            format = PIPE_FORMAT_R32G32B32A32_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R8_UNORM:
            format = PIPE_FORMAT_R8_UNORM;
            break;
        case PSBC_VERTEX_FORMAT_R8_SNORM:
            format = PIPE_FORMAT_R8_SNORM;
            break;
        case PSBC_VERTEX_FORMAT_R8_UINT:
            format = PIPE_FORMAT_R8_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R8_SINT:
            format = PIPE_FORMAT_R8_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R8G8_UNORM:
            format = PIPE_FORMAT_R8G8_UNORM;
            break;
        case PSBC_VERTEX_FORMAT_R8G8_SNORM:
            format = PIPE_FORMAT_R8G8_SNORM;
            break;
        case PSBC_VERTEX_FORMAT_R8G8_UINT:
            format = PIPE_FORMAT_R8G8_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R8G8_SINT:
            format = PIPE_FORMAT_R8G8_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R8G8B8A8_SNORM:
            format = PIPE_FORMAT_R8G8B8A8_SNORM;
            break;
        case PSBC_VERTEX_FORMAT_R8G8B8A8_UINT:
            format = PIPE_FORMAT_R8G8B8A8_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R8G8B8A8_SINT:
            format = PIPE_FORMAT_R8G8B8A8_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R16_UNORM:
            format = PIPE_FORMAT_R16_UNORM;
            break;
        case PSBC_VERTEX_FORMAT_R16_SNORM:
            format = PIPE_FORMAT_R16_SNORM;
            break;
        case PSBC_VERTEX_FORMAT_R16_UINT:
            format = PIPE_FORMAT_R16_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R16_SINT:
            format = PIPE_FORMAT_R16_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R16_FLOAT:
            format = PIPE_FORMAT_R16_FLOAT;
            break;
        case PSBC_VERTEX_FORMAT_R16G16_UNORM:
            format = PIPE_FORMAT_R16G16_UNORM;
            break;
        case PSBC_VERTEX_FORMAT_R16G16_SNORM:
            format = PIPE_FORMAT_R16G16_SNORM;
            break;
        case PSBC_VERTEX_FORMAT_R16G16_UINT:
            format = PIPE_FORMAT_R16G16_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R16G16_SINT:
            format = PIPE_FORMAT_R16G16_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R16G16_FLOAT:
            format = PIPE_FORMAT_R16G16_FLOAT;
            break;
        case PSBC_VERTEX_FORMAT_R16G16B16A16_UNORM:
            format = PIPE_FORMAT_R16G16B16A16_UNORM;
            break;
        case PSBC_VERTEX_FORMAT_R16G16B16A16_SNORM:
            format = PIPE_FORMAT_R16G16B16A16_SNORM;
            break;
        case PSBC_VERTEX_FORMAT_R16G16B16A16_UINT:
            format = PIPE_FORMAT_R16G16B16A16_UINT;
            break;
        case PSBC_VERTEX_FORMAT_R16G16B16A16_SINT:
            format = PIPE_FORMAT_R16G16B16A16_SINT;
            break;
        case PSBC_VERTEX_FORMAT_R16G16B16A16_FLOAT:
            format = PIPE_FORMAT_R16G16B16A16_FLOAT;
            break;
        default:
            if (previous_nir)
                ralloc_free(previous_nir);
            ralloc_free(nir);
            psbc_shutdown();
            return PSBC_RESULT_INTERNAL_ERROR;
        }
        if (attribute->location >= PSBC_MAX_VERTEX_ATTRIBUTES ||
            attribute->binding >= MAX_VBS ||
            (!attribute->stride && opts->target != PSBC_TARGET_PS5) ||
            !attribute->alignment) {
            if (previous_nir)
                ralloc_free(previous_nir);
            ralloc_free(nir);
            psbc_shutdown();
            return PSBC_RESULT_INTERNAL_ERROR;
        }
        const uint32_t location = attribute->location;
        gfx_state.vi.attributes_valid |= BITFIELD_BIT(location);
        gfx_state.vi.vertex_attribute_formats[location] = format;
        gfx_state.vi.vertex_attribute_bindings[location] =
            attribute->binding;
        gfx_state.vi.vertex_attribute_offsets[location] = attribute->offset;
        gfx_state.vi.vertex_attribute_strides[location] = attribute->stride;
        gfx_state.vi.vertex_binding_align[attribute->binding] =
            attribute->alignment;
        if (attribute->instance_divisor) {
            gfx_state.vi.instance_rate_inputs |= BITFIELD_BIT(location);
            gfx_state.vi.instance_rate_divisors[location] =
                attribute->instance_divisor;
        }
    }

    /* RADV normally lowers fragment coordinates before collecting shader
     * info.  Keep standalone compilation in that order so the PS argument
     * map enables POS_FIXED_PT when the optimization selects it. */
    if (mesa_stage == MESA_SHADER_FRAGMENT &&
        !gfx_state.ms.sample_shading_enable &&
        !nir->info.fs.uses_sample_shading)
        NIR_PASS(_, nir, radv_nir_lower_opt_fs_frag_pos,
                 gfx_state.vrs_may_be_enabled,
                 gfx_state.ms.sample_shading_enable ||
                    nir->info.fs.uses_sample_shading);

    radv_nir_shader_info_pass(
        &compiler_info, nir, &layout, &stage.key, &gfx_state,
        RADV_PIPELINE_GRAPHICS, false, &stage.info
    );
    if (paired_geometry) {
        radv_nir_shader_info_pass(
            &compiler_info, previous_nir, &layout, &previous.key, &gfx_state,
            RADV_PIPELINE_GRAPHICS, false, &previous.info
        );
    }
    /* Legacy Gallium texture indices carry no Vulkan deref for RADV's info
     * pass to discover; that caller keeps the explicit PSBC layout. A caller
     * that asserts static descriptor use leaves RADV's NIR-derived set mask
     * exactly as the optimized shader produced it, so a layout binding the NIR
     * never dereferences cannot acquire a native descriptor dependency. */
    uint64_t static_binding_mask[PSBC_MAX_DESCRIPTOR_SETS] = {0};
    if (opts->descriptor_binding_count && !opts->static_descriptor_use) {
        uint32_t descriptor_set_mask = 0;
        for (uint32_t i = 0; i < opts->descriptor_binding_count; ++i)
            descriptor_set_mask |= 1u << opts->descriptor_bindings[i].set;
        stage.info.desc_set_used_mask |= descriptor_set_mask;
        if (paired_geometry)
            previous.info.desc_set_used_mask |= descriptor_set_mask;
    } else if (opts->descriptor_binding_count) {
        /* The deref form is only visible before the descriptor lowering that
         * precedes ACO, so record the used bindings here, next to the info pass
         * that records the used sets from the same sources. */
        uint32_t used_set_mask = 0;
        gather_static_descriptor_use(nir, &used_set_mask, static_binding_mask);
        for (uint32_t i = 0; i < opts->descriptor_binding_count; ++i) {
            const PsbcDescriptorBinding *declared = &opts->descriptor_bindings[i];
            uint32_t set = declared->set;
            if (set >= PSBC_MAX_DESCRIPTOR_SETS || declared->binding >= 64u ||
                !(stage.info.desc_set_used_mask & (1u << set)) ||
                static_binding_mask[set] != 0)
                continue;
            /* A set the NIR uses whose binding the walk could not name must not
             * look unused: fall back to this set's whole declaration. */
            static_binding_mask[set] |= 1ull << declared->binding;
        }
    }
    /* The native PS5 runtime uploads one bounded block and supplies its low
     * gfx1013 address. Prevent RADV from replacing any fields with inline
     * SGPR values so metadata and command encoding have one stable ABI. */
    if (opts->force_indirect_push_constants) {
        if (stage.info.loads_push_constants) {
            stage.info.can_inline_all_push_constants = false;
            stage.info.inline_push_constant_mask = 0;
        }
        if (paired_geometry && previous.info.loads_push_constants) {
            previous.info.can_inline_all_push_constants = false;
            previous.info.inline_push_constant_mask = 0;
        }
    }
    debug_stage("shader-info-end");
    debug_shader_io("info", nir, &stage.info);
    if (paired_geometry)
        debug_shader_io("previous-info", previous_nir, &previous.info);

    if (opts->ngg) {
        struct radv_shader_stage stages[MESA_VULKAN_SHADER_STAGES] = {0};
        stages[mesa_stage] = stage;
        if (paired_geometry)
            stages[MESA_SHADER_VERTEX] = previous;
        radv_nir_shader_info_link(&compiler_info, &gfx_state, stages);
        stage.info = stages[mesa_stage].info;
        if (paired_geometry)
            previous.info = stages[MESA_SHADER_VERTEX].info;
        /* Standalone linking conservatively adds PrimitiveID without an FS.
         * Drop only the implicit, final per-primitive parameter when the
         * caller has proved it dead. Keep explicit outputs and unknown
         * consumers unchanged, and update both code lowering and metadata. */
        if (mesa_stage == MESA_SHADER_VERTEX &&
            !(nir->info.outputs_written & VARYING_BIT_PRIMITIVE_ID) &&
            stage.info.outinfo.export_prim_id_per_primitive &&
            stage.info.outinfo.prim_param_exports == 1 &&
            stage.info.outinfo.vs_output_param_offset[VARYING_SLOT_PRIMITIVE_ID] ==
                stage.info.outinfo.param_exports) {
            if (opts->omit_implicit_primitive_id) {
                stage.info.outinfo.export_prim_id_per_primitive = false;
                stage.info.outinfo.prim_param_exports = 0;
                stage.info.outinfo.vs_output_param_offset[VARYING_SLOT_PRIMITIVE_ID] =
                    AC_EXP_PARAM_UNDEFINED;
            } else if (opts->target == PSBC_TARGET_PS5) {
                /* Native traces have correct counts/offsets but undefined
                 * per-primitive ID values. Use Mesa's per-vertex LDS route:
                 * retain the export slot, disable passthrough, and let the
                 * existing NGG pass size LDS and insert its barrier. */
                stage.info.outinfo.export_prim_id_per_primitive = false;
                stage.info.outinfo.prim_param_exports = 0;
                stage.info.outinfo.export_prim_id = true;
                ++stage.info.outinfo.param_exports;
                stage.info.is_ngg_passthrough = false;
            }
        }
        debug_shader_io("linked", nir, &stage.info);
    }

    /* PSBC exposes explicit descriptor-set tables to its standalone caller. Do not
     * inherit RADV's pipeline-library indirection for merged shaders: the
     * direct set pointer fits the PS5 merged user-SGPR budget and matches the
     * public metadata/runtime ABI. */
    if (opts->descriptor_binding_count) {
        stage.info.force_indirect_descriptors = false;
        if (paired_geometry)
            previous.info.force_indirect_descriptors = false;
    }

    /* Determine previous stage for shader args declaration.
     * HS/GS need previous_stage=VERTEX so that the merged-pipeline args
     * (prolog_inputs, vertex_buffers, etc.) are declared for the NIR
     * lowering passes. The VS-specific input usage slots are filtered
     * out in buildshaderbinary for non-VS stages. */
    mesa_shader_stage previous_stage = MESA_SHADER_NONE;
    switch (mesa_stage) {
    case MESA_SHADER_TESS_CTRL: previous_stage = MESA_SHADER_VERTEX;    break;
    case MESA_SHADER_TESS_EVAL: previous_stage = MESA_SHADER_TESS_CTRL; break;
    case MESA_SHADER_GEOMETRY:  previous_stage = MESA_SHADER_VERTEX;    break;
    default:                    previous_stage = MESA_SHADER_NONE;      break;
    }

    radv_declare_shader_args(
        &compiler_info, &gfx_state, &stage, previous_stage, NULL
    );
    debug_stage("declare-args-end");

    stage.info.user_sgprs_locs = stage.args.user_sgprs_locs;
    stage.info.inline_push_constant_mask = stage.args.ac.inline_push_const_mask;
    if (paired_geometry) {
        previous.args = stage.args;
        previous.info.user_sgprs_locs = stage.info.user_sgprs_locs;
        previous.info.inline_push_constant_mask =
            stage.info.inline_push_constant_mask;
    }

    /* For geometry shaders, run the legacy GS lowering pass.
     * On GFX10.3 (PS5), NGG GS requires merging with the previous stage,
     * which is not possible for standalone compilation. We use the legacy
     * GS path instead, which ACO supports on all GFX levels. */
    if (mesa_stage == MESA_SHADER_GEOMETRY && !stage.info.is_ngg) {
        /* Convert emit_vertex to emit_vertex_with_counter */
        NIR_PASS(_, nir, nir_lower_gs_intrinsics,
                 nir_lower_gs_intrinsics_per_stream);
        NIR_PASS(_, nir, nir_lower_vars_to_ssa);

        ac_nir_lower_legacy_gs_options gs_options = {
            .gfx_level = gfxlevel,
            .export_clipdist_mask = 0xff,
            .write_pos_to_clipvertex = true,
            .has_param_exports = stage.info.outinfo.param_exports > 0,
            .disable_streamout = true,
        };
        nir_shader* gs_copy = NULL;
        ac_nir_legacy_gs_info gs_out_info = {0};
        NIR_PASS(_, nir, ac_nir_lower_legacy_gs, &gs_options, &gs_copy, &gs_out_info);
        /* The GS copy shader is discarded for standalone compilation.
         * The PS4 runtime generates the copy shader separately. */
        if (gs_copy)
            ralloc_free(gs_copy);
    }

    if (paired_geometry)
        radv_postprocess_nir(&compiler_info, &gfx_state, &previous);
    radv_postprocess_nir(&compiler_info, &gfx_state, &stage);
    if (mesa_stage == MESA_SHADER_FRAGMENT &&
        opts->target == PSBC_TARGET_PS5 && opts->provoking_vtx_last) {
        unsigned vertex_id =
            ps5_last_provoking_vertex(opts->primitive_type);
        if (vertex_id)
            nir_shader_intrinsics_pass(nir, lower_flat_input_vertex,
                                       nir_metadata_control_flow,
                                       (void*)opts);
    }
    if (mesa_stage == MESA_SHADER_FRAGMENT &&
        opts->target == PSBC_TARGET_PS5 &&
        !split_ps5_mixed_inputs(nir, &stage.info, opts->primitive_id_per_primitive)) {
        if (previous_nir)
            ralloc_free(previous_nir);
        ralloc_free(nir);
        psbc_shutdown();
        return PSBC_RESULT_INTERNAL_ERROR;
    }
    NIR_PASS(_, nir, nir_opt_licm, psbc_licm_speculatable);
    debug_stage("postprocess-end");
    debug_shader_io("postprocess", nir, &stage.info);
    if (getenv("PSBC_DEBUG_NIR"))
        nir_print_shader(nir, stderr);

    /* Snapshot the final flat/interpolated input forms before ACO consumes
     * and may rewrite the NIR. */
    PsbcInputSemantics input_semantics = {.valid = true};
    if (mesa_stage == MESA_SHADER_FRAGMENT &&
        !build_input_semantics(nir, &stage.info, opts->primitive_id_per_primitive,
                               &input_semantics) && stage.info.ps.prim_id_input) {
        /* Never package a live PrimitiveID consumer with incomplete linkage. */
        ralloc_free(nir);
        psbc_shutdown();
        return PSBC_RESULT_INTERNAL_ERROR;
    }

    if (opts->ngg) {
        gfx10_get_ngg_info(&compiler_info,
                           paired_geometry ? &previous.info : &stage.info,
                           paired_geometry ? &stage.info : NULL,
                           &stage.info.ngg_info);
        stage.info.nir_shared_size = stage.info.ngg_info.lds_size;
    }

    /* Compile NIR to GCN ISA via ACO */
    nir_shader* shaders[2] = {nir, NULL};
    unsigned shader_count = 1;
    if (paired_geometry) {
        shaders[0] = previous_nir;
        shaders[1] = nir;
        shader_count = 2;
    }
    if (getenv("PSBC_DEBUG_DISASM"))
        stage.key.keep_executable_info = true;
    struct radv_shader_binary* binary = radv_shader_nir_to_asm(
        &compiler_info, &stage, shaders, shader_count, &gfx_state
    );
    debug_stage("aco-end");

    if (!binary) {
        if (previous_nir)
            ralloc_free(previous_nir);
        ralloc_free(nir);
        psbc_shutdown();
        return PSBC_RESULT_COMPILE_ACO;
    }

    /* Extract code from radv_shader_binary_legacy */
    struct radv_shader_binary_legacy* legacy =
        (struct radv_shader_binary_legacy*)binary;

    if (getenv("PSBC_DEBUG_DISASM")) {
        fprintf(stderr, "PSBC executable code=%u ir=%u disasm=%u\n",
                legacy->code_size, legacy->ir_size, legacy->disasm_size);
        if (legacy->ir_size) {
            const char* ir = (const char*)legacy->data + legacy->stats_size +
                             legacy->code_size;
            fprintf(stderr, "%.*s\n", (int)legacy->ir_size, ir);
        }
    }
    if (getenv("PSBC_DEBUG_DISASM") && legacy->disasm_size) {
        const uint8_t* disasm = legacy->data + legacy->stats_size +
                                legacy->code_size + legacy->ir_size;
        fwrite(disasm, 1, legacy->disasm_size, stderr);
        fputc('\n', stderr);
    }

    /* The data layout in radv_shader_binary_legacy is:
     * [stats | code | ir | disasm | debug_info]
     * Code starts at offset stats_size. */
    const uint32_t* code = (const uint32_t*)(legacy->data + legacy->stats_size);
    uint32_t code_dw = legacy->code_size / sizeof(uint32_t);

    /* Build the PS4/PS5 shader binary */
    BuildContext buildctx = {
        .rinfo = &binary->info,
        .rargs = &stage.args,
        .config = &binary->config,
        .nir = nir,
        .gfx_level = gfxlevel,
        .family = chipfamily,
        .stage = mesa_stage,
        .psbc_stage = opts->stage,
        .spirv_data = spirv,
        .spirv_size = spirv_size,
        .target = opts->target,
        .ngg = opts->ngg,
        .neo = neo,
        .address32_hi = opts->address32_hi,
        .options = opts,
        .input_semantics = &input_semantics,
    };
    memcpy(buildctx.descriptor_used_binding_mask, static_binding_mask,
           sizeof(buildctx.descriptor_used_binding_mask));

    uint8_t* output_data = NULL;
    size_t output_size = 0;
    PsbcResult result = buildshaderbinary(&buildctx, code, code_dw,
                                          &output_data, &output_size);
    if (result != PSBC_RESULT_OK) {
        free(binary);
        if (previous_nir)
            ralloc_free(previous_nir);
        ralloc_free(nir);
        psbc_shutdown();
        return result;
    }

    void* machine_code = malloc(legacy->code_size);
    if (!machine_code) {
        free(output_data);
        free(binary);
        if (previous_nir)
            ralloc_free(previous_nir);
        ralloc_free(nir);
        psbc_shutdown();
        return PSBC_RESULT_OUT_OF_MEMORY;
    }
    memcpy(machine_code, code, legacy->code_size);

    out->data = output_data;
    out->size = output_size;
    out->machine_code = machine_code;
    out->machine_code_size = legacy->code_size;
    fill_shader_metadata(&buildctx, &out->metadata);

    free(binary);
    if (previous_nir)
        ralloc_free(previous_nir);
    ralloc_free(nir);
    psbc_shutdown();

    return result;
}

PsbcResult psbc_compile_shader(
    const uint32_t* spirv,
    size_t spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
) {
    return psbc_compile_impl(spirv, spirv_size, NULL, NULL, 0, NULL,
                             opts, out);
}

PsbcResult psbc_compile_nir(
    const struct nir_shader* nir,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
) {
    return psbc_compile_impl(NULL, 0, nir, NULL, 0, NULL, opts, out);
}

PsbcResult psbc_compile_geometry_pipeline(
    const uint32_t* vertex_spirv,
    size_t vertex_spirv_size,
    const uint32_t* geometry_spirv,
    size_t geometry_spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
) {
    return psbc_compile_impl(geometry_spirv, geometry_spirv_size, NULL,
                             vertex_spirv, vertex_spirv_size, NULL,
                             opts, out);
}

/* Find one shader register in a metadata block, or NULL when the stage did not
 * publish it. */
static PsbcRegisterWrite* find_shader_register(PsbcShaderMetadata* metadata,
                                               uint16_t offset) {
    for (uint32_t i = 0; i < metadata->shader_register_count; ++i)
        if (metadata->shader_registers[i].offset == offset)
            return &metadata->shader_registers[i];
    return NULL;
}

/* The combined LS/HS resource registers radv produces for a hull stage
 * (radv_shader_combine_cfg_vs_tcs in the pinned tree): the vertex half is the
 * base, the control half raises the VGPR/SGPR/LS_VGPR_COMP_CNT fields, and its
 * scratch enable does not leak into the LS half. */
static void combine_vs_tcs_config(uint32_t vs_rsrc1, uint32_t vs_rsrc2,
                                  uint32_t tcs_rsrc1, uint32_t tcs_rsrc2,
                                  uint32_t* rsrc1_out, uint32_t* rsrc2_out) {
    uint32_t rsrc1 = vs_rsrc1;
    if (G_00B848_VGPRS(tcs_rsrc1) > G_00B848_VGPRS(rsrc1))
        rsrc1 = (rsrc1 & C_00B848_VGPRS) | (tcs_rsrc1 & ~C_00B848_VGPRS);
    if (G_00B228_SGPRS(tcs_rsrc1) > G_00B228_SGPRS(rsrc1))
        rsrc1 = (rsrc1 & C_00B228_SGPRS) | (tcs_rsrc1 & ~C_00B228_SGPRS);
    if (G_00B428_LS_VGPR_COMP_CNT(tcs_rsrc1) > G_00B428_LS_VGPR_COMP_CNT(rsrc1))
        rsrc1 = (rsrc1 & C_00B428_LS_VGPR_COMP_CNT) |
                (tcs_rsrc1 & ~C_00B428_LS_VGPR_COMP_CNT);
    *rsrc1_out = rsrc1;
    *rsrc2_out = vs_rsrc2 | (tcs_rsrc2 & ~C_00B12C_SCRATCH_EN);
}

PsbcResult psbc_compile_tess_pipeline(
    const uint32_t* vertex_spirv,
    size_t vertex_spirv_size,
    const uint32_t* tess_ctrl_spirv,
    size_t tess_ctrl_spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
) {
    if (!out || !opts || !vertex_spirv || !tess_ctrl_spirv)
        return PSBC_RESULT_INVALID_SPIRV;
    if (opts->stage != PSBC_STAGE_TESS_CTRL || opts->target != PSBC_TARGET_PS5)
        return PSBC_RESULT_UNSUPPORTED_STAGE;

    /* The vertex half of a tessellation pipeline feeds the control stage and is
     * not an NGG program, so it is compiled with the control stage as its
     * consumer and with NGG off. */
    PsbcCompileOptions ls_options = *opts;
    ls_options.stage = PSBC_STAGE_VERTEX;
    ls_options.ngg = false;
    PsbcShaderOutput ls = {0};
    PsbcResult result = psbc_compile_shader(vertex_spirv, vertex_spirv_size,
                                            &ls_options, &ls);
    if (result != PSBC_RESULT_OK)
        return result;

    PsbcShaderOutput hs = {0};
    result = psbc_compile_shader(tess_ctrl_spirv, tess_ctrl_spirv_size, opts, &hs);
    if (result != PSBC_RESULT_OK) {
        psbc_free_output(&ls);
        return result;
    }

    const uint16_t ls_lo = PSBC_SH_OFFSET(R_00B520_SPI_SHADER_PGM_LO_LS);
    const uint16_t ls_hi = PSBC_SH_OFFSET(R_00B524_SPI_SHADER_PGM_HI_LS);
    const uint16_t ls_rsrc1_off = PSBC_SH_OFFSET(R_00B528_SPI_SHADER_PGM_RSRC1_LS);
    const uint16_t ls_rsrc2_off = PSBC_SH_OFFSET(R_00B52C_SPI_SHADER_PGM_RSRC2_LS);
    const PsbcRegisterWrite* vs_rsrc1 =
        find_shader_register(&ls.metadata,
                             PSBC_SH_OFFSET(R_00B128_SPI_SHADER_PGM_RSRC1_VS));
    const PsbcRegisterWrite* vs_rsrc2 =
        find_shader_register(&ls.metadata,
                             PSBC_SH_OFFSET(R_00B12C_SPI_SHADER_PGM_RSRC2_VS));
    PsbcRegisterWrite* hs_rsrc1 =
        find_shader_register(&hs.metadata,
                             PSBC_SH_OFFSET(R_00B428_SPI_SHADER_PGM_RSRC1_HS));
    PsbcRegisterWrite* hs_rsrc2 =
        find_shader_register(&hs.metadata,
                             PSBC_SH_OFFSET(R_00B42C_SPI_SHADER_PGM_RSRC2_HS));
    if (!vs_rsrc1 || !vs_rsrc2 || !hs_rsrc1 || !hs_rsrc2) {
        psbc_free_output(&ls);
        psbc_free_output(&hs);
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    }

    uint32_t combined_rsrc1 = 0, combined_rsrc2 = 0;
    combine_vs_tcs_config(vs_rsrc1->value, vs_rsrc2->value,
                          hs_rsrc1->value, hs_rsrc2->value,
                          &combined_rsrc1, &combined_rsrc2);
    hs_rsrc1->value = combined_rsrc1;
    hs_rsrc2->value = combined_rsrc2;

    hs.metadata.hull_ls_valid = true;
    hs.metadata.hull_ls_code_size = (uint32_t)ls.machine_code_size;
    /* Carry the LS program in the same buffer as the HS program: the HS half
     * keeps offset 0, and hull_ls_code_offset is where the LS half starts.  A
     * consumer that wants one program still reads the bytes it always did. */
    const size_t hs_code_size = hs.machine_code_size;
    uint8_t* combined = malloc(hs_code_size + ls.machine_code_size);
    if (!combined) {
        psbc_free_output(&ls);
        psbc_free_output(&hs);
        return PSBC_RESULT_OUT_OF_MEMORY;
    }
    memcpy(combined, hs.machine_code, hs_code_size);
    memcpy(combined + hs_code_size, ls.machine_code, ls.machine_code_size);
    free(hs.machine_code);
    hs.machine_code = combined;
    hs.machine_code_size = hs_code_size + ls.machine_code_size;
    hs.metadata.hull_ls_code_offset = (uint32_t)hs_code_size;
    hs.metadata.hull_ls_pgm_lo = (PsbcRegisterWrite){.offset = ls_lo, .value = 0};
    hs.metadata.hull_ls_pgm_hi = (PsbcRegisterWrite){.offset = ls_hi, .value = 0};
    hs.metadata.hull_ls_rsrc1 =
        (PsbcRegisterWrite){.offset = ls_rsrc1_off, .value = vs_rsrc1->value};
    hs.metadata.hull_ls_rsrc2 =
        (PsbcRegisterWrite){.offset = ls_rsrc2_off, .value = vs_rsrc2->value};
    /* The LS machine code itself is not packaged yet, so the result stays
     * explicitly short of a loadable hull package. */
    hs.metadata.unresolved_fields |= PSBC_UNRESOLVED_TESS_PIPELINE;
    psbc_free_output(&ls);

    *out = hs;
    return PSBC_RESULT_OK;
}

PsbcResult psbc_compile_nir_geometry_pipeline(
    const struct nir_shader* vertex_nir,
    const struct nir_shader* geometry_nir,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
) {
    return psbc_compile_impl(NULL, 0, geometry_nir, NULL, 0, vertex_nir,
                             opts, out);
}

void psbc_free_output(PsbcShaderOutput* out) {
    if (!out)
        return;
    if (out->data)
        free(out->data);
    if (out->machine_code)
        free(out->machine_code);
    memset(out, 0, sizeof(*out));
}

const char* psbc_result_string(PsbcResult result) {
    switch (result) {
    case PSBC_RESULT_OK:               return "success";
    case PSBC_RESULT_INVALID_SPIRV:    return "invalid SPIR-V bytecode";
    case PSBC_RESULT_UNSUPPORTED_STAGE: return "unsupported shader stage";
    case PSBC_RESULT_COMPILE_NIR:      return "SPIR-V to NIR compilation failed";
    case PSBC_RESULT_COMPILE_ACO:      return "ACO shader compilation failed";
    case PSBC_RESULT_OUT_OF_MEMORY:    return "out of memory";
    case PSBC_RESULT_INTERNAL_ERROR:   return "internal error";
    case PSBC_RESULT_UNSUPPORTED_CAPABILITY: return "unsupported SPIR-V capability";
    default:                           return "unknown error";
    }
}

PsbcStage psbc_stage_from_name(const char* name) {
    if (!name)
        return PSBC_STAGE_NONE;
    if (!strcmp(name, "vertex"))      return PSBC_STAGE_VERTEX;
    if (!strcmp(name, "tess-ctrl"))   return PSBC_STAGE_TESS_CTRL;
    if (!strcmp(name, "tess-eval"))   return PSBC_STAGE_TESS_EVAL;
    if (!strcmp(name, "geometry"))    return PSBC_STAGE_GEOMETRY;
    if (!strcmp(name, "fragment"))    return PSBC_STAGE_FRAGMENT;
    if (!strcmp(name, "compute"))     return PSBC_STAGE_COMPUTE;
    if (!strcmp(name, "task"))        return PSBC_STAGE_TASK;
    if (!strcmp(name, "export"))      return PSBC_STAGE_EXPORT;
    if (!strcmp(name, "local"))       return PSBC_STAGE_LOCAL;
    return PSBC_STAGE_NONE;
}
