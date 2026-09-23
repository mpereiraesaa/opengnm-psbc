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

/* PS5 is GFX10_1 hardware, but this standalone compiler uses GFX10_3/NAVI21
 * granularity for ACO. The latter reads subgroup ID from the GFX10_3 TG_SIZE
 * wave-ID field, which the PS5 compute dispatch does not supply. Form the
 * index from the local invocation coordinates instead. This also works for
 * 2D/3D workgroups and does not depend on a dispatch packet's wave-ID field. */
static bool lower_ps5_compute_subgroup_id(nir_builder* b, nir_instr* instruction,
                                           void* data) {
    if (instruction->type != nir_instr_type_intrinsic)
        return false;
    nir_intrinsic_instr* intrinsic = nir_instr_as_intrinsic(instruction);
    if (intrinsic->intrinsic != nir_intrinsic_load_subgroup_id)
        return false;
    const unsigned wave_size = *(const unsigned*)data;
    if (wave_size != 32 && wave_size != 64)
        return false;

    b->cursor = nir_before_instr(instruction);
    nir_def* local = nir_load_local_invocation_id(b);
    nir_def* x = nir_channel(b, local, 0);
    nir_def* y = nir_channel(b, local, 1);
    nir_def* z = nir_channel(b, local, 2);
    nir_def* width;
    nir_def* height;
    if (b->shader->info.workgroup_size_variable) {
        nir_def* size = nir_load_workgroup_size(b);
        width = nir_channel(b, size, 0);
        height = nir_channel(b, size, 1);
    } else {
        width = nir_imm_int(b, b->shader->info.workgroup_size[0]);
        height = nir_imm_int(b, b->shader->info.workgroup_size[1]);
    }
    nir_def* index = nir_iadd(b, x, nir_imul(b, width,
                           nir_iadd(b, y, nir_imul(b, height, z))));
    nir_def_replace(&intrinsic->def, nir_udiv_imm(b, index, wave_size));
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
    /* ES half of a merged vertex+geometry pair; NULL for every other compile. */
    const struct radv_shader_info* es_info;
    /* True when this control-stage compile carried its vertex half, so the
     * published program is a real merged LS/HS image rather than a control
     * half that has nothing to run with. */
    bool ls_merged;
    bool domain_linked; /* evaluation half linked to its control half */
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
    bool hull_push_use_valid;
    bool push_use_valid;
    uint64_t stage_push_dwords, previous_stage_push_dwords;
    uint64_t hull_vertex_push_dwords, hull_control_push_dwords;
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
                /* A distance the PIXEL stage reads is a packed position
                 * register: the rasterizer interpolates it like a varying and
                 * the linker places it at the attribute slot of the register
                 * the pre-raster stage exported. Name that register with the
                 * same private key the export side uses, so the two halves pair
                 * on the register (not on the feature - one register holds clip
                 * and cull components together) and the producer's word carries
                 * the parameter index to interpolate from. A stage whose
                 * declared width is outside the two packed registers is left
                 * undescribed, which the pipeline then refuses. */
                const bool clip_distance = io.location == VARYING_SLOT_CLIP_DIST0 ||
                    io.location == VARYING_SLOT_CLIP_DIST1;
                const bool cull_distance = io.location == VARYING_SLOT_CULL_DIST0 ||
                    io.location == VARYING_SLOT_CULL_DIST1;
                if (clip_distance || cull_distance) {
                    /* nir_merge_clip_cull_distance_vars packs both built-ins
                     * into CLIP_DIST slots, including a cull-only interface.
                     * An unlowered CULL_DIST slot has no packed identity yet. */
                    if (cull_distance)
                        return false;
                    const unsigned declared = nir->info.clip_distance_array_size +
                        nir->info.cull_distance_array_size;
                    const uint32_t attribute = nir_intrinsic_base(intrin);
                    /* base is the compact pixel input index, not the packed
                     * distance register. Reading only ClipDistance[4] leaves
                     * CLIP_DIST1 at base zero. Preserve the logical location
                     * when constructing the producer/consumer match key. */
                    const uint32_t distance_register =
                        io.location - VARYING_SLOT_CLIP_DIST0;
                    if (io.num_slots != 1 || !declared || declared > 8u ||
                        attribute >= generic_count ||
                        distance_register > 1u)
                        return false;
                    const uint32_t word =
                        PSBC_SEMANTIC_DISTANCE_REGISTER + distance_register;
                    if (seen[attribute] && words_by_attribute[attribute] != word)
                        return false;
                    words_by_attribute[attribute] = word;
                    seen[attribute] = true;
                    continue;
                }
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
    /* The geometry stage's viewport selection is a parameter export too: RADV
     * gives it a slot in vs_output_param_offset and counts it in
     * param_exports, so a merged pair that writes gl_ViewportIndex exports one
     * parameter that the varying loop above cannot describe - the name it
     * carries is VARYING_SLOT_VIEWPORT rather than a user location. Naming it
     * here is what keeps the list and the export count equal; without it the
     * whole linkage field stays unresolved and the pair is refused, which is
     * exactly the shape the viewport-routing witness hit. A pipeline that does
     * not write it is byte-identical to before, because this emits nothing. */
    const uint8_t viewport_index =
        info->outinfo.vs_output_param_offset[VARYING_SLOT_VIEWPORT];
    if (viewport_index < PSBC_MAX_SEMANTICS) {
        if (metadata->output_semantic_count >= PSBC_MAX_SEMANTICS)
            return false;
        metadata->output_semantics[metadata->output_semantic_count++] =
            PSBC_SEMANTIC_VIEWPORT_INDEX | ((uint32_t)viewport_index << 8);
    }
    /* The packed distance registers are pixel attributes too, and a pixel
     * stage that reads a distance needs to name the register it reads: publish
     * one word per register, after the described varyings and with the
     * parameter index those registers occupy in the export space. Without this
     * the pixel input list stays unresolved for a distance attribute and the
     * AGC linker has nothing to map the read to (the pin's behaviour). */
    const unsigned clip_components = util_bitcount(info->outinfo.clip_dist_mask);
    const unsigned cull_components = util_bitcount(info->outinfo.cull_dist_mask);
    const unsigned distance_registers =
        (clip_components + cull_components + 3u) / 4u;
    for (unsigned register_index = 0; register_index < distance_registers;
         ++register_index) {
        /* TES can insert an implicit PrimitiveID parameter between generic
         * varyings and distances. Use the assigned export index, never the
         * number of user varying semantics as an approximation. */
        const uint8_t parameter = info->outinfo.vs_output_param_offset[
            VARYING_SLOT_CLIP_DIST0 + register_index];
        if (metadata->output_semantic_count >= PSBC_MAX_SEMANTICS ||
            parameter >= PSBC_MAX_SEMANTICS)
            return false;
        metadata->output_semantics[metadata->output_semantic_count++] =
            (PSBC_SEMANTIC_DISTANCE_REGISTER + register_index) |
            ((uint32_t)parameter << 8);
    }
    return metadata->output_semantic_count ==
        info->outinfo.param_exports + info->outinfo.prim_param_exports;
}

/* The GE's parameter-cache allocation, when the caller supplied the device
 * facts. radv programs GE_PC_ALLOC for every gfx10 pre-raster pipeline, NGG or
 * legacy, through the same ac_compute_late_alloc(); the NGG argument is the
 * program's own shape. */
static void publish_ge_pc_alloc(const BuildContext* ctx,
                                PsbcShaderMetadata* metadata) {
    if (ctx->options->ngg_device_facts) {
        struct radeon_info facts = {0};
        unsigned late_alloc_wave64 = 0;
        unsigned cu_mask = 0xffff;
        facts.gfx_level = ctx->gfx_level;
        facts.family = ctx->family;
        facts.pc_lines = ctx->options->ngg_pc_lines;
        facts.min_good_cu_per_sa = ctx->options->ngg_min_good_cu_per_sa;
        ac_compute_late_alloc(&facts, ctx->ngg, ctx->options->ngg_culling,
                              ctx->options->ngg_uses_scratch,
                              &late_alloc_wave64, &cu_mask);
        uint32_t oversub_pc_lines =
            late_alloc_wave64 ? ctx->options->ngg_pc_lines / 4 : 0;
        if (ctx->options->ngg_culling) {
            /* Same oversubscription factor radv applies to a culling NGG
             * pipeline, from the exports this stage already produces. */
            unsigned oversub_factor = 2;
            if (ctx->rinfo->outinfo.param_exports > 4)
                oversub_factor = 4;
            else if (ctx->rinfo->outinfo.param_exports > 2)
                oversub_factor = 3;
            oversub_pc_lines *= oversub_factor;
        }
        metadata->linkage_ge_pc_alloc_valid = true;
        metadata->linkage_ge_pc_alloc = (PsbcRegisterWrite) {
            .offset = PSBC_UC_OFFSET(R_030980_GE_PC_ALLOC),
            .value = S_030980_OVERSUB_EN(oversub_pc_lines > 0) |
                     S_030980_NUM_PC_LINES(oversub_pc_lines - 1),
        };
    }
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
static bool gather_push_dwords(const nir_shader *nir, uint64_t *mask)
{
    *mask = 0;
    nir_foreach_function_impl(impl, nir) {
        nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
                if (instr->type != nir_instr_type_intrinsic) continue;
                const nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
                if (intrin->intrinsic != nir_intrinsic_load_push_constant) continue;
                uint64_t offset = nir_intrinsic_base(intrin);
                uint64_t size = nir_intrinsic_range(intrin);
                if (nir_src_is_const(intrin->src[0])) {
                    offset += nir_src_as_uint(intrin->src[0]);
                    size = intrin->num_components * (intrin->def.bit_size / 8u);
                }
                if (!size || offset >= 256 || size > 256 - offset) {
                    *mask = 0; return false;
                }
                for (unsigned dw = (unsigned)(offset / 4);
                     dw < (offset + size + 3) / 4; ++dw)
                    *mask |= UINT64_C(1) << dw;
            }
        }
    }
    return true;
}

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
    } else if (nir->info.tess._primitive_mode == TESS_PRIMITIVE_ISOLINES) {
        tf_topology = V_028B6C_OUTPUT_LINE;
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
    metadata->hull_push_use_valid = ctx->hull_push_use_valid;
    metadata->push_use_valid = ctx->push_use_valid;
    metadata->stage_push_dwords = ctx->stage_push_dwords;
    metadata->previous_stage_push_dwords = ctx->previous_stage_push_dwords;
    metadata->hull_vertex_push_dwords = ctx->hull_vertex_push_dwords;
    metadata->hull_control_push_dwords = ctx->hull_control_push_dwords;
    if (ctx->rinfo->loads_push_constants &&
        ctx->rargs->ac.push_constants.used) {
        metadata->push_constants_valid = true;
        metadata->push_constants_user_data_dword =
            ctx->rargs->user_sgprs_locs
                .shader_data[AC_UD_PUSH_CONSTANTS].sgpr_idx;
        metadata->push_constant_size = ctx->rinfo->push_constant_size;
    }
    if (ctx->rargs->ps5_ring_table.used) {
        metadata->ps5_ring_table_valid = true;
        metadata->ps5_ring_table_user_data_dword =
            ctx->rargs->user_sgprs_locs.shader_data[AC_UD_PS5_RING_TABLE].sgpr_idx;
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
    const bool has_vertex_half = ctx->stage == MESA_SHADER_TESS_CTRL &&
        ctx->rinfo->vs.as_ls;
    if ((has_vertex_half || ctx->stage == MESA_SHADER_VERTEX ||
         (ctx->stage == MESA_SHADER_GEOMETRY && ctx->ngg)) &&
        ctx->rargs->ac.vertex_buffers.used &&
        (!has_vertex_half || ctx->rinfo->vs.vb_desc_usage_mask)) {
        metadata->vertex_buffer_table_valid = true;
        metadata->vertex_buffer_usage_mask = ctx->rinfo->vs.vb_desc_usage_mask;
        metadata->vertex_buffer_per_attribute = ctx->rinfo->vs.use_per_attribute_vb_descs;
        metadata->vertex_buffer_table_user_data_dword =
            ctx->rargs->user_sgprs_locs
                .shader_data[AC_UD_VS_VERTEX_BUFFERS].sgpr_idx;
    }
    if ((has_vertex_half || ctx->stage == MESA_SHADER_VERTEX ||
         (ctx->stage == MESA_SHADER_GEOMETRY && ctx->ngg)) &&
        ctx->rargs->ac.base_vertex.used) {
        metadata->base_vertex_valid = true;
        metadata->base_vertex_user_data_dword =
            ctx->rargs->user_sgprs_locs
                .shader_data[AC_UD_VS_BASE_VERTEX_START_INSTANCE].sgpr_idx;
    }
    if ((has_vertex_half || ctx->stage == MESA_SHADER_VERTEX ||
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
    if ((has_vertex_half || ctx->stage == MESA_SHADER_VERTEX ||
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
        /* What this pixel stage reads of the distance built-ins: the SPIR-V
         * reader records the declared widths, so a consumer learns the input
         * width without re-parsing the module. */
        metadata->ps_clip_distance_reads = ctx->nir->info.clip_distance_array_size;
        metadata->ps_cull_distance_reads = ctx->nir->info.cull_distance_array_size;
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
         ctx->stage == MESA_SHADER_TESS_EVAL ||
         ctx->stage == MESA_SHADER_GEOMETRY) && ctx->ngg) {
        const bool has_geometry = ctx->stage == MESA_SHADER_GEOMETRY;
        const bool merged_tess_eval = ctx->es_info &&
            ctx->es_info->stage == MESA_SHADER_TESS_EVAL;
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
            .value = S_028B54_ES_EN(merged_tess_eval?
                         V_028B54_ES_STAGE_DS:V_028B54_ES_STAGE_REAL) |
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
        /* Publish the ES half of a merged vertex+geometry pair and, when the
         * caller supplied the device facts, the GE PC-line allocation radv
         * programs for every NGG pipeline.  Nothing here is derived from a
         * guessed constant: the merged fields come from the linked ES shader
         * info this compile already built, and the allocation comes from the
         * caller's device facts through the same ac_compute_late_alloc() radv
         * uses. */
        if (ctx->es_info) {
            metadata->merged_geometry = true;
            metadata->merged_es_source_stage=merged_tess_eval?
                PSBC_STAGE_TESS_EVAL:PSBC_STAGE_VERTEX;
            metadata->merged_es_itemsize = ctx->es_info->esgs_itemsize;
            metadata->merged_esgs_ring_itemsize =
                ctx->rinfo->ngg_info.vgt_esgs_ring_itemsize;
        }
        publish_ge_pc_alloc(ctx, metadata);
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
        /* The same counts the register above describes, published for the two
         * system SGPRs a merged pair reads: the shader uses them to disable the
         * lanes each half does not need, and a caller that cannot supply them
         * would run a different number of ES and GS lanes per workgroup than
         * the program was built for. */
        if (ctx->rargs->ac.gs_tg_info.used &&
            ctx->rargs->ac.merged_wave_info.used) {
            metadata->esgs_system_sgprs_valid = true;
            metadata->esgs_gs_tg_info_sgpr =
                ctx->rargs->ac.args[ctx->rargs->ac.gs_tg_info.arg_index].offset;
            metadata->esgs_merged_wave_info_sgpr =
                ctx->rargs->ac.args[ctx->rargs->ac.merged_wave_info.arg_index].offset;
            metadata->esgs_es_verts_per_subgroup =
                ctx->rinfo->ngg_info.hw_max_esverts;
            metadata->esgs_gs_inst_prims_per_subgroup =
                ctx->rinfo->ngg_info.max_gsprims * gs_num_invocations;
            metadata->esgs_prim_amp_factor = ctx->rinfo->ngg_info.prim_amp_factor;
            metadata->esgs_workgroup_size = ac_compute_ngg_workgroup_size(
                ctx->rinfo->ngg_info.hw_max_esverts,
                ctx->rinfo->ngg_info.max_gsprims * gs_num_invocations,
                ctx->rinfo->ngg_info.max_out_verts,
                ctx->rinfo->ngg_info.prim_amp_factor);
        }
        /* Every driver-visible argument lands at its user-data dword plus the
         * stage's window base, so the base must be the same for all of them.
         * Publish it only when the pairs that carry both an SGPR offset and a
         * dword index agree - a consumer that derived it from one pair would
         * silently misplace the whole block if that pair were inconsistent. */
        {
            struct {
                const struct ac_arg *arg;
                uint32_t dword;
                bool published;
            } pairs[4] = {
                { &ctx->rargs->ac.base_vertex, metadata->base_vertex_user_data_dword,
                  metadata->base_vertex_valid },
                { &ctx->rargs->ac.start_instance, metadata->start_instance_user_data_dword,
                  metadata->start_instance_valid },
                { &ctx->rargs->ac.draw_id, metadata->draw_id_user_data_dword,
                  metadata->draw_id_valid },
                { &ctx->rargs->ngg_lds_layout, metadata->ngg_lds_layout_user_data_dword,
                  metadata->ngg_lds_layout_valid },
            };
            uint32_t base = 0;
            bool any = false, agree = true;
            for (unsigned i = 0; i < 4 && agree; ++i) {
                if (!pairs[i].published || !pairs[i].arg->used)
                    continue;
                const uint32_t offset =
                    ctx->rargs->ac.args[pairs[i].arg->arg_index].offset;
                if (offset < pairs[i].dword) {
                    agree = false;
                    break;
                }
                if (!any) {
                    base = offset - pairs[i].dword;
                    any = true;
                } else if (offset - pairs[i].dword != base) {
                    agree = false;
                }
            }
            if (any && agree)
                metadata->user_data_window_base = base;
        }
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
        publish_ge_pc_alloc(ctx, metadata);

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
        /* The three registers an NGG pipeline leaves to the engine and a
         * legacy one must state: GS mode off, vertex reuse on, no primitive
         * id export - the same set the legacy evaluation half publishes. */
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028A40_VGT_GS_MODE), 0);
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028AB4_VGT_REUSE_OFF), 0);
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028A84_VGT_PRIMITIVEID_EN), 0);

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
        /* The gfx10 legacy-VS resource registers radv's preamble writes:
         * every CU enabled, the wave limit open, late allocation off (PAL
         * documents LIMIT = 0 as a supported configuration). */
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B118_SPI_SHADER_PGM_RSRC3_VS),
            S_00B118_CU_EN(0xffff) | S_00B118_WAVE_LIMIT(0x3f));
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B104_SPI_SHADER_PGM_RSRC4_VS),
            S_00B104_CU_EN(0xffff));
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B11C_SPI_SHADER_LATE_ALLOC_VS),
            S_00B11C_LIMIT(0));
        return;
    }

    if (ctx->stage == MESA_SHADER_TESS_CTRL) {
        /* Publish the merged LS/HS user-data window from the actual ring
         * argument, not from an NGG-only LDS-layout argument absent here. */
        if (has_vertex_half && ctx->rargs->ps5_ring_table.used) {
            const uint32_t offset = ctx->rargs->ac.args[
                ctx->rargs->ps5_ring_table.arg_index].offset;
            if (offset >= metadata->ps5_ring_table_user_data_dword)
                metadata->user_data_window_base =
                    offset - metadata->ps5_ring_table_user_data_dword;
        }
        /* The merged LS/HS program.  On GFX10 this hardware stage is ONE
         * program counter, and the pinned radv_get_shader_regs() puts it at
         * the LS block while the resource pair stays at the HS block:
         *
         *   AC_HW_HULL_SHADER, gfx_level >= GFX10:
         *     pgm_lo    = R_00B520_SPI_SHADER_PGM_LO_LS
         *     pgm_rsrc1 = R_00B428_SPI_SHADER_PGM_RSRC1_HS
         *     pgm_rsrc2 = R_00B42C_SPI_SHADER_PGM_RSRC2_HS
         *
         * R_00B420_SPI_SHADER_PGM_LO_HS is the address register only for
         * gfx_level < GFX9, so it is deliberately NOT published here: writing
         * it on this target names no program the hardware will launch.  The
         * config is the merged binary's own - ACO compiled both halves in one
         * call, so there is nothing left to combine by hand. */
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028B6C_VGT_TF_PARAM), tess_tf_param(ctx->nir));
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B520_SPI_SHADER_PGM_LO_LS), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B524_SPI_SHADER_PGM_HI_LS), 0);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B428_SPI_SHADER_PGM_RSRC1_HS),
            ctx->config->rsrc1);
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B42C_SPI_SHADER_PGM_RSRC2_HS),
            ctx->config->rsrc2);
        /* The tessellation workgroup layout this stage computed when the
         * caller supplied the pipeline's patch control points: the patch count
         * per workgroup, the LDS the LS/HS workgroup needs and its size. This
         * is the compiler's own tessellation workgroup computation (the same
         * numbers the io-to-mem lowering sized the offchip layout with), which
         * is what makes the driver's VGT_LS_HS_CONFIG and launch derivation
         * come from the compile rather than from a driver guess. */
        if (ctx->rinfo->tcs.lds_size || ctx->rinfo->num_tess_patches) {
            metadata->hull_tess_wg_valid = true;
            metadata->hull_num_patches_per_wg = ctx->rinfo->num_tess_patches;
            metadata->hull_tcs_lds_size = ctx->rinfo->tcs.lds_size;
            metadata->hull_workgroup_size = ctx->rinfo->workgroup_size;
        }
        /* A merged pair is a launchable hull program: one image, its address
         * register, its resource pair and the workgroup layout are all here.
         * A control half compiled ALONE still is not - it has no vertex half
         * to run with - so it keeps the unresolved bit. */
        if (!ctx->ls_merged)
            metadata->unresolved_fields |= PSBC_UNRESOLVED_TESS_PIPELINE;
        else
            metadata->hardware_stage = PSBC_HW_STAGE_HULL;
        return;
    }

    if (ctx->stage == MESA_SHADER_TESS_EVAL && !ctx->ngg && ctx->domain_linked) {
        /* The evaluation half as a LEGACY hardware vertex shader: radv's
         * "Tessellation Evaluation Shader as VS" (radv_shader.c). The stage is
         * fed by the tessellator through VS_EN = VS_STAGE_DS, with no ES/GS
         * and no primitive generator, and its program lives in the VS block.
         * radv_postprocess_config already put the TES-as-VS facts into the
         * resource pair: VGPR_COMP_CNT for the patch/tess-coord inputs and
         * OC_LDS_EN for the off-chip patch data.
         *
         * The stage enables follow radv_pipeline_generate_vgt_shader_config
         * for a legacy tessellation pipeline on gfx9+: LS and HS on, the
         * domain on the VS stage, DYNAMIC_HS, and the gfx9+ primgroup wave
         * bound. The remaining context state is the legacy vertex shape's
         * plus the three registers an NGG pipeline leaves to the engine and a
         * legacy one must clear: GS mode off, vertex reuse on, no primitive
         * id export. */
        const uint32_t nparams = MAX2(ctx->rinfo->outinfo.param_exports, 1);
        const unsigned num_pos_exports =
            get_num_pos_exports(ctx->rinfo, NULL, NULL);

        metadata->hardware_stage = PSBC_HW_STAGE_VERTEX;
        if (!fill_output_semantics(ctx->rinfo, metadata))
            metadata->unresolved_fields |= PSBC_UNRESOLVED_AGC_LINKAGE;
        metadata->linkage_valid = true;
        metadata->linkage_ge_cntl = (PsbcRegisterWrite) {
            .offset = PSBC_UC_OFFSET(R_03096C_GE_CNTL),
            .value = S_03096C_PRIM_GRP_SIZE_GFX10(128) |
                     S_03096C_VERT_GRP_SIZE(256),
        };
        metadata->linkage_stages_en = (PsbcRegisterWrite) {
            .offset = PSBC_CX_OFFSET(R_028B54_VGT_SHADER_STAGES_EN),
            .value = S_028B54_LS_EN(V_028B54_LS_STAGE_ON) |
                     S_028B54_HS_EN(V_028B54_HS_STAGE_ON) |
                     S_028B54_VS_EN(V_028B54_VS_STAGE_DS) |
                     S_028B54_DYNAMIC_HS(1) |
                     S_028B54_MAX_PRIMGRP_IN_WAVE(2) |
                     S_028B54_VS_W32_EN(ctx->rinfo->wave_size == 32),
        };
        metadata->linkage_user_vgpr_en = (PsbcRegisterWrite) {
            .offset = PSBC_UC_OFFSET(R_030988_GE_USER_VGPR_EN),
            .value = 0,
        };
        publish_ge_pc_alloc(ctx, metadata);

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
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028A40_VGT_GS_MODE), 0);
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028AB4_VGT_REUSE_OFF), 0);
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028A84_VGT_PRIMITIVEID_EN), 0);
        /* radv_precompute_registers_hw_vs, gfx10+: "Required programming for
         * tessellation (legacy pipeline only)". The legacy domain's wave
         * grouping comes from this register even though no GS is present; the
         * values are radv's, verbatim. Left at an NGG program's counts, the
         * first legacy patch draw on this device stalled past its fence. */
        metadata_add_register(cx, cx_count, PSBC_MAX_CONTEXT_REGISTERS,
            PSBC_CX_OFFSET(R_028A44_VGT_GS_ONCHIP_CNTL),
            S_028A44_ES_VERTS_PER_SUBGRP(250) |
            S_028A44_GS_PRIMS_PER_SUBGRP(126) |
            S_028A44_GS_INST_PRIMS_IN_SUBGRP(126));

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
        /* The gfx10 legacy-VS resource registers radv's preamble writes and
         * this platform has nothing else to write: every CU enabled, the
         * wave limit open, and late allocation off (the conservative
         * ac_compute_late_alloc fallback, as for the NGG stage). */
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B118_SPI_SHADER_PGM_RSRC3_VS),
            S_00B118_CU_EN(0xffff) | S_00B118_WAVE_LIMIT(0x3f));
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B104_SPI_SHADER_PGM_RSRC4_VS),
            S_00B104_CU_EN(0xffff));
        metadata_add_register(sh, sh_count, PSBC_MAX_SHADER_REGISTERS,
            PSBC_SH_OFFSET(R_00B11C_SPI_SHADER_LATE_ALLOC_VS),
            S_00B11C_LIMIT(0));
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

/* Distance arrays are independent API built-ins but share hardware vectors.
 * Scalarize before moving cull components so a non-vec4 clip prefix can split
 * a formerly vectorized load across two parameter registers safely. */
static bool link_fragment_distance_layout(nir_shader *nir,
                                          const PsbcCompileOptions *opts) {
    if (!opts->fragment_distance_layout_valid || nir->info.stage != MESA_SHADER_FRAGMENT)
        return true;
    const unsigned old_clip = nir->info.clip_distance_array_size;
    const unsigned cull = nir->info.cull_distance_array_size;
    const unsigned prefix = opts->fragment_clip_distance_count;
    if (prefix > 8 || prefix < old_clip || cull > 8 - prefix)
        return false;
    if (!cull || prefix == old_clip)
        return true;
    NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_in, NULL, NULL);
    nir_foreach_function_impl(impl, nir) {
        nir_builder b = nir_builder_create(impl);
        nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
                if (instr->type != nir_instr_type_intrinsic)
                    continue;
                nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
                if (in->intrinsic != nir_intrinsic_load_input &&
                    in->intrinsic != nir_intrinsic_load_interpolated_input)
                    continue;
                nir_io_semantics io = nir_intrinsic_io_semantics(in);
                if (io.location != VARYING_SLOT_CLIP_DIST0 &&
                    io.location != VARYING_SLOT_CLIP_DIST1)
                    continue;
                nir_src *offset = nir_get_io_offset_src(in);
                if (!offset || !nir_src_is_const(*offset) || in->num_components != 1 ||
                    nir_src_as_uint(*offset) > 1 || nir_intrinsic_component(in) > 3)
                    return false;
                unsigned component = (io.location - VARYING_SLOT_CLIP_DIST0) * 4 +
                    nir_intrinsic_component(in) + nir_src_as_uint(*offset) * 4;
                if (component < old_clip)
                    continue;
                if (component - old_clip >= cull)
                    return false;
                component += prefix - old_clip;
                io.location = VARYING_SLOT_CLIP_DIST0 + component / 4;
                io.num_slots = 1;
                nir_intrinsic_set_io_semantics(in, io);
                nir_intrinsic_set_component(in, component % 4);
                b.cursor = nir_before_instr(instr);
                nir_src_rewrite(offset, nir_imm_int(&b, 0));
            }
        }
    }
    return true;
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

    if (opts->target == PSBC_TARGET_PS5 && nir->info.stage == MESA_SHADER_COMPUTE) {
        const unsigned wave_size = compiler_info->key.cs_wave_size;
        nir_shader_instructions_pass(nir, lower_ps5_compute_subgroup_id,
                                     nir_metadata_control_flow, (void*)&wave_size);
    }

    /* Indirect array lowering can replace the distance variables with
     * temporaries before gather_info runs again. Preserve their declared
     * widths from the normalized entry-point interface, not the surviving
     * variable list after optimization. */
    const unsigned fragment_clip_count = nir->info.clip_distance_array_size;
    const unsigned fragment_cull_count = nir->info.cull_distance_array_size;

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
    if (nir->info.stage == MESA_SHADER_FRAGMENT) {
        nir->info.clip_distance_array_size = fragment_clip_count;
        nir->info.cull_distance_array_size = fragment_cull_count;
    }
    radv_nir_lower_io(nir);
    if (!link_fragment_distance_layout(nir, opts)) {
        ralloc_free(nir);
        return NULL;
    }
    if (nir->info.stage == MESA_SHADER_FRAGMENT && opts->fragment_distance_layout_valid) {
        nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
        nir->info.clip_distance_array_size = fragment_clip_count;
        nir->info.cull_distance_array_size = fragment_cull_count;
    }
    /* TCS offchip visibility is gathered from lowered store_output and
     * store_per_vertex_output, not the preceding store_deref form. Leaving
     * these masks at zero makes ac_nir_get_tess_io_info discard all stores
     * consumed by TES, even when the linked TES input mask is correct. */
    if (nir->info.stage == MESA_SHADER_TESS_CTRL)
        nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
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

static bool specialization_parameters_valid(uint32_t count,
    const PsbcSpecializationConstant *entries)
{
    if (count > PSBC_MAX_SPECIALIZATION_CONSTANTS) return false;
    for (uint32_t i = 0; i < count; ++i) {
        if (!entries[i].size || entries[i].size > PSBC_MAX_SPECIALIZATION_BYTES)
            return false;
        for (uint32_t j = 0; j < i; ++j)
            if (entries[j].constant_id == entries[i].constant_id) return false;
    }
    return true;
}

static void select_linked_parameters(PsbcCompileOptions *out,
    const PsbcCompileOptions *base, const PsbcLinkedStageParameters *source)
{
    *out = *base;
    if (!source->enabled) return; /* Legacy callers preserve inherited behaviour. */
    out->entrypoint = source->entrypoint;
    out->specialization_constant_count = source->specialization_constant_count;
    memcpy(out->specialization_constants, source->specialization_constants,
           sizeof(out->specialization_constants));
}

/* Fold the runtime fragment-coordinate selection the lowering may leave behind
 * to the shape this standalone compile already committed to. The bit it would
 * otherwise read comes from the PS state user SGPR, which a standalone caller
 * does not supply; the compile's own argument map decides the same question
 * through pos_fixed_pt. */
static bool
fold_use_float_frag_coord_xy(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
    if (intrin->intrinsic != nir_intrinsic_load_use_float_frag_coord_xy_amd)
        return false;
    const bool use_float = *(const bool *)data;
    b->cursor = nir_before_instr(&intrin->instr);
    if (getenv("PSBC_DEBUG_FOLD"))
        fprintf(stderr, "PSBC_FOLD intrinsic replaced use_float=%d\n", use_float ? 1 : 0);
    /* The intrinsic's destination is a 1-bit boolean (nir_intrinsics.py:
     * bit_sizes=[1]), not a 32-bit integer. */
    nir_def_replace(&intrin->def, nir_imm_bool(b, use_float));
    return true;
}

static PsbcResult psbc_compile_impl(
    const uint32_t*       spirv,
    size_t                spirv_size,
    const nir_shader*     input_nir,
    const uint32_t*       previous_spirv,
    size_t                previous_spirv_size,
    const nir_shader*     previous_input_nir,
    /* A LINK-ONLY next stage, compiled into nothing and used purely for the
     * cross-stage facts the consumer cannot know about itself. Today only the
     * hull pair supplies one: the tessellator's domain, spacing, winding and
     * point mode are declared in the EVALUATION half, and a control half
     * compiled without it has no way to learn them. */
    const uint32_t*       next_link_spirv,
    size_t                next_link_spirv_size,
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

    /* A previous stage makes this a MERGED compile: GFX9+ runs vertex+geometry
     * as one ES/GS program and vertex+control as one LS/HS program, and in both
     * cases ACO wants the pair in one call so it can emit a single image whose
     * halves dispatch on merged_wave_info.  paired_previous is "there is a
     * previous stage"; the two named forms are the stage-specific parts. */
    const bool paired_previous = previous_spirv || previous_input_nir;
    mesa_shader_stage mesa_stage = psbc_to_mesa_stage(opts->stage);
    if (mesa_stage == MESA_SHADER_NONE)
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    const bool paired_geometry =
        paired_previous && mesa_stage == MESA_SHADER_GEOMETRY;
    const bool paired_tess_geometry = paired_geometry && next_link_spirv;
    const bool paired_hull =
        paired_previous && mesa_stage == MESA_SHADER_TESS_CTRL;
    /* The DOMAIN pair is a LINK, not a merge: the evaluation half is its own
     * NGG program, but without its control half's info radv leaves
     * num_tess_patches at zero, and radv_nir_lower_abi then makes the shader
     * read the patch count, the attribute stride and tes_reads_tess_factors
     * at RUNTIME out of the tcs_offchip_layout user SGPR. Linking the pair
     * here keeps all three compile-time constants, which is both correct and
     * the only form this driver's ABI can serve. */
    const bool paired_domain =
        paired_previous && mesa_stage == MESA_SHADER_TESS_EVAL;
    /* The geometry pair is NGG; the hull pair is never NGG - LS/HS is a
     * fixed-function-fed hardware stage, and the NGG program in a tessellation
     * pipeline is the DOMAIN half, compiled separately. */
    if (paired_previous &&
        ((!paired_geometry && !paired_hull && !paired_domain) ||
         (paired_geometry && !opts->ngg) || (paired_hull && opts->ngg) ||
         opts->target != PSBC_TARGET_PS5))
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (opts->ps5_global_streamout && !paired_geometry)
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (opts->primitive_id_per_primitive &&
        (opts->target != PSBC_TARGET_PS5 || mesa_stage != MESA_SHADER_FRAGMENT))
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (previous_input_nir &&
        previous_input_nir->info.stage !=
            (paired_domain ? MESA_SHADER_TESS_CTRL :
             paired_tess_geometry ? MESA_SHADER_TESS_EVAL : MESA_SHADER_VERTEX))
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
         (mesa_stage != MESA_SHADER_VERTEX && mesa_stage != MESA_SHADER_TESS_EVAL &&
          !paired_geometry)))
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (opts->descriptor_binding_count > PSBC_MAX_DESCRIPTOR_BINDINGS)
        return PSBC_RESULT_INTERNAL_ERROR;
    if (!specialization_parameters_valid(opts->specialization_constant_count,
                                        opts->specialization_constants) ||
        (opts->previous_parameters.enabled &&
         !specialization_parameters_valid(opts->previous_parameters.specialization_constant_count,
                                          opts->previous_parameters.specialization_constants)) ||
        (opts->next_link_parameters.enabled &&
         !specialization_parameters_valid(opts->next_link_parameters.specialization_constant_count,
                                          opts->next_link_parameters.specialization_constants)))
        return PSBC_RESULT_INTERNAL_ERROR;
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
    if (opts->target == PSBC_TARGET_PS5 && opts->stage == PSBC_STAGE_COMPUTE) {
        /* Dispatch uses wave32. Keep NIR's subgroup arithmetic and ACO's
         * selected wave size in agreement even for wide subgroup intrinsics. */
        compiler_info.subgroup_size = 32;
        compiler_info.min_subgroup_size = 32;
        compiler_info.max_subgroup_size = 32;
    }
    compiler_info.key.family = chipfamily;
    compiler_info.key.load_grid_size_from_user_sgpr = (gfxlevel >= GFX10_3);
    compiler_info.key.use_ngg = opts->ngg;
    compiler_info.key.ps5_global_streamout = opts->ps5_global_streamout;
    compiler_info.key.ps5_fragment_view_index = opts->target == PSBC_TARGET_PS5;
    compiler_info.key.ps5_tess_ring_table = opts->target == PSBC_TARGET_PS5;
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
    if (paired_previous) {
        /* The consumer's own previous stage: a control half consumes the
         * vertex half, an evaluation half consumes the control half. */
        previous.stage = paired_domain ? MESA_SHADER_TESS_CTRL :
            paired_tess_geometry ? MESA_SHADER_TESS_EVAL : MESA_SHADER_VERTEX;
        /* THE switch that decides the vertex half's hardware role. RADV's
         * gather_shader_info_vs sets vs.as_ls only when next_stage is
         * TESS_CTRL and vs.as_es only when it is GEOMETRY; with the default
         * FRAGMENT the half is compiled as a legacy VS that exports to the
         * parameter cache instead of writing its outputs to LDS for the
         * consumer. */
        previous.next_stage = mesa_stage;
        PsbcCompileOptions previous_options;
        select_linked_parameters(&previous_options, opts, &opts->previous_parameters);
        previous_nir = prepare_stage_nir(
            &compiler_info, &previous, previous_spirv, previous_spirv_size,
            previous_input_nir, &previous_options
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
    /* The link-only evaluation half of a hull pair.
     *
     * The tessellator's configuration - domain, spacing, winding, point mode -
     * is declared in the EVALUATION half by GLSL convention, and the Vulkan
     * spec allows either tessellation stage to declare it. A control half
     * compiled from the vertex and control SPIR-V alone therefore has no way
     * to know the domain, and ac_nir_lower_tess_io_to_mem.c branches on
     * exactly that: nir_load_tcs_primitive_mode_amd selects the triangle
     * layout for TESS_PRIMITIVE_TRIANGLES, the isoline layout for
     * TESS_PRIMITIVE_ISOLINES, and falls through to the QUAD layout for
     * everything else - including TESS_PRIMITIVE_UNSPECIFIED, which is 0 in
     * shader_enums.h and exactly what an unlinked control half reports.
     *
     * The consequence is not subtle and was measured on hardware: the hull
     * stored outer 2.0, 2.0, 2.0 at ring words 16384-16386 and inner 1.0 at
     * word 16388, leaving 16387 zero. That gap is the quad layout's fourth
     * outer slot. The tessellator, configured by VGT_TF_PARAM.TYPE as a
     * TRIANGLE domain, reads inner[0] from the contiguous fourth dword - the
     * word the hull left zero - so the patch is tessellated with a zero inner
     * level and nothing reaches the rasteriser.
     *
     * This shader is compiled into nothing. It exists so the control half can
     * be linked to it, which is the mirror of what the domain compile already
     * does in the other direction. */
    /* The facts the link yields, extracted as plain values so the link-only
     * shader can be freed the moment it has been read. It is consumed long
     * before the compile ends and there are a dozen early-return paths
     * between here and there; carrying the nir_shader across all of them
     * leaked it (603 KB, caught by the sanitizer gate), and adding a free to
     * every exit would have been one edit away from leaking again. */
    bool next_link_valid = false;
    bool next_link_reads_tess_factors = false;
    uint64_t next_link_inputs_read = 0, next_link_patch_inputs_read = 0;
    enum tess_primitive_mode next_link_primitive_mode = TESS_PRIMITIVE_UNSPECIFIED;
    nir_shader* next_link_nir = NULL;
    if (paired_hull && next_link_spirv) {
        struct radv_shader_stage next_link = {0};
        next_link.stage = MESA_SHADER_TESS_EVAL;
        next_link.next_stage = MESA_SHADER_FRAGMENT;
        PsbcCompileOptions next_options;
        select_linked_parameters(&next_options, opts, &opts->next_link_parameters);
        next_link_nir = prepare_stage_nir(
            &compiler_info, &next_link, next_link_spirv,
            next_link_spirv_size, NULL, &next_options
        );
        if (!next_link_nir) {
            ralloc_free(previous_nir);
            ralloc_free(nir);
            psbc_shutdown();
            return PSBC_RESULT_COMPILE_NIR;
        }
        /* merge_tess_info(), the same union the domain compile performs: the
         * two halves may each declare part of the tessellator configuration
         * and the backend's view has to be both. Here the control half is the
         * one being compiled, so the merged view lands on it. */
        nir->info.tess.tcs_vertices_out |= next_link_nir->info.tess.tcs_vertices_out;
        nir->info.tess.spacing |= next_link_nir->info.tess.spacing;
        nir->info.tess._primitive_mode |= next_link_nir->info.tess._primitive_mode;
        nir->info.tess.ccw |= next_link_nir->info.tess.ccw;
        nir->info.tess.point_mode |= next_link_nir->info.tess.point_mode;
        debug_shader_io("next-link-SPIR-V", next_link_nir, NULL);
        next_link_reads_tess_factors =
            !!(next_link_nir->info.inputs_read &
               (VARYING_BIT_TESS_LEVEL_INNER | VARYING_BIT_TESS_LEVEL_OUTER));
        next_link_inputs_read = next_link_nir->info.inputs_read;
        next_link_patch_inputs_read = next_link_nir->info.patch_inputs_read;
        next_link_primitive_mode = nir->info.tess._primitive_mode;
        next_link_valid = true;
        ralloc_free(next_link_nir);
        next_link_nir = NULL;
    }
    if (paired_domain) {
        /* merge_tess_info(), from the pinned radv_pipeline_graphics.c. The
         * Vulkan spec lets the domain, spacing, winding, point mode and output
         * vertex count be declared in EITHER tessellation stage, and requires
         * only that they agree where both declare them - so the backend's view
         * has to be the union. GLSL conventionally puts the output vertex
         * count on the control half and the domain/spacing/winding on the
         * evaluation half, which means neither half alone carries the whole
         * tessellator configuration.
         *
         * Without this the evaluation half sees tcs_vertices_out == 0, and
         * radv_nir_lower_abi asserts on it while computing the attribute
         * stride as soon as the patch count is a compile-time constant. */
        nir->info.tess.tcs_vertices_out |= previous_nir->info.tess.tcs_vertices_out;
        nir->info.tess.spacing |= previous_nir->info.tess.spacing;
        nir->info.tess._primitive_mode |= previous_nir->info.tess._primitive_mode;
        nir->info.tess.ccw |= previous_nir->info.tess.ccw;
        nir->info.tess.point_mode |= previous_nir->info.tess.point_mode;
        /* and the merged view back onto the control half. */
        previous_nir->info.tess.tcs_vertices_out = nir->info.tess.tcs_vertices_out;
        previous_nir->info.tess._primitive_mode = nir->info.tess._primitive_mode;
        previous_nir->info.tess.spacing = nir->info.tess.spacing;
        previous_nir->info.tess.ccw = nir->info.tess.ccw;
        previous_nir->info.tess.point_mode = nir->info.tess.point_mode;
        if (getenv("PSBC_DEBUG_TESSMERGE"))
            fprintf(stderr, "PSBC_TESSMERGE tes_out=%u tcs_out=%u mode=%d spacing=%d ccw=%d\n",
                    nir->info.tess.tcs_vertices_out,
                    previous_nir->info.tess.tcs_vertices_out,
                    (int)nir->info.tess._primitive_mode,
                    (int)nir->info.tess.spacing, (int)nir->info.tess.ccw);
    }

    /* Shader info + args + postprocess */
    struct radv_shader_stage control_link = {0};
    if (paired_tess_geometry) {
        /* The merged executable is TES+GS. TCS is link-only, so TES keeps the
         * same offchip layout/patch-count derivation as standalone domain. */
        control_link.stage = MESA_SHADER_TESS_CTRL;
        control_link.next_stage = MESA_SHADER_TESS_EVAL;
        PsbcCompileOptions control_options;
        select_linked_parameters(&control_options,opts,&opts->next_link_parameters);
        control_link.nir=prepare_stage_nir(&compiler_info,&control_link,
            next_link_spirv,next_link_spirv_size,NULL,&control_options);
        if(!control_link.nir) {
            ralloc_free(previous_nir);ralloc_free(nir);psbc_shutdown();
            return PSBC_RESULT_COMPILE_NIR;
        }
        /* Tie the link-only allocation to the current shader on all exits. */
        ralloc_steal(nir,control_link.nir);
        previous_nir->info.tess.tcs_vertices_out |= control_link.nir->info.tess.tcs_vertices_out;
        previous_nir->info.tess.spacing |= control_link.nir->info.tess.spacing;
        previous_nir->info.tess._primitive_mode |= control_link.nir->info.tess._primitive_mode;
        previous_nir->info.tess.ccw |= control_link.nir->info.tess.ccw;
        previous_nir->info.tess.point_mode |= control_link.nir->info.tess.point_mode;
        control_link.nir->info.tess.tcs_vertices_out=previous_nir->info.tess.tcs_vertices_out;
        control_link.nir->info.tess.spacing=previous_nir->info.tess.spacing;
        control_link.nir->info.tess._primitive_mode=previous_nir->info.tess._primitive_mode;
        control_link.nir->info.tess.ccw=previous_nir->info.tess.ccw;
        control_link.nir->info.tess.point_mode=previous_nir->info.tess.point_mode;
        radv_nir_shader_info_init(control_link.stage,control_link.next_stage,&control_link.info);
        control_link.info.outputs_linked=true;
        control_link.nir->info.tess.tcs_outputs_read_by_tes &= previous_nir->info.inputs_read;
        control_link.nir->info.tess.tcs_patch_outputs_read_by_tes &= previous_nir->info.patch_inputs_read;
    }
    stage.nir = nir;
    /* radv_link_shaders_info() detects a stage by its .nir pointer. The
     * geometry pair deliberately leaves previous.nir unset and marks its
     * linked slot counts by hand below, so its published metadata does not
     * move; the hull pair needs the real VS->TCS linking, which computes
     * vs.tcs_inputs_via_lds, the LSHS workgroup size and tcs_in_out_eq from
     * the two shaders' actual IO. */
    if (paired_hull || paired_domain || paired_tess_geometry)
        previous.nir = previous_nir;
    radv_nir_shader_info_init(stage.stage, stage.next_stage, &stage.info);
    stage.info.is_ngg = opts->ngg;
    /* Linked output mapping must precede gather_shader_info_tcs: its
     * io_info.highest_remapped_vram_output defines the per-patch region
     * base. Setting this only after info gathering leaves fixed-location
     * padding on the hull while the linked domain uses compact slots. */
    if (next_link_valid) {
        stage.info.outputs_linked = true;
        /* A TCS output may exist only for cross-invocation LDS communication.
         * Standalone lowering initially marks every output TES-readable.
         * Use the actual linked consumer before gather_shader_info_tcs derives
         * highest_remapped_vram_output and the patch-region base. Keep the
         * independent outputs_read/written masks intact: they still size LDS.
         * The late tes_inputs_read override alone cannot repair this base. */
        nir->info.tess.tcs_outputs_read_by_tes &= next_link_inputs_read;
        nir->info.tess.tcs_patch_outputs_read_by_tes &= next_link_patch_inputs_read;
    }
    if (paired_previous) {
        radv_nir_shader_info_init(previous.stage, previous.next_stage,
                                  &previous.info);
        /* The LS half of a hull pair is not NGG; only the ES half of a
         * geometry pair is. */
        previous.info.is_ngg = paired_geometry;
        /* The control half of a domain pair needs its own patch-count
         * derivation to run, which is what the link then copies across. */
    }
    if (paired_domain) {
        /* Mirror radv_graphics_shaders_fill_linked_tcs_tes_io_info: the
         * paired hull uses compact TES-consumed slots, so the domain must
         * not read RADV's unlinked fixed-location layout. Include the same
         * tess-level/per-patch reservation used by the upstream linker. */
        const uint64_t levels = VARYING_BIT_TESS_LEVEL_OUTER |
                                VARYING_BIT_TESS_LEVEL_INNER;
        previous.info.outputs_linked = true;
        stage.info.tes.num_linked_inputs =
            util_bitcount64(nir->info.inputs_read & ~levels);
        stage.info.tes.num_linked_patch_inputs =
            util_bitcount64(nir->info.inputs_read & levels) +
            util_bitcount64(nir->info.patch_inputs_read);
        stage.info.inputs_linked = true;
    }
    if (paired_geometry) {
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
        if(paired_tess_geometry) {
            const uint64_t levels=VARYING_BIT_TESS_LEVEL_OUTER|VARYING_BIT_TESS_LEVEL_INNER;
            previous.info.tes.num_linked_inputs=util_bitcount64(previous_nir->info.inputs_read&~levels);
            previous.info.tes.num_linked_patch_inputs=util_bitcount64(previous_nir->info.inputs_read&levels)+
                util_bitcount64(previous_nir->info.patch_inputs_read);
            previous.info.inputs_linked=true;
            previous.info.tes.num_linked_outputs=linked_slots;
        } else previous.info.vs.num_linked_outputs = linked_slots;
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
    if(paired_tess_geometry)control_link.layout=layout;
    if (paired_previous)
        previous.layout = layout;
    struct radv_graphics_state_key gfx_state = {0};
    gfx_state.rs.provoking_vtx_last = opts->provoking_vtx_last;
    /* The tessellation workgroup layout is derived from the pipeline's patch
     * control points (the input patch size, not the control stage's output
     * vertex count), so the hull compile can size its LDS and name the patch
     * count the LS_HS launch state needs. Zero keeps the tessellation info
     * uncomputed, exactly like a pipeline that does not draw patches. */
    gfx_state.ts.patch_control_points = opts->patch_control_points;
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
    /* The pipeline's own sample-shading state. radv's graphics path fills this
     * from the pipeline and the fragment-coordinate lowering then chooses the
     * per-sample position path statically; a standalone caller has to say the
     * same thing, or the lowering falls back to a runtime selection that reads
     * the PS state user SGPR - which no standalone caller supplies, so a shader
     * that reads gl_FragCoord gets the same coordinate for every sample
     * iteration (measured: the pinned min_sample_shading leaves receive one
     * unique colour per pixel where the oracle requires one per shaded sample). */
    gfx_state.ms.sample_shading_enable = opts->sample_shading_enable;
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
    /* Standalone fragment compiles run the fragment-coordinate lowering
     * UNCONDITIONALLY, exactly as the graphics pipeline path does
     * (radv_pipeline_graphics.c: the pass is called for every fragment stage
     * with sample_shading = state || nir->info.fs.uses_sample_shading).
     *
     * It used to be skipped whenever the pipeline or the shader asked for
     * sample shading, and that skip is what this profile measured as a crash:
     * the pass is then run later, inside radv_postprocess_nir, AFTER
     * radv_nir_shader_info_pass has already decided the stage's arguments. A
     * shader that reads gl_FragCoord while declaring gl_SampleID (the pinned
     * CTS leaf min_sample_shading.min_0_0.samples_2.primitive_triangle) leaves
     * the pass with a dynamic float/integer fragment-coordinate choice, so it
     * emits load_use_float_frag_coord_xy_amd - and the ABI, built from the
     * pre-lowering info, has no ps_state argument for ABI lowering to read it
     * from. The compiler then ABORTS the process on assert(arg.used)
     * (src/amd/common/nir/ac_nir.c ac_nir_load_arg_at_offset); that abort, not
     * a driver refusal, is what killed the focused CTS payload. */
    if (mesa_stage == MESA_SHADER_FRAGMENT)
        NIR_PASS(_, nir, radv_nir_lower_opt_fs_frag_pos,
                 gfx_state.vrs_may_be_enabled,
                 gfx_state.ms.sample_shading_enable ||
                    nir->info.fs.uses_sample_shading);

    /* The pass above can leave a RUNTIME selection behind: when the fragment
     * coordinate is used as float in both components and the shader has no
     * sample position, it emits load_use_float_frag_coord_xy_amd so a driver can
     * pick between the interpolated coordinate and pixel_coord + 0.5 through a
     * bit of the PS state user SGPR.
     *
     * A standalone compile supplies no such SGPR, and needs none: the pipeline
     * it is compiling for already says whether the pixel stage runs once per
     * fragment or once per sample. Per-sample shading needs the INTERPOLATED
     * coordinate - the fixed-point one is the pixel's, never the sample's - so
     * the selection is folded to that state HERE, before the shader-info pass
     * and the argument map that follows it. Folding it afterwards (after
     * radv_declare_shader_args) left POS_FIXED_PT enabled and the shader reading
     * the pixel centre for every sample, which is exactly what the pinned
     * min_sample_shading leaves measured: one unique colour per pixel where the
     * oracle requires one per shaded sample. */
    if (mesa_stage == MESA_SHADER_FRAGMENT) {
        const bool sample_shaded = gfx_state.ms.sample_shading_enable ||
            nir->info.fs.uses_sample_shading;
        /* The lowering ran here, so the standalone postprocess must not run it
         * a second time: the second run decides from what the first left and
         * can re-emit the very selection folded below. */
        gfx_state.frag_pos_already_lowered = true;
        NIR_PASS(_, nir, nir_shader_intrinsics_pass, fold_use_float_frag_coord_xy,
                 nir_metadata_control_flow, (void *)&sample_shaded);
    }

    /* The pass above can leave a RUNTIME selection behind: when the fragment
     * coordinate is used as float in both components and the shader has no
     * sample position, it emits load_use_float_frag_coord_xy_amd so the driver
     * can pick between the interpolated coordinate and pixel_coord + 0.5 with
     * a bit of the PS state user SGPR.
     *
     * A standalone compile has no such SGPR - and does not need one, because it
     * already committed to one of the two shapes itself: its argument map
     * carries pos_fixed_pt exactly when the fixed-point position is what the
     * hardware will deliver. Fold the selection to that fact, so the branch
     * cannot disagree with what the pipeline was built for. Measured before
     * this: the pinned min_sample_shading leaves took the pixel-centre branch on
     * every sample iteration because the never-written SGPR read as zero, so
     * every sample received (0.5, 0.5) and the oracle's unique-colour count
     * could not be satisfied. */
    radv_nir_shader_info_pass(
        &compiler_info, nir, &layout, &stage.key, &gfx_state,
        RADV_PIPELINE_GRAPHICS, false, &stage.info
    );
    if (paired_previous) {
        radv_nir_shader_info_pass(
            &compiler_info, previous_nir, &layout, &previous.key, &gfx_state,
            RADV_PIPELINE_GRAPHICS, false, &previous.info
        );
    }
    if(paired_tess_geometry)
        radv_nir_shader_info_pass(&compiler_info,control_link.nir,&layout,
            &control_link.key,&gfx_state,RADV_PIPELINE_GRAPHICS,false,&control_link.info);
    if (paired_hull) {
        /* Keep the unlinked semantic mapping, but one merged LS/HS program
         * needs ONE LDS input stride. TCS sizing already uses its consumed
         * input span; LS lowering stores only tcs_inputs_via_lds. Unconsumed
         * high-location VS outputs must not enlarge just the writer's stride.
         * Do this after both info passes and before their merged link/lowering. */
        previous.info.vs.num_linked_outputs = stage.info.tcs.num_linked_inputs;
    }
    /* Legacy Gallium texture indices carry no Vulkan deref for RADV's info
     * pass to discover; that caller keeps the explicit PSBC layout. A caller
     * that asserts static descriptor use leaves RADV's NIR-derived set mask
     * exactly as the optimized shader produced it, so a layout binding the NIR
     * never dereferences cannot acquire a native descriptor dependency. */
    uint64_t static_binding_mask[PSBC_MAX_DESCRIPTOR_SETS] = {0};
    uint64_t stage_push = 0, previous_stage_push = 0;
    bool push_valid = gather_push_dwords(nir, &stage_push);
    if (paired_hull || paired_geometry) {
        bool previous_valid = gather_push_dwords(previous_nir, &previous_stage_push);
        push_valid = push_valid && previous_valid;
    }
    if (!push_valid) stage_push = previous_stage_push = 0;
    uint64_t hull_vertex_push = 0, hull_control_push = 0;
    bool hull_push_valid = false;
    if (paired_hull) {
        bool vertex_valid = gather_push_dwords(previous_nir, &hull_vertex_push);
        bool control_valid = gather_push_dwords(nir, &hull_control_push);
        hull_push_valid = vertex_valid && control_valid;
        if (!hull_push_valid) hull_vertex_push = hull_control_push = 0;
    }
    if (opts->descriptor_binding_count && !opts->static_descriptor_use) {
        uint32_t descriptor_set_mask = 0;
        for (uint32_t i = 0; i < opts->descriptor_binding_count; ++i)
            descriptor_set_mask |= 1u << opts->descriptor_bindings[i].set;
        stage.info.desc_set_used_mask |= descriptor_set_mask;
        if (paired_previous)
            previous.info.desc_set_used_mask |= descriptor_set_mask;
    } else if (opts->descriptor_binding_count) {
        /* The deref form is only visible before the descriptor lowering that
         * precedes ACO, so record the used bindings here, next to the info pass
         * that records the used sets from the same sources. */
        uint32_t used_set_mask = 0;
        gather_static_descriptor_use(nir, &used_set_mask, static_binding_mask);
        uint32_t unknown_sets = stage.info.desc_set_used_mask;
        for (unsigned set = 0; set < PSBC_MAX_DESCRIPTOR_SETS; ++set)
            if (static_binding_mask[set])
                unknown_sets &= ~(1u << set);
        /* LS/HS and ES/GS execute both input modules with one shared table.
         * Collect both BEFORE lowering removes descriptor dereferences. A
         * precise mask from one half must not hide unknown use in the other. */
        if (paired_geometry || paired_hull) {
            uint64_t previous_bindings[PSBC_MAX_DESCRIPTOR_SETS] = {0};
            uint32_t previous_sets = 0;
            gather_static_descriptor_use(previous_nir, &previous_sets, previous_bindings);
            uint32_t previous_unknown = previous.info.desc_set_used_mask;
            for (unsigned set = 0; set < PSBC_MAX_DESCRIPTOR_SETS; ++set) {
                if (previous_bindings[set])
                    previous_unknown &= ~(1u << set);
                static_binding_mask[set] |= previous_bindings[set];
            }
            unknown_sets |= previous_unknown;
        }
        for (uint32_t i = 0; i < opts->descriptor_binding_count; ++i) {
            const PsbcDescriptorBinding *declared = &opts->descriptor_bindings[i];
            uint32_t set = declared->set;
            if (set >= PSBC_MAX_DESCRIPTOR_SETS || declared->binding >= 64u ||
                !(unknown_sets & (1u << set)))
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
        if (paired_previous && previous.info.loads_push_constants) {
            previous.info.can_inline_all_push_constants = false;
            previous.info.inline_push_constant_mask = 0;
        }
    }
    debug_stage("shader-info-end");
    debug_shader_io("info", nir, &stage.info);
    if (paired_previous)
        debug_shader_io("previous-info", previous_nir, &previous.info);

    /* The info link runs for every linked or NGG shape, the legacy domain
     * included: radv_nir_shader_info_link is where the pre-raster stage's
     * export parameters are assigned, NGG or not. */
    if (opts->ngg || paired_hull || paired_domain) {
        struct radv_shader_stage stages[MESA_VULKAN_SHADER_STAGES] = {0};
        stages[mesa_stage] = stage;
        if (paired_previous)
            stages[previous.stage] = previous;
        if(paired_tess_geometry)stages[MESA_SHADER_TESS_CTRL]=control_link;
        radv_nir_shader_info_link(&compiler_info, &gfx_state, stages);
        stage.info = stages[mesa_stage].info;
        if (paired_previous)
            previous.info = stages[previous.stage].info;
        /* The TCS<->TES half of the link, which radv performs in
         * radv_link_shaders_info() when both stages are present and which is
         * skipped here because the evaluation half is not one of the compiled
         * stages. Every value is a pure function of the evaluation half's
         * NIR, so the link-only shader is enough and no second info pass is
         * needed:
         *
         *   tcs.tes_reads_tess_factors  <- gather_shader_info_tes()'s own
         *       definition, tes.reads_tess_factors, which is exactly this
         *       test on inputs_read.
         *   tcs.tes_inputs_read, tcs.tes_patch_inputs_read <- verbatim; the
         *       control half otherwise assumes ~0ULL, that is, that the
         *       evaluation half reads EVERYTHING, which oversizes the
         *       off-chip patch.
         *   tes._primitive_mode <- verbatim, and this is the one that decides
         *       the tessellation-factor layout.
         *   outputs_linked <- radv_graphics_shaders_fill_linked_tcs_tes_io_info(),
         *       which sets it on the control half; it is what makes
         *       radv_nir_lower_abi lower the primitive mode and the
         *       tess-levels-to-TES flag to CONSTANTS instead of to fields of a
         *       tcs_offchip_layout user SGPR that nothing on this platform
         *       supplies. */
        /* The passthrough override, applied after the link so nothing
         * recomputes it, and before ACO reads it. radv_shader.c passes
         * info->is_ngg_passthrough straight to aco as options.passthrough,
         * and this compile publishes the same flag into
         * VGT_SHADER_STAGES_EN.PRIMGEN_PASSTHRU_EN, so one assignment keeps
         * the program and the register describing the same pipeline. */
        if (opts->ngg_no_passthrough)
            stage.info.is_ngg_passthrough = false;
        if (next_link_valid) {
            stage.info.tcs.tes_reads_tess_factors = next_link_reads_tess_factors;
            stage.info.tcs.tes_inputs_read = next_link_inputs_read;
            stage.info.tcs.tes_patch_inputs_read = next_link_patch_inputs_read;
            stage.info.tes._primitive_mode = next_link_primitive_mode;
            stage.info.outputs_linked = true;
        }
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
        if (paired_previous)
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
    case MESA_SHADER_GEOMETRY: previous_stage = paired_tess_geometry?
        MESA_SHADER_TESS_EVAL:MESA_SHADER_VERTEX; break;
    default:                    previous_stage = MESA_SHADER_NONE;      break;
    }

    /* PSBC emits LS and HS in one ACO program. The generic unlinked IO
     * layout does not make them separately executable shader objects: that
     * ABI reserves prolog/epilog PCs and descriptor pointers never supplied
     * by this native pipeline. Keep IO mapping unchanged, but select the
     * merged argument ABI for the program we actually compile. */
    if (paired_hull) {
        stage.info.merged_shader_compiled_separately = false;
        previous.info.merged_shader_compiled_separately = false;
    }
    radv_declare_shader_args(
        &compiler_info, &gfx_state, &stage, previous_stage, NULL
    );
    debug_stage("declare-args-end");

    /* Temporary T04 ABI trace: which SGPR each declared argument landed in, and
     * which user-data slots the stage publishes. Off unless asked for. */
    if (getenv("PSBC_DEBUG_ARGS")) {
        const struct ac_shader_args* a = &stage.args.ac;
        fprintf(stderr, "PSBC_ARGS stage=%d num_user_sgprs=%u num_sgprs_used=%u arg_count=%u\n",
                (int)mesa_stage, stage.args.num_user_sgprs, a->num_sgprs_used, a->arg_count);
        const struct { const char* name; const struct ac_arg* arg; } named[] = {
            { "ring_offsets", &a->ring_offsets },
            { "gs_tg_info", &a->gs_tg_info },
            { "merged_wave_info", &a->merged_wave_info },
            { "tess_offchip_offset", &a->tess_offchip_offset },
            { "scratch_offset", &a->scratch_offset },
            { "ngg_lds_layout", &stage.args.ngg_lds_layout },
            { "ngg_state", &stage.args.ngg_state },
            { "base_vertex", &a->base_vertex },
            { "start_instance", &a->start_instance },
            { "draw_id", &a->draw_id },
            { "view_index", &a->view_index },
            { "vertex_buffers", &a->vertex_buffers },
            { "push_constants", &a->push_constants },
        };
        for (unsigned i = 0; i < sizeof(named) / sizeof(named[0]); ++i) {
            const struct ac_arg* arg = named[i].arg;
            if (!arg->used) continue;
            const __typeof__(a->args[0])* def = &a->args[arg->arg_index];
            fprintf(stderr, "  %-20s arg=%u file=%u offset=%u size=%u type=%u skip=%u\n",
                    named[i].name, arg->arg_index, (unsigned)def->file, def->offset,
                    def->size, (unsigned)def->type, def->skip ? 1u : 0u);
        }
        for (unsigned i = 0; i < AC_UD_MAX_UD; ++i) {
            const struct radv_userdata_info* info = &stage.args.user_sgprs_locs.shader_data[i];
            if (info->sgpr_idx < 0) continue;
            fprintf(stderr, "  ud[%u] sgpr=%d count=%u\n", i, (int)info->sgpr_idx,
                    (unsigned)info->num_sgprs);
        }
    }

    stage.info.user_sgprs_locs = stage.args.user_sgprs_locs;
    stage.info.inline_push_constant_mask = stage.args.ac.inline_push_const_mask;
    if (paired_geometry || paired_hull) {
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

    if (paired_geometry || paired_hull)
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
    /* A merge hands ACO both halves; the domain LINK compiles only the
     * evaluation half - its control half was prepared for the info link and
     * is not part of this program. */
    if (paired_geometry || paired_hull) {
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
        .es_info = paired_geometry ? &previous.info : NULL,
        .ls_merged = paired_hull,
        .domain_linked = paired_domain,
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
        .hull_push_use_valid = hull_push_valid,
        .push_use_valid = push_valid,
        .stage_push_dwords = stage_push,
        .previous_stage_push_dwords = previous_stage_push,
        .hull_vertex_push_dwords = hull_vertex_push,
        .hull_control_push_dwords = hull_control_push,
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
    return psbc_compile_impl(spirv, spirv_size, NULL, NULL, 0, NULL, NULL, 0,
                             opts, out);
}

PsbcResult psbc_compile_nir(
    const struct nir_shader* nir,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
) {
    return psbc_compile_impl(NULL, 0, nir, NULL, 0, NULL, NULL, 0, opts, out);
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
                             NULL, 0, opts, out);
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
    const uint32_t* tess_eval_spirv,
    size_t tess_eval_spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
) {
    if (!out || !opts || !vertex_spirv || !tess_ctrl_spirv || !tess_eval_spirv)
        return PSBC_RESULT_INVALID_SPIRV;
    if (opts->stage != PSBC_STAGE_TESS_CTRL || opts->target != PSBC_TARGET_PS5)
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    /* The merged workgroup layout - patches per workgroup, its LDS size and
     * its thread count - is computed from the pipeline's INPUT patch size, so
     * without it the pair cannot be launched and must not be published as if
     * it could.  radv_link_shaders_info() skips the whole LSHS workgroup
     * derivation when patch_control_points is zero. */
    if (!opts->patch_control_points || opts->patch_control_points > 32)
        return PSBC_RESULT_UNSUPPORTED_STAGE;

    /* ONE merged LS/HS program, exactly like the geometry pair one line up.
     *
     * This used to compile the two halves as INDEPENDENT programs - the vertex
     * half through psbc_compile_shader() as a plain PSBC_STAGE_VERTEX with NGG
     * off - and then memcpy them back to back.  That cannot work on GFX9+.
     * RADV's gather_shader_info_vs() sets vs.as_ls only when next_stage is
     * MESA_SHADER_TESS_CTRL, so a half compiled standalone got next_stage =
     * FRAGMENT and came out as a legacy VS: it exported position and
     * parameters to the parameter cache, never wrote its outputs to LDS for
     * the control half, and carried the VS argument layout instead of the LS
     * one.  Two such images concatenated have no merged entry dispatching on
     * merged_wave_info either, while the hardware runs LS and HS as ONE stage
     * from ONE program counter - so every patch draw faulted whatever the
     * tessellation levels were.
     *
     * Routing through psbc_compile_impl() with the vertex SPIR-V as the
     * previous stage hands both shaders to ACO in a single call, which is the
     * same path the ES/GS pair already used and which aco_compile_shader()
     * supports directly through its shader_count argument. */
    return psbc_compile_impl(tess_ctrl_spirv, tess_ctrl_spirv_size, NULL,
                             vertex_spirv, vertex_spirv_size, NULL,
                             tess_eval_spirv, tess_eval_spirv_size,
                             opts, out);
}

PsbcResult psbc_compile_domain_pipeline(
    const uint32_t* tess_ctrl_spirv,
    size_t tess_ctrl_spirv_size,
    const uint32_t* tess_eval_spirv,
    size_t tess_eval_spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
) {
    if (!out || !opts || !tess_ctrl_spirv || !tess_eval_spirv)
        return PSBC_RESULT_INVALID_SPIRV;
    if (opts->stage != PSBC_STAGE_TESS_EVAL || opts->target != PSBC_TARGET_PS5)
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    if (!opts->patch_control_points || opts->patch_control_points > 32)
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    /* Link-only: the control half is prepared for radv_link_shaders_info so
     * the evaluation half inherits num_tess_patches (and the tess-factor and
     * input-read facts) as compile-time constants. Only the evaluation half
     * is compiled into the returned program. */
    return psbc_compile_impl(tess_eval_spirv, tess_eval_spirv_size, NULL,
                             tess_ctrl_spirv, tess_ctrl_spirv_size, NULL,
                             NULL, 0, opts, out);
}

PsbcResult psbc_compile_tess_geometry_pipeline(
    const uint32_t *control, size_t control_size,
    const uint32_t *evaluation, size_t evaluation_size,
    const uint32_t *geometry, size_t geometry_size,
    const PsbcCompileOptions *opts, PsbcShaderOutput *out)
{
    if(out)memset(out,0,sizeof(*out));
    if(!out || !opts || !control || !evaluation || !geometry)
        return PSBC_RESULT_INVALID_SPIRV;
    if(opts->target!=PSBC_TARGET_PS5 || opts->stage!=PSBC_STAGE_GEOMETRY ||
       !opts->ngg || !opts->patch_control_points || opts->patch_control_points>32)
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    const PsbcResult validation=validate_spirv_capabilities(control,control_size,opts);
    if(validation!=PSBC_RESULT_OK)return validation;
    const PsbcResult result=psbc_compile_impl(geometry,geometry_size,NULL,evaluation,evaluation_size,
        NULL,control,control_size,opts,out);
    /* A compiled TES-fed GS must carry its distinct source and ring ABI.
     * Native execution/feature promotion remains the driver's responsibility. */
    if(result==PSBC_RESULT_OK && (!out->metadata.merged_geometry ||
       out->metadata.merged_es_source_stage!=PSBC_STAGE_TESS_EVAL ||
       !out->metadata.ps5_ring_table_valid)) {
        psbc_free_output(out);
        return PSBC_RESULT_INTERNAL_ERROR;
    }
    return result;
}

PsbcResult psbc_compile_nir_geometry_pipeline(
    const struct nir_shader* vertex_nir,
    const struct nir_shader* geometry_nir,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
) {
    return psbc_compile_impl(NULL, 0, geometry_nir, NULL, 0, vertex_nir,
                             NULL, 0, opts, out);
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
