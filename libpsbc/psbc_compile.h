/*
 * libpsbc — SPIR-V to PS4/PS5 Shader Binary Compiler Library
 *
 * Extracted from cmd/psbc/main.c to provide a reusable C API for
 * compiling SPIR-V shaders into GCN shader binaries with GnmShaderFileHeader.
 *
 * Used by:
 *   - opengnm-psbc CLI (cmd/psbc/main.c)
 *   - vulkan-ps4 ICD (vkCreateShaderModule)
 */

#ifndef PSBC_COMPILE_H
#define PSBC_COMPILE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 17: the merged pair's system-SGPR indices and launch counts, the driver
 * user-data window base, and the pixel stage's distance reads. A consumer that
 * cached a program against version 16 must not interpret those fields with the
 * new layout, so the version and the driver's cache key move together. */
/* 21: merged ES source identity distinguishes vertex-fed GS from TES-fed GS. */
#define PSBC_SHADER_METADATA_VERSION 21u

struct nir_shader;
struct nir_shader_compiler_options;

/* === Types === */

typedef enum {
    PSBC_TARGET_PS4_BASE = 0,   /* GFX7 (Sea Islands) */
    PSBC_TARGET_PS4_NEO  = 1,   /* GFX8 (Polaris) */
    PSBC_TARGET_PS5      = 2,   /* GFX10.3 (RDNA2) */
} PsbcTarget;

typedef enum {
    PSBC_STAGE_NONE       = 0,
    PSBC_STAGE_VERTEX     = 1,
    PSBC_STAGE_TESS_CTRL  = 2,
    PSBC_STAGE_TESS_EVAL  = 3,
    PSBC_STAGE_GEOMETRY   = 4,
    PSBC_STAGE_FRAGMENT   = 5,
    PSBC_STAGE_COMPUTE    = 6,
    PSBC_STAGE_TASK       = 7,
    PSBC_STAGE_EXPORT     = 8,  /* VS as ES (Export Shader) */
    PSBC_STAGE_LOCAL      = 9,  /* VS as LS (Local Shader) */
} PsbcStage;

typedef enum {
    PSBC_RESULT_OK = 0,
    PSBC_RESULT_INVALID_SPIRV,
    PSBC_RESULT_UNSUPPORTED_STAGE,
    PSBC_RESULT_COMPILE_NIR,
    PSBC_RESULT_COMPILE_ACO,
    PSBC_RESULT_OUT_OF_MEMORY,
    PSBC_RESULT_INTERNAL_ERROR,
    PSBC_RESULT_UNSUPPORTED_CAPABILITY,
} PsbcResult;

#define PSBC_MAX_CONTEXT_REGISTERS 16
#define PSBC_MAX_SHADER_REGISTERS 8
#define PSBC_MAX_SEMANTICS 32
/* Private matching keys shared by our AGC producer/consumer packages, after
 * the generic keys 15..46. This is not a PSSL system-semantic enum: the two
 * halves of a pipeline pair on these values, and the producer's word also
 * carries the parameter index of what it exports in bits 8..15. */
#define PSBC_SEMANTIC_PRIMITIVE_ID 47u
/* One key per packed clip/cull distance register the pre-raster stage exports
 * and the pixel stage reads: clip components come first, cull continues after
 * them, four components per register, so a stage pairs the (whole) register
 * rather than a feature - a register can hold clip and cull components at the
 * same time. The low byte identifies the register; the producer adds its
 * parameter index above it. */
#define PSBC_SEMANTIC_DISTANCE_REGISTER 48u
/* The geometry stage's viewport selection. RADV counts it as a parameter
 * export and gives it a slot in vs_output_param_offset, so a pipeline that
 * writes gl_ViewportIndex exports one parameter more than its varyings and
 * packed distance registers account for; naming it keeps the semantic list and
 * the export count in agreement instead of leaving the linkage unresolved. */
#define PSBC_SEMANTIC_VIEWPORT_INDEX 49u
#define PSBC_MAX_VERTEX_ATTRIBUTES 32
#define PSBC_MAX_DESCRIPTOR_BINDINGS 64
#define PSBC_MAX_DESCRIPTOR_SETS 4
#define PSBC_MAX_SPECIALIZATION_CONSTANTS 64
#define PSBC_MAX_SPECIALIZATION_BYTES 8
/* Reserve distinct 16-sampler banks for merged Gallium vertex/geometry stages. */
#define PSBC_GALLIUM_UBO_BINDING_BASE 32

typedef enum {
    PSBC_VERTEX_FORMAT_NONE = 0,
    PSBC_VERTEX_FORMAT_R32_FLOAT,
    PSBC_VERTEX_FORMAT_R32G32_FLOAT,
    PSBC_VERTEX_FORMAT_R32G32B32_FLOAT,
    PSBC_VERTEX_FORMAT_R32G32B32A32_FLOAT,
    PSBC_VERTEX_FORMAT_B8G8R8A8_UNORM,
    PSBC_VERTEX_FORMAT_R10G10B10A2_UNORM,
    PSBC_VERTEX_FORMAT_B10G10R10A2_UNORM,
    PSBC_VERTEX_FORMAT_R10G10B10A2_SNORM,
    PSBC_VERTEX_FORMAT_B10G10R10A2_SNORM,
    PSBC_VERTEX_FORMAT_R10G10B10A2_USCALED,
    PSBC_VERTEX_FORMAT_B10G10R10A2_USCALED,
    PSBC_VERTEX_FORMAT_R10G10B10A2_SSCALED,
    PSBC_VERTEX_FORMAT_B10G10R10A2_SSCALED,
    PSBC_VERTEX_FORMAT_R32_SINT,
    PSBC_VERTEX_FORMAT_R32G32_SINT,
    PSBC_VERTEX_FORMAT_R32G32B32_SINT,
    PSBC_VERTEX_FORMAT_R32G32B32A32_SINT,
    PSBC_VERTEX_FORMAT_R32_UINT,
    PSBC_VERTEX_FORMAT_R32G32_UINT,
    PSBC_VERTEX_FORMAT_R32G32B32_UINT,
    PSBC_VERTEX_FORMAT_R32G32B32A32_UINT,
    PSBC_VERTEX_FORMAT_R8G8B8A8_UNORM,
    PSBC_VERTEX_FORMAT_R8_UNORM,
    PSBC_VERTEX_FORMAT_R8_SNORM,
    PSBC_VERTEX_FORMAT_R8_UINT,
    PSBC_VERTEX_FORMAT_R8_SINT,
    PSBC_VERTEX_FORMAT_R8G8_UNORM,
    PSBC_VERTEX_FORMAT_R8G8_SNORM,
    PSBC_VERTEX_FORMAT_R8G8_UINT,
    PSBC_VERTEX_FORMAT_R8G8_SINT,
    PSBC_VERTEX_FORMAT_R8G8B8A8_SNORM,
    PSBC_VERTEX_FORMAT_R8G8B8A8_UINT,
    PSBC_VERTEX_FORMAT_R8G8B8A8_SINT,
    PSBC_VERTEX_FORMAT_R16_UNORM,
    PSBC_VERTEX_FORMAT_R16_SNORM,
    PSBC_VERTEX_FORMAT_R16_UINT,
    PSBC_VERTEX_FORMAT_R16_SINT,
    PSBC_VERTEX_FORMAT_R16_FLOAT,
    PSBC_VERTEX_FORMAT_R16G16_UNORM,
    PSBC_VERTEX_FORMAT_R16G16_SNORM,
    PSBC_VERTEX_FORMAT_R16G16_UINT,
    PSBC_VERTEX_FORMAT_R16G16_SINT,
    PSBC_VERTEX_FORMAT_R16G16_FLOAT,
    PSBC_VERTEX_FORMAT_R16G16B16A16_UNORM,
    PSBC_VERTEX_FORMAT_R16G16B16A16_SNORM,
    PSBC_VERTEX_FORMAT_R16G16B16A16_UINT,
    PSBC_VERTEX_FORMAT_R16G16B16A16_SINT,
    PSBC_VERTEX_FORMAT_R16G16B16A16_FLOAT,
} PsbcVertexFormat;

typedef struct {
    uint8_t          location;
    uint8_t          binding;
    PsbcVertexFormat format;
    uint32_t         offset;
    uint32_t         stride; /* PS5: zero repeats a current/constant attribute. */
    uint32_t         alignment;
    uint32_t         instance_divisor;
} PsbcVertexAttribute;

typedef enum {
    PSBC_DESCRIPTOR_NONE = 0,
    PSBC_DESCRIPTOR_UNIFORM_BUFFER,
    PSBC_DESCRIPTOR_UNIFORM_TEXEL_BUFFER,
    PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER,
    PSBC_DESCRIPTOR_STORAGE_BUFFER,
    /* Resource-only image data read by a fragment shader through
     * subpassLoad: eight DWORDs, exactly like a sampled T# but with no
     * sampler words, so it is never conflated with a combined T#/S# pair. */
    PSBC_DESCRIPTOR_INPUT_ATTACHMENT,
    /* Eight-DWORD resource-only UAV record; no sampler words. */
    PSBC_DESCRIPTOR_STORAGE_IMAGE,
    /* Separate descriptors, combined in the shader with OpSampledImage (the
     * form DXVK's DXBC translation emits for every D3D sample): a
     * four-DWORD S# record and an eight-DWORD sampled T# record. */
    PSBC_DESCRIPTOR_SAMPLER,
    PSBC_DESCRIPTOR_SAMPLED_IMAGE,
    /* Four-DWORD typed buffer V# written or read through imageStore /
     * imageLoad on a Buffer-dimension storage image. */
    PSBC_DESCRIPTOR_STORAGE_TEXEL_BUFFER,
} PsbcDescriptorType;

typedef struct {
    uint8_t            set;
    uint8_t            binding;
    PsbcDescriptorType type;
    uint32_t           array_size;
    uint32_t           offset;
    uint32_t           stride;
} PsbcDescriptorBinding;

/* Vulkan specialization-map scalar. Composite constants are specialized by
 * their scalar constituent IDs; Vulkan map entries therefore fit in 8 bytes. */
typedef struct {
    uint32_t constant_id;
    uint32_t size;
    uint8_t  data[PSBC_MAX_SPECIALIZATION_BYTES];
} PsbcSpecializationConstant;

/* Source-specific parameters for linked stages. An enabled empty map means
 * defaults, not inheritance from the stage being compiled. */
typedef struct {
    bool enabled;
    const char *entrypoint;
    uint32_t specialization_constant_count;
    PsbcSpecializationConstant specialization_constants[PSBC_MAX_SPECIALIZATION_CONSTANTS];
} PsbcLinkedStageParameters;

typedef enum {
    PSBC_HW_STAGE_UNKNOWN = 0,
    PSBC_HW_STAGE_VERTEX  = 1,
    PSBC_HW_STAGE_PIXEL   = 2,
    PSBC_HW_STAGE_NGG     = 3,
    /* The merged LS/HS pair: one program the hardware launches from the LS
     * block with the resource pair at the HS block (GFX10). */
    PSBC_HW_STAGE_HULL    = 4,
} PsbcHardwareStage;

typedef enum {
    PSBC_UNRESOLVED_NONE                   = 0,
    PSBC_UNRESOLVED_PROGRAM_CHECKSUM       = 1u << 0,
    PSBC_UNRESOLVED_NGG_ESGS_RING_ITEMSIZE = 1u << 1,
    PSBC_UNRESOLVED_AGC_LINKAGE             = 1u << 2,
    /* The stand-alone tessellation stages compile to ISA, but this compiler has
     * no tessellation *pipeline* entry point: radv builds the hull shader as one
     * merged vertex+tessellation-control program (AC_HW_HULL_SHADER, programmed
     * through SPI_SHADER_PGM_LO_LS on gfx10) and the domain half as a separate
     * stage, and it programs the hull/domain state (VGT_LS_HS_CONFIG,
     * VGT_TF_PARAM, VGT_TF_RING_SIZE, VGT_HS_OFFCHIP_PARAM) through the
     * pipeline's context rolls. A TESS_CTRL/TESS_EVAL result from
     * psbc_compile_shader() is therefore an ISA-level diagnostic and must never
     * be treated as a loadable package. */
    PSBC_UNRESOLVED_TESS_PIPELINE          = 1u << 3,
} PsbcUnresolvedField;

typedef struct {
    uint16_t offset;
    uint16_t padding;
    uint32_t value;
} PsbcRegisterWrite;

typedef struct {
    uint32_t             version;
    PsbcTarget           target;
    PsbcStage            source_stage;
    PsbcHardwareStage    hardware_stage;
    uint32_t             unresolved_fields;
    uint32_t             context_register_count;
    PsbcRegisterWrite    context_registers[PSBC_MAX_CONTEXT_REGISTERS];
    uint32_t             shader_register_count;
    PsbcRegisterWrite    shader_registers[PSBC_MAX_SHADER_REGISTERS];
    bool                 linkage_valid;
    PsbcRegisterWrite    linkage_ge_cntl;
    PsbcRegisterWrite    linkage_stages_en;
    PsbcRegisterWrite    linkage_user_vgpr_en;
    /* Merged vertex+geometry pre-raster stage.  The merged program occupies
     * the same NGG hardware slot as a vertex-only one and the AGC linked block
     * carries no geometry variant, so nothing downstream can tell the two
     * apart from the rest of this metadata.  When merged_geometry is true the
     * es_* fields describe the ES half the caller has to launch with it, in
     * the same units radv uses: merged_es_itemsize in bytes (the ES half's
     * export item size) and merged_esgs_ring_itemsize in dwords (the value the
     * pair programs into VGT_ESGS_RING_ITEMSIZE).  Both are zero/false for
     * every other compile.  The merged program's argument declaration is
     * shared by both halves and is already reported through the user-data
     * slots and the register set, and the ES half's parameter exports follow
     * from the item size, so they are not duplicated here. */
    bool                 merged_geometry;
    PsbcStage            merged_es_source_stage;
    uint32_t             merged_es_itemsize;
    uint32_t             merged_esgs_ring_itemsize;
    /* GE PC-line allocation (UC R_030980).  radv programs it for every NGG
     * pipeline from ac_compute_late_alloc(), which is a property of the
     * running device (SA/CU topology and per-shader-engine PC-line budget),
     * not of the shader.  The register write is emitted only when the caller
     * supplied those facts in PsbcCompileOptions; otherwise the field stays
     * invalid rather than carrying a guessed constant. */
    bool                 linkage_ge_pc_alloc_valid;
    PsbcRegisterWrite    linkage_ge_pc_alloc;
    /* Hull halves.  A GFX10 hull stage runs two programs: the LS half (the
     * vertex shader, R_00B520..) and the HS half (the tessellation-control
     * shader, R_00B420..).  psbc_compile_tess_pipeline() compiles both and
     * publishes the LS half here, with the HS half's RSRC1/RSRC2 replaced by
     * the combined pair radv_shader_combine_cfg_vs_tcs() produces.
     * hull_ls_valid stays false for a stand-alone control shader. */
    bool                 hull_ls_valid;
    uint32_t             hull_ls_code_size;
    /* Byte offset of the LS program inside PsbcShaderOutput::machine_code.  The
     * HS program starts at offset 0, so a consumer that only knows the HS half
     * still reads the same bytes it always did. */
    uint32_t             hull_ls_code_offset;
    PsbcRegisterWrite    hull_ls_pgm_lo;
    PsbcRegisterWrite    hull_ls_pgm_hi;
    PsbcRegisterWrite    hull_ls_rsrc1;
    PsbcRegisterWrite    hull_ls_rsrc2;
    /* Tessellation workgroup layout, computed when the caller supplied the
     * pipeline's patch control points in PsbcCompileOptions: the patch count
     * per LS/HS workgroup (VGT_LS_HS_CONFIG NUM_PATCHES), the LDS bytes that
     * workgroup needs and the workgroup size the launch derives from. The
     * fields are the compiler's own tessellation workgroup computation, not a
     * driver guess, and stay false for a compile that was not given the patch
     * state. */
    bool                 hull_tess_wg_valid;
    uint32_t             hull_num_patches_per_wg;
    uint32_t             hull_tcs_lds_size;
    uint32_t             hull_workgroup_size;
    /* Fragment-stage distance usage.  The masks above describe what a pre-raster
     * stage exports; these two describe what a pixel stage declares and reads,
     * which is the input the rasterizer would have to deliver to it.  Both are
     * zero for a fragment stage that reads neither, and for every other stage.
     * A consumer that cannot route distances to the pixel stage can use a
     * non-zero value here to refuse precisely instead of guessing. */
    uint32_t             ps_clip_distance_reads;
    uint32_t             ps_cull_distance_reads;
    /* Merged pre-raster pair: the system SGPRs the caller must supply and the
     * launch shape they describe.  A merged vertex+geometry stage is two
     * programs in one, and each half disables the lanes it does not need from
     * these two registers (radv declares them as the merged stage's first
     * system registers and the compiler reads them to gate the geometry half).
     * They are counts, not addresses: merged_wave_info holds the ES lane count
     * in byte 0 and the GS lane count in byte 1, and gs_tg_info holds the
     * vertices of the group in bits 12..20 and its primitives in bits 22..30.
     * The counts are the ones this metadata already programs into
     * VGT_GS_ONCHIP_CNTL.  A caller that leaves the registers stale changes how
     * many lanes each half runs, so esgs_system_sgprs_valid is the flag to
     * check before executing the program.
     *
     * The two indices are SGPR offsets in the stage's SYSTEM block, i.e. below
     * user_data_window_base: measured on three compiled programs (a merged
     * vertex+geometry pair, a clip/cull vertex program and a draw-parameter
     * program whose base_vertex/BaseInstance semantics are hardware-verified),
     * an argument's SGPR offset is its user-data dword plus that base.  A
     * driver writes its user data into the window and cannot address the system
     * block, so these two fields describe registers the linked GE state must
     * supply; a consumer that treats them as user-data dwords would write eight
     * SGPRs away from the registers the shader reads. */
    bool                 esgs_system_sgprs_valid;
    uint32_t             esgs_gs_tg_info_sgpr;
    uint32_t             esgs_merged_wave_info_sgpr;
    uint32_t             esgs_es_verts_per_subgroup;
    uint32_t             esgs_gs_inst_prims_per_subgroup;
    uint32_t             esgs_prim_amp_factor;
    uint32_t             esgs_workgroup_size;
    /* SGPR index the first driver-supplied user-data dword lands on.  The
     * driver writes user_sgpr_count dwords into the window that starts here;
     * everything below it is a system register the linked GE state supplies.
     * Zero means the stage declares no such window, and a consumer must not
     * assume a base it was not given. */
    uint32_t             user_data_window_base;
    uint32_t             input_semantic_count;
    uint32_t             input_semantics[PSBC_MAX_SEMANTICS];
    uint32_t             output_semantic_count;
    uint32_t             output_semantics[PSBC_MAX_SEMANTICS];
    uint32_t             clip_distance_mask;
    uint32_t             cull_distance_mask;
    uint32_t             address32_hi;
    uint32_t             user_sgpr_count;
    bool                 vertex_buffer_table_valid;
    uint32_t             vertex_buffer_table_user_data_dword;
    bool                 descriptor_set0_valid;
    uint32_t             descriptor_set0_user_data_dword;
    /* Direct 32-bit descriptor-table pointers used by the PS5 RADV ABI.  The
     * set-0 fields above remain source-level aliases for existing consumers;
     * metadata version 9 and later identify the enlarged binary structure. */
    bool                 descriptor_set_valid[PSBC_MAX_DESCRIPTOR_SETS];
    uint32_t             descriptor_set_user_data_dword[PSBC_MAX_DESCRIPTOR_SETS];
    bool                 push_constants_valid;
    uint32_t             push_constants_user_data_dword;
    uint32_t             push_constant_size;
    /* Version19: reads of the source stage and genuinely merged previous VS.
     * Link-only TCS input of TES is excluded. Unknown never means unused. */
    bool                 push_use_valid;
    uint64_t             stage_push_dwords;
    uint64_t             previous_stage_push_dwords;
    /* Version18: bounded256-byte push reads before merged LS/HS linkage.
     * BitN is DWORD N; dynamic offsets conservatively include their NIR range.
     * Invalid means unknown, never permission to treat a half as unused. */
    bool                 hull_push_use_valid;
    uint64_t             hull_vertex_push_dwords;
    uint64_t             hull_control_push_dwords;
    uint32_t             descriptor_binding_count;
    PsbcDescriptorBinding descriptor_bindings[PSBC_MAX_DESCRIPTOR_BINDINGS];
    /* Bindings this stage statically uses, per descriptor set, derived from the
     * optimized NIR that ACO consumes. Bit B of entry S means (set S, binding B)
     * is dereferenced by the compiled stage; a set the NIR uses whose binding
     * cannot be identified falls back to every layout binding of that set so a
     * consumer cannot treat an unknown entry as unused. All entries are zero
     * unless the caller asked for static descriptor use. */
    uint64_t             descriptor_used_binding_mask[PSBC_MAX_DESCRIPTOR_SETS];
    bool                 base_vertex_valid;
    uint32_t             base_vertex_user_data_dword;
    /* DrawIndex, the Vulkan DrawIndex built-in (gl_DrawID). RADV declares
     * ac.draw_id in the same ABI user-data dword block as the base vertex
     * (AC_UD_VS_BASE_VERTEX_START_INSTANCE), so the slot reported here is an
     * offset inside that block exactly like start_instance_user_data_dword.
     * It is valid only when the compiled stage really reads the built-in; a
     * caller that sees draw_id_valid false has no DrawIndex value to deliver
     * and must not invent one. Version 13 added these two fields. */
    bool                 draw_id_valid;
    uint32_t             draw_id_user_data_dword;
    /* ViewIndex, the multiview built-in (gl_ViewIndex). RADV declares it as its
     * own user-data location rather than a dword inside the base-vertex block,
     * so the slot reported here is that location's SGPR index in the user-SGPR
     * block - the same quantity base_vertex_user_data_dword carries - and no
     * offset arithmetic against another argument is involved. It is valid only
     * when the compiled stage really reads the built-in and only for the stages
     * this profile delivers a view index to; a caller that sees
     * view_index_valid false has no ViewIndex value to deliver and must not
     * invent one. On PS5, vertex and fragment stages can declare this slot;
     * the caller must fill each stage's independently assigned location for
     * every replayed view. Version 14 added these two fields. */
    bool                 view_index_valid;
    uint32_t             view_index_user_data_dword;
    bool                 start_instance_valid;
    uint32_t             start_instance_user_data_dword;
    bool                 streamout_valid;
    uint32_t             streamout_buffer_table_user_data_dword;
    uint32_t             streamout_enabled_stream_buffers_mask;
    uint32_t             streamout_strides_dwords[4];
    uint32_t             streamout_config_sgpr;
    uint32_t             streamout_write_index_sgpr;
    uint32_t             streamout_offset_sgprs[4];
    bool                 scratch_valid;
    uint32_t             scratch_bytes_per_wave;
    uint32_t             scratch_size_per_thread;
    uint32_t             scratch_buffer_table_user_data_dword;
    /* PS5 tessellation only: where the driver must write the 64-bit address
     * of the ring descriptor table, as a user-data dword index. The
     * tessellation stages dereference that table (entry 5 is the tess-factor
     * ring, entry 6 the off-chip ring); on this platform the system-block
     * ring_offsets at s0/s1 cannot be written, so it travels as user data. */
    bool                 ps5_ring_table_valid;
    uint32_t             ps5_ring_table_user_data_dword;

    bool                 ngg_lds_layout_valid;
    uint32_t             ngg_lds_layout_user_data_dword;
    uint32_t             ngg_lds_layout; /* GS output base in bytes, after ES inputs. */
    /* Optimized RADV vertex-buffer SRD usage. Descriptors are densely packed
     * in increasing set-bit order. The mask indexes bindings unless the
     * per-attribute flag is true, in which case it indexes attribute locations.
     * Both fields are zero when vertex_buffer_table_valid is false. */
    uint32_t             vertex_buffer_usage_mask;
    bool                 vertex_buffer_per_attribute;
} PsbcShaderMetadata;

typedef struct {
    void*               data;         /* Legacy PSSL/GNM wrapper */
    size_t              size;
    void*               machine_code; /* Raw ACO machine code; no wrapper/trailer */
    size_t              machine_code_size;
    PsbcShaderMetadata  metadata;
} PsbcShaderOutput;

typedef struct {
    PsbcTarget  target;
    PsbcStage   stage;
    const char* entrypoint;  /* Default: "main" */
    bool        optimise;    /* Default: true */
    bool        ngg;         /* Experimental PS5 VS->FS NGG lowering */
    bool        omit_implicit_primitive_id; /* Caller proves FS does not read it; false if unknown. */
    bool        primitive_id_per_primitive; /* FS consumes an implicit PS5 NGG VS export, not a GS varying. */
    bool        fragment_distance_layout_valid; /* Link cull inputs to the producer's packed clip prefix. */
    uint32_t    fragment_clip_distance_count;
    bool        ps5_global_streamout; /* No-GDS NGG GS counters */
    bool        force_accelerated_dot; /* Diagnostic only: emit native dot ISA */
    uint32_t    primitive_type; /* GFX10 DI primitive type: 0, 1..6, 10..13 */
    bool        provoking_vtx_last; /* Select the final flat-shaded vertex */
    uint32_t    address32_hi; /* Upper half for ACO 32-bit GPU pointers */
    uint32_t    vertex_attribute_count;
    PsbcVertexAttribute vertex_attributes[PSBC_MAX_VERTEX_ATTRIBUTES];
    uint32_t    descriptor_binding_count;
    PsbcDescriptorBinding descriptor_bindings[PSBC_MAX_DESCRIPTOR_BINDINGS];
    uint32_t    specialization_constant_count;
    PsbcSpecializationConstant
        specialization_constants[PSBC_MAX_SPECIALIZATION_CONSTANTS];
    PsbcLinkedStageParameters previous_parameters;
    PsbcLinkedStageParameters next_link_parameters;
    /* Keep push constants indirect so the public runtime can upload one
     * bounded block and bind its 32-bit gfx1013 address in a user SGPR. */
    bool        force_indirect_push_constants;
    uint32_t    rasterization_samples; /* 0=single/default, otherwise 1/2/4/8 */
    /* Whether the pipeline enables per-sample shading (Vulkan's
     * sampleShadingEnable). The standalone fragment compile has to know it:
     * radv_nir_lower_opt_fs_frag_pos chooses between the per-sample position
     * path and the pixel-centre one from this state, and when it cannot decide
     * it emits a runtime selection that reads the PS state user SGPR - an ABI a
     * standalone caller does not otherwise supply. Zero keeps every earlier
     * caller's behaviour (sample shading disabled). */
    bool        sample_shading_enable;
    /* The pipeline's input patch size. A tessellation hull compile needs it to
     * size the LS/HS workgroup and name the patch count its launch state
     * carries; every other compile ignores it. */
    uint32_t    patch_control_points;
    uint32_t    spi_shader_col_format; /* Per-MRT export nibbles; 0=legacy defaults */
    uint32_t    color_is_int8;         /* Per-MRT narrow integer clamp masks */
    uint32_t    color_is_int10;
    /* Appended opt-in SPIR-V capabilities.  A compiler consumer must mirror
     * the features enabled on its logical device instead of silently
     * accepting a module that the public API did not enable. */
    bool        enable_int8;
    bool        enable_int16;
    bool        enable_storage_buffer_8bit_access;
    bool        enable_uniform_and_storage_buffer_8bit_access;
    bool        enable_storage_buffer_16bit_access;
    bool        enable_uniform_and_storage_buffer_16bit_access;
    bool        enable_physical_storage_buffer_addresses;
    bool        enable_vulkan_memory_model;
    bool        enable_vulkan_memory_model_device_scope;
    /* Derive the used descriptor sets, and the used bindings inside them, from
     * the optimized NIR instead of treating every layout binding as used.  A
     * caller that supplies legacy texture indices carrying no Vulkan deref must
     * leave this false and keep the conservative layout fallback. */
    bool        static_descriptor_use;
    /* Device facts the merged NGG pipeline state needs and this compiler
     * cannot derive.  ac_compute_late_alloc() reads the SA/CU topology
     * (min_good_cu_per_sa, family) and the per-shader-engine PC-line budget
     * (pc_lines) of the device the shader will run on; libpsbc has no
     * radeon_info of its own, and guessing those numbers is exactly what the
     * native evidence already rejected.  When ngg_device_facts is false the
     * metadata carries no GE PC-line allocation (the previous behaviour). */
    /* DIAGNOSTIC: compile an NGG stage with passthrough forced OFF.
     *
     * radv enables NGG passthrough whenever the stage does not cull and is
     * not a vertex shader exporting the primitive ID, which includes a
     * tessellation EVALUATION half. On PS5 no working pipeline has ever run
     * passthrough with ES_EN = ES_STAGE_DS - every passing NGG draw on the
     * device is vertex-fed - so the combination is untested there. Forcing it
     * off has to happen in the COMPILER, because the mode decides both the
     * generated code and the published VGT_SHADER_STAGES_EN: clearing the
     * register bit alone would mismatch a shader ACO already built for
     * passthrough. Default false keeps radv's own decision. */
    bool        ngg_no_passthrough;
    bool        ngg_device_facts;
    uint32_t    ngg_pc_lines;           /* radeon_info::pc_lines */
    uint32_t    ngg_min_good_cu_per_sa; /* radeon_info::min_good_cu_per_sa */
    bool        ngg_culling;            /* info->has_ngg_culling */
    bool        ngg_uses_scratch;       /* scratch_bytes_per_wave > 0 */
} PsbcCompileOptions;

/* === API === */

/*
 * Initialize the psbc library. Must be called once before any compilation.
 * Refcounted — call psbc_shutdown() once per psbc_init() when done.
 * Performs Mesa one-time init (glsl_type_singleton, etc.).
 */
void psbc_init(void);

/*
 * Shutdown the psbc library. Decrements refcount, cleans up when it reaches 0.
 */
void psbc_shutdown(void);

/* Immutable PS5 ACO NIR options for Gallium/front-end shader creation. */
const struct nir_shader_compiler_options*
psbc_get_nir_options(PsbcStage stage);

/*
 * Compile a SPIR-V shader into a PS4/PS5 shader binary.
 *
 *   spirv       — pointer to SPIR-V bytecode
 *   spirv_size  — size of SPIR-V bytecode in bytes
 *   opts        — compilation options (target, stage, entrypoint, etc.)
 *   out         — receives the legacy wrapper, raw code, and typed metadata;
 *                 release all owned memory with psbc_free_output()
 *
 * The standalone builder emits GNM wrappers for all represented stages.
 * On GFX9+, a standalone geometry result is a compiler diagnostic only;
 * a loadable geometry pipeline also requires a merged pre-raster stage and,
 * for legacy geometry, a separately compiled copy shader.
 *
 * Returns PSBC_RESULT_OK on success, error code otherwise.
 * On error, all output fields are zero.
 */
PsbcResult psbc_compile_shader(
    const uint32_t*       spirv,
    size_t                spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput*     out
);

/*
 * Compile an existing NIR shader.  The input remains owned by the caller;
 * libpsbc clones it before optimization and target lowering.  This is the
 * Gallium-facing path and requires opts->stage to match nir->info.stage.
 */
PsbcResult psbc_compile_nir(
    const struct nir_shader* nir,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
);

/* Compile a complete PS5 NGG vertex+geometry pair into one hardware shader. */
PsbcResult psbc_compile_geometry_pipeline(
    const uint32_t* vertex_spirv,
    size_t vertex_spirv_size,
    const uint32_t* geometry_spirv,
    size_t geometry_spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
);

/* Experimental TES+GS merged executable. TCS is link-only. Current parameters
 * name GS; previous_parameters name TES; next_link_parameters name TCS.
 * Native launch acceptance is separate from successful compilation. */
PsbcResult psbc_compile_tess_geometry_pipeline(
    const uint32_t *control, size_t control_size,
    const uint32_t *evaluation, size_t evaluation_size,
    const uint32_t *geometry, size_t geometry_size,
    const PsbcCompileOptions *opts, PsbcShaderOutput *out);

/* NIR equivalent used by Gallium when a geometry shader is bound. */
PsbcResult psbc_compile_nir_geometry_pipeline(
    const struct nir_shader* vertex_nir,
    const struct nir_shader* geometry_nir,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
);

/* Compile a PS5 tessellation pipeline's two hull halves: the vertex shader as
 * the LS half and the tessellation-control shader as the HS half.  Both are
 * compiled with the pinned ACO path, the LS half is published through the
 * hull_ls_* metadata fields, and the HS half's RSRC1/RSRC2 are replaced by the
 * combined pair radv_shader_combine_cfg_vs_tcs() produces.  The returned
 * output is still not a loadable hull package (the LS code and the hull state
 * the pipeline owns are not in it), which PSBC_UNRESOLVED_TESS_PIPELINE keeps
 * explicit. */
/* Compile a PS5 tessellation pipeline's DOMAIN half with its control half
 * linked in. The control half is used only for cross-stage info - the
 * returned program is the evaluation half alone, as an NGG package - but the
 * link is what keeps num_tess_patches, the attribute stride and
 * tes_reads_tess_factors compile-time constants. Compiled alone, the
 * evaluation half instead reads all three at runtime from the
 * tcs_offchip_layout user SGPR (radv_nir_lower_abi), which is a different ABI
 * and one no consumer here supplies. */
PsbcResult psbc_compile_domain_pipeline(
    const uint32_t* tess_ctrl_spirv,
    size_t tess_ctrl_spirv_size,
    const uint32_t* tess_eval_spirv,
    size_t tess_eval_spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
);

PsbcResult psbc_compile_tess_pipeline(
    const uint32_t* vertex_spirv,
    size_t vertex_spirv_size,
    const uint32_t* tess_ctrl_spirv,
    size_t tess_ctrl_spirv_size,
    /* The evaluation half, LINK-ONLY: none of it is compiled into the
     * returned program. It is required because the tessellator's domain,
     * spacing, winding and point mode are declared in the evaluation half by
     * GLSL convention, and the tessellation-factor LAYOUT the control half
     * stores depends on the domain - triangles get four contiguous dwords,
     * quads get six. A control half compiled without it reports
     * TESS_PRIMITIVE_UNSPECIFIED and ac_nir_lower_tess_io_to_mem falls
     * through to the quad layout, which puts inner[0] where the hardware
     * tessellator reads outer[3]. */
    const uint32_t* tess_eval_spirv,
    size_t tess_eval_spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput* out
);

/*
 * Free a PsbcShaderOutput previously filled by psbc_compile_shader().
 */
void psbc_free_output(PsbcShaderOutput* out);

/*
 * Convert a PsbcResult to a human-readable string.
 */
const char* psbc_result_string(PsbcResult result);

/*
 * Convert a stage name string ("vertex", "fragment", etc.) to PsbcStage.
 * Returns PSBC_STAGE_NONE if the name is not recognized.
 */
PsbcStage psbc_stage_from_name(const char* name);

#ifdef __cplusplus
}
#endif

#endif /* PSBC_COMPILE_H */
