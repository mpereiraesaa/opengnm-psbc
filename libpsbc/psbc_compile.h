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

#define PSBC_SHADER_METADATA_VERSION 14u

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
/* Private matching key shared by our AGC producer/consumer packages, after
 * the generic keys 15..46. This is not a PSSL system-semantic enum. */
#define PSBC_SEMANTIC_PRIMITIVE_ID 47u
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

typedef enum {
    PSBC_HW_STAGE_UNKNOWN = 0,
    PSBC_HW_STAGE_VERTEX  = 1,
    PSBC_HW_STAGE_PIXEL   = 2,
    PSBC_HW_STAGE_NGG     = 3,
} PsbcHardwareStage;

typedef enum {
    PSBC_UNRESOLVED_NONE                   = 0,
    PSBC_UNRESOLVED_PROGRAM_CHECKSUM       = 1u << 0,
    PSBC_UNRESOLVED_NGG_ESGS_RING_ITEMSIZE = 1u << 1,
    PSBC_UNRESOLVED_AGC_LINKAGE             = 1u << 2,
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
     * invent one. Version 14 added these two fields. */
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
    /* Keep push constants indirect so the public runtime can upload one
     * bounded block and bind its 32-bit gfx1013 address in a user SGPR. */
    bool        force_indirect_push_constants;
    uint32_t    rasterization_samples; /* 0=single/default, otherwise 1/2/4/8 */
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
    /* Derive the used descriptor sets, and the used bindings inside them, from
     * the optimized NIR instead of treating every layout binding as used.  A
     * caller that supplies legacy texture indices carrying no Vulkan deref must
     * leave this false and keep the conservative layout fallback. */
    bool        static_descriptor_use;
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

/* NIR equivalent used by Gallium when a geometry shader is bound. */
PsbcResult psbc_compile_nir_geometry_pipeline(
    const struct nir_shader* vertex_nir,
    const struct nir_shader* geometry_nir,
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
