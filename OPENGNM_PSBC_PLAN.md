# opengnm-psbc — Plan

> **Goal:** A SPIR-V to PS4 Shader Binary compiler, built on Mesa 26.2.0
> (NIR + ACO), producing the `GnmShaderFileHeader` / `GnmVsShader` /
> `GnmPsShader` binary container format that `sceGnmSetVsShader` /
> `sceGnmSetPsShader` consume.
>
> **Mesa version:** 26.2.0-devel (vendored from workspace `mesa/`)
> **GNM headers:** opengnm (`../opengnm/include/`)
> **Firmware RE reference:** `tools/gnm_driver_fw900_analysis.md` (shader binary parser RE'd)

---

## Architecture

```
SPIR-V input
    ↓
[Mesa SPIRV-to-NIR]  — src/compiler/spirv/
    ↓ NIR
[Mesa NIR optimizations] — src/compiler/nir/
    ↓ NIR
[radv shader info + args] — src/amd/vulkan/
    ↓
[ACO instruction selection] — src/amd/compiler/
    ↓ GCN ISA machine code
[PS4 Shader Binary packaging] — cmd/psbc/main.c
    ↓
GnmShaderFileHeader + GnmVsShader/GnmPsShader + GCN code + GnmShaderBinaryInfo
    ↓
Output .sb file → sceGnmSetVsShader/sceGnmSetPsShader
```

## Project Structure

```
opengnm-psbc/
├── cmd/psbc/
│   ├── main.c              # SPIRV→NIR→ACO→ShaderBinary compiler
│   ├── crc32_sb.c          # CRC32 for shader binary validation
│   └── crc32_sb.h
├── include/
│   ├── pssl_types.h        # PSSL binary format types (ported)
│   └── mesa/               # Mesa compat headers
├── src/                     # Vendored Mesa 26.2.0
│   ├── amd/
│   │   ├── common/          # AMD shared utilities
│   │   ├── compiler/        # ACO shader compiler
│   │   ├── registers/       # GPU register JSON definitions
│   │   └── vulkan/          # radv shader info/pipeline (shader-related only)
│   │       └── nir/         # radv NIR passes
│   ├── compiler/
│   │   ├── nir/             # NIR shader IR
│   │   ├── spirv/           # SPIRV-to-NIR frontend
│   │   └── glsl/            # shader_enums.h
│   ├── gallium/include/     # pipe format definitions
│   ├── mesa/main/           # Mesa config headers (minimal)
│   └── util/                # Mesa utility library
├── config.mak
├── Makefile
├── .gitignore
├── COPYING
└── README.md
```

## Mesa 26.2.0 Vendoring Notes

**901 files vendored, ~24.5 MB** (full Mesa tree is 510MB).

| Component | Path | Files | Size |
|-----------|------|-------|------|
| NIR (shader IR) | `src/compiler/nir/` | 274 | 5.5M |
| SPIRV-to-NIR | `src/compiler/spirv/` | 28 | 1.7M |
| ACO (AMD backend) | `src/amd/compiler/` | 57 | 2.4M |
| AMD common | `src/amd/common/` | 128 | 3.6M |
| RADV shader files | `src/amd/vulkan/` | 49 | 1.2M |
| AMD registers (JSON) | `src/amd/registers/` | 22 | 3.6M |
| Mesa util | `src/util/` | 316 | 4.5M |
| Gallium include | `src/gallium/include/` | 18 | 452K |
| Mesa main/program | `src/mesa/` | 6 | 60K |
| Compiler headers | `src/compiler/` | 3 | — |
| Mesa include (vulkan/) | `include/mesa/` | 26 | 1.5M |

**Key structural changes vs old psbc Mesa snapshot:**
1. `radv_private.h` and `radv_debug.h` do NOT exist — refactored to `radv_constants.h`,
   `radv_instance.h`, `radv_formats.h`
2. `shader_enums.h` is at `src/compiler/shader_enums.h` (not under `glsl/`)
3. ACO public API: `aco_interface.h` (not `aco.h`); IR in `aco_ir.h`
4. SPIRV public header: `nir_spirv.h` (not `spirv_to_nir.h`)
5. AMD register headers are generated from JSON at build time (no static `.h` files)
6. `include/mesa/vulkan/` contains full Vulkan C headers (needed by `radv_shader.h`)

## Mesa 26.2.0 API Changes (porting from old psbc)

The old psbc was built on an older Mesa snapshot (~22.x). Mesa 26.2.0 restructured
the radv/ACO/NIR APIs. Key changes that require porting in `cmd/psbc/main.c`:

| Old API (psbc) | New API (Mesa 26.2.0) | Change |
|----------------|----------------------|--------|
| `radv_shader_spirv_to_nir(gfxlevel, input, len, stage, opts, entry, disabled)` | `radv_shader_spirv_to_nir(compiler_info, stage, options, is_internal)` | Now takes `radv_compiler_info*` + `radv_shader_stage*` structs |
| `radv_link_shader(nir)` | `radv_link_shaders_info(compiler_info, stages, count)` | Renamed, takes stage array |
| `radv_nir_shader_info_init(&info)` | `radv_nir_shader_info_init(stage, next_stage, &info)` | Added stage params |
| `radv_nir_shader_info_pass(gfxlevel, family, nir, ...)` | `radv_nir_shader_info_pass(compiler_info, nir, layout, stage_key, gfx_state, ...)` | Takes compiler_info + layout + keys |
| `radv_declare_shader_args(gfxlevel, &info, stage, ...)` | `radv_declare_shader_args(compiler_info, gfx_state, stage, ...)` | Takes compiler_info + gfx_state |
| `radv_postprocess_nir(gfxlevel, family, nir, &args, &info, ...)` | `radv_postprocess_nir(compiler_info, gfx_state, stage)` | Takes compiler_info + stage struct |
| `radv_postprocess_binary_config(gfxlevel, &info, &args, stage, &cfg)` | `radv_postprocess_binary_config(compiler_info, binary, ...)` | Restructured |
| `aco_compile_shader(...)` | `aco_compile_shader(...)` | **Same** (ACO interface stable) |

**Porting approach:** Construct `radv_compiler_info` and `radv_shader_stage` structs
at the top of `main()`, populate them with the gfx level/family/spirv data, then pass
them through the new API signatures. The ACO compilation step is unchanged.

## Phases

### Phase 1: Project skeleton + Mesa vendoring [IN PROGRESS]
- Create opengnm-psbc/ with fresh git history
- Vendor Mesa 26.2.0 components from workspace `mesa/`
- Copy psbc-specific code (cmd/, crc32_sb)
- Port PSSL types header to use opengnm headers
- Write build system (Makefile, config.mak)

### Phase 2: Port main.c to Mesa 26.2.0 APIs [CRITICAL]
- Construct `radv_compiler_info` (ac_compiler_info, hw config)
- Construct `radv_shader_stage` (spirv data, entrypoint, stage info)
- Update `radv_shader_spirv_to_nir` call to new signature
- Update `radv_nir_shader_info_init` / `info_pass` calls
- Update `radv_declare_shader_args` call
- Update `radv_postprocess_nir` call
- Update `radv_postprocess_binary_config` call
- Keep `aco_compile_shader` call as-is (stable interface)
- Update `buildsb` callback to use new `radv_postprocess_binary_config` signature

### Phase 3: Build system completion [CRITICAL]
- Add all Mesa source files to Makefile
- Add Python codegen rules for generated sources (nir_opcodes, aco_opcodes, etc.)
- Add compile rules for C and C++ sources
- Add link rule

### Phase 4: Compile + test [IMPORTANT]
- Compile opengnm-psbc on host
- Test with a simple SPIR-V vertex shader
- Verify output .sb file has correct GnmShaderFileHeader magic
- Verify CRC32 is correct
- Compare output with RE-6 shader binary parser findings

### Phase 5: Complete shader stage support [LOW]
- Add GS/HS/LS/ES/CS support (currently only VS/PS)
- Fill remaining shader binary metadata (resource table, input usage slots)
- Fix resource table index generation

## RE Reference

The shader binary format is RE'd in `tools/gnm_driver_fw900_analysis.md`:
- Shader binary parser section: metadata field offsets, CRC32 algorithm
- Shader setup functions: PM4 register offsets per stage
- `crc32_sb.c` implements the exact CRC32 used by the firmware

## Dependencies

- **Mesa 26.2.0** — vendored in `src/` (NIR, ACO, SPIRV, radv, util)
- **opengnm** — `../opengnm/include/` (GnmShaderFileHeader, GnmVsShader, etc.)
- **Python 3 + py3-mako** — for Mesa codegen scripts
- **C11 + C++17 compiler** — for building
