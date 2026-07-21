# opengnm-psbc — Plan

> **Goal:** A SPIR-V to PS5 Shader Binary compiler, built on Mesa 26.2.0
> (NIR + ACO), producing the `GnmShaderFileHeader` / `GnmVsShader` /
> `GnmPsShader` binary container format that `sceGnmSetVsShader` /
> `sceGnmSetPsShader` consume. The PS5 GPU is RDNA2 (GFX10.3).
> Legacy PS4 (GFX7) and PS4 Pro (GFX8/NEO) targets are also supported.
>
> **Mesa version:** 26.2.0-devel (vendored from workspace `mesa/`)
> **GNM headers:** opengnm (`../opengnm/include/`)
> **Firmware RE reference:** `tools/gnm_driver_fw900_analysis.md` (shader binary parser RE'd)

---

## Architecture

```
SPIR-V input
    ↓
[libpsbc: psbc_compile_shader()]
    ↓
[Mesa SPIRV-to-NIR]  — src/compiler/spirv/
    ↓ NIR
[Mesa NIR optimizations] — src/compiler/nir/
    ↓ NIR
[radv shader info + args] — src/amd/vulkan/
    ↓
[ACO instruction selection] — src/amd/compiler/
    ↓ GCN ISA machine code
[buildshaderbinary] — libpsbc/psbc_compile.c
    ↓
PSSL header + GnmShaderFileHeader + GnmVsShader/GnmPsShader/GnmCsShader/
GnmGsShader/GnmHsShader/GnmEsShader/GnmLsShader
    + GCN code + GnmShaderBinaryInfo (with CRC32) + PsslBinaryParamInfo
    ↓
Output .sb file → sceGnmSetVsShader/sceGnmSetPsShader/sceGnmSetCsShader/
                 sceGnmSetGsShader/sceGnmSetHsShader/sceGnmSetEsShader/
                 sceGnmSetLsShader

Consumers:
  - opengnm-psbc CLI (cmd/psbc/main.c) — host tool for offline compilation
  - vulkan-ps4 ICD (vk_ps4_compile_shader_module) — runtime compilation on PS4
```

## Project Structure

```
opengnm-psbc/
├── cmd/psbc/
│   ├── main.c              # SPIRV→NIR→ACO→ShaderBinary compiler (CLI)
│   ├── crc32_sb.c          # CRC32 for shader binary validation
│   └── crc32_sb.h
├── libpsbc/
│   ├── psbc_compile.c      # Reusable compilation library (SPIRV→GCN binary)
│   └── psbc_compile.h      # Public C API (psbc_compile_shader, psbc_init, etc.)
├── psbc_stubs.c            # Stubs for excluded Mesa functions (ac_gpu_info, etc.)
├── tests/
│   ├── tri.vert            # Simple vertex shader test
│   ├── tri.frag            # Simple fragment shader test
│   ├── test.comp           # Compute shader test (64x1x1 workgroup)
│   ├── test.geom           # Geometry shader test
│   ├── test.tesc           # Tessellation control (hull) shader test
│   ├── test.tese           # Tessellation evaluation (domain) shader test
│   └── verify_sb.py        # Automated .sb output verification (magic, CRC32, structure)
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
├── config.mak               # Host build config
├── config.orbis.mak         # OpenOrbis (PS4) cross-compile config
├── Makefile                 # Host build (CLI + libpsbc.a)
├── Makefile.orbis           # Orbis cross-compile (libpsbc.orbis.a)
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

> **Development priority:** opengnm-psbc is deferred until opengnm is complete.
> opengnm (the runtime GNM library) is the primary deliverable. opengnm-psbc
> (the shader compiler) depends on opengnm's headers and is developed afterward.
> Phases 1-7 are done. All 8 shader stages (VS/PS/CS/GS/HS/DS/ES/LS) are
> implemented with full metadata (shader hash, input usage slots, chunk
> offsets) and pass structural + CRC32 verification. Phase 7 hardware
> validation confirmed VS+PS shaders render correctly on a real PS4 (FW 9.00).

### Phase 1: Project skeleton + Mesa vendoring [DONE]
- [x] Create opengnm-psbc/ with fresh git history
- [x] Vendor Mesa 26.2.0 components from workspace `mesa/` (901 files, 24.5 MB)
- [x] Copy psbc-specific code (cmd/psbc/main.c, crc32_sb.c/h)
- [x] Port PSSL types header to use opengnm headers (`<gnm_shaderbinary.h>`)
- [x] Write build system skeleton (Makefile, config.mak)
- [x] Write plan document with Mesa 26.2.0 API migration table
- [x] Initial git commit (46a71ba)

### Phase 2: Port main.c to Mesa 26.2.0 APIs [DONE]
- [x] Construct `radv_compiler_info` (ac_compiler_info, hw config, NIR options)
- [x] Construct `radv_shader_stage` (spirv data, entrypoint, stage info, key)
- [x] Update `radv_shader_spirv_to_nir` call to new signature
- [x] Update `radv_nir_shader_info_init` / `radv_nir_shader_info_pass` calls
- [x] Update `radv_declare_shader_args` call
- [x] Update `radv_postprocess_nir` call (set `stage.nir` before call)
- [x] Use `radv_shader_nir_to_asm()` for ACO compilation (replaces direct `aco_compile_shader`)
- [x] Add GFX10.3 (PS5/RDNA2) target support with `CHIP_NAVI21`
- [x] Fix `get_num_pos_exports` to match Mesa's `radv_get_num_pos_exports` (clip/cull dist masks)
- [x] Fix `compute_db_shader_control` to match Mesa 26.2.0 `radv_precompute_registers_hw_ps`
- [x] Fix `spipsincontrol` NUM_INTERP handling for GFX10.3
- [x] Replace `gl_shader_stage` with `mesa_shader_stage` (not defined in new Mesa headers)
- [x] Create `radv_constants.h` with `MAX_SETS`, `MAX_VERTEX_ATTRIBS`, `MAX_RTS`, `RADV_MAX_HEAPS`
- [x] Fix `.gitignore` to not ignore `cmd/psbc/` directory

### Phase 3: Build system completion [DONE]
- [x] Add all Mesa source files to Makefile (NIR, SPIRV, ACO, AMD common, RADV, util)
- [x] Add Python codegen rules for generated sources (nir_opcodes, aco_opcodes, etc.)
- [x] Add compile rules for C and C++ sources
- [x] Add link rule for host CLI (`opengnm-psbc`)
- [x] Add OpenOrbis cross-compile support (`Makefile.orbis`, `config.orbis.mak`)
- [x] Produce `libpsbc.orbis.a` (477 PS4/FreeBSD ELF objects) for PS4 target
- [x] Link `libpsbc.orbis.a` into `vulkan-ps4` ICD (`libvulkan_ps4.so`)
- [x] Fix OpenOrbis portability: `alloca`, `strcasecmp`, futex backend, `detect_os.h`
- [x] Fix GL header compat stub (`openorbis-compat/include/GL/gl.h`)

### Phase 4: Compile + test [DONE]
- [x] Compile opengnm-psbc on host (macOS arm64)
- [x] Test with a simple SPIR-V vertex shader (`tests/tri.vert`)
- [x] Verify output .sb file has correct GnmShaderFileHeader magic ("Shdr")
- [x] Verify CRC32 is correct
- [x] Add fragment shader test (`tests/tri.frag`)
- [x] Add compute shader test (`tests/test.comp`)
- [x] Write automated verification script (`tests/verify_sb.py`)
- [ ] Compare output with RE-6 shader binary parser findings (deferred to hardware test)
- [ ] Validate on PS4 hardware (deferred)

### Phase 5: Shader stage support [DONE — VS/PS/CS/GS/HS/DS/ES/LS]
- [x] Compute shader (CS) support: `GnmCsShader` struct, `buildshaderbinary` CS case
- [x] Thread group size from `nir->info.workgroup_size`
- [x] Input usage slots for all stages
- [x] Geometry shader (GS) support: `GnmGsShader` struct + legacy GS lowering + header construction
- [x] Hull shader (HS/TCS) support: `GnmHsShader` struct + VGT_TF_PARAM encoding + header construction
- [x] Domain shader (DS/TES) support: uses `GnmVsShader` struct with `GNM_SHB_DS_VS` binary type
- [x] Export shader (ES) support: `GnmEsShader` struct with `GNM_SHB_VS_ES` binary type
- [x] Local shader (LS) support: `GnmLsShader` struct with `GNM_SHB_VS_LS` binary type
- [x] HS input semantics fix: allocate + write `GnmVertexInputSemantic` entries
- [x] CRC32 verification in `verify_sb.py` using PS4-specific algorithm
- [x] All 8 shader types pass structural + CRC32 verification

### Phase 6: Shader binary metadata [DONE]
- [x] Populate `shaderhash0`/`shaderhash1` from FNV-1a hash of SPIR-V input
- [x] Populate `numinputusageslots` in `GnmShaderBinaryInfo` (matches shader header)
- [x] Compute `chunkusagebaseoffsetdwords` — offset from OrbShdr to input usage slot table
- [x] Add metadata verification to `verify_sb.py` (hash non-zero, chunk offset valid)
- [x] All 8 shader types pass with metadata verification

### Phase 7: Hardware validation [DONE]
- [x] Validate on PS4 hardware with `sceGnmSetVsShader` / `sceGnmSetPsShader`
- [x] Test pipeline binding with real GNM command buffers

**Result (2026-07-20):** VS+PS shaders compiled by opengnm-psbc with `-4` (GFX7/PS4 base)
were validated on a real PS4 (FW 9.00). The test app loaded the `.sb` binaries, patched
shader addresses to garlic memory, submitted draw command buffers via `sceGnmSubmitCommandBuffers`,
and rendered 600 frames (10 seconds at 60fps) without errors. The GPU completed all frames
successfully, confirming that opengnm-psbc produces valid PS4 shader binaries that are accepted
by the GNM driver and execute correctly on GFX7 hardware.

### Code review fixes [DONE]
- [x] Null-layout NIR descriptor index handling (all 5 `state->layout->set[].layout` guards)
- [x] Darwin-only `AR` path → conditional on `uname -s`
- [x] Remove `_DARWIN_C_SOURCE` from Orbis config
- [x] Futex timeout return convention (re-check value after ETIMEDOUT)
- [x] Futex thundering herd (signal for count==1, broadcast for count>1)
- [x] `ac_get_harvested_configs` null-check asymmetry
- [x] `g_init_refcount` thread safety (already protected by pthread_mutex)
- [x] Hoist `psbc_init`/`psbc_shutdown` to device lifecycle in vulkan-ps4

## Future Work

All 7 phases are complete. opengnm-psbc is feature-complete and hardware-validated.
Potential future improvements:

- **PS5 hardware validation**: Test on PS5 (GFX10.3) hardware once available
- **Texture/sampler descriptor support**: Full descriptor set lowering for
  textures and samplers (currently handles UBO/SSBO)
- **Geometry/tessellation pipeline hardware test**: Validate GS/HS/DS/ES/LS
  shaders on PS4 hardware with a multi-stage pipeline test app
- **Compute shader hardware test**: Validate CS execution on PS4 hardware
- **Shader caching**: Cache compiled shader binaries to avoid recompilation
- **SPIR-V optimization passes**: Enable additional NIR optimization passes
  for smaller/faster generated GCN code
- **Debug info**: Optional source mapping / disassembly output for debugging

## RE Reference

The shader binary format is RE'd in `tools/gnm_driver_fw900_analysis.md`:
- Shader binary parser section: metadata field offsets, CRC32 algorithm
- Shader setup functions: PM4 register offsets per stage
- `crc32_sb.c` implements the exact CRC32 used by the firmware

## Dependencies

- **Mesa 26.2.0** — vendored in `src/` (NIR, ACO, SPIRV, radv, util)
- **opengnm** — `../opengnm/include/` (GnmShaderFileHeader, GnmVsShader, GnmPsShader, GnmCsShader, etc.)
- **OpenOrbis PS4 Toolchain** — for PS4 cross-compilation (`OO_PS4_TOOLCHAIN`)
- **openorbis-compat** — `openorbis-compat/include/` (GL/POSIX stub headers for PS4, vendored)
- **Python 3 + py3-mako** — for Mesa codegen scripts
- **C11 + C++17 compiler** — for building (clang recommended)
- **glslangValidator** — for compiling GLSL test shaders to SPIR-V

## Build Commands

### Host build (CLI tool + libpsbc.a)
```bash
cd opengnm-psbc
make -j$(nproc)           # builds opengnm-psbc CLI + libpsbc.a
python3 tests/verify_sb.py  # compile + verify test shaders
```

### PS4 (Orbis) cross-compile (libpsbc.orbis.a)
```bash
export OO_PS4_TOOLCHAIN=/path/to/OpenOrbis-PS4-Toolchain
cd opengnm-psbc
make -f Makefile.orbis -j$(nproc)  # builds libpsbc.orbis.a (477 objects)
```

### vulkan-ps4 ICD (links against libpsbc.orbis.a)
```bash
export OO_PS4_TOOLCHAIN=/path/to/OpenOrbis-PS4-Toolchain
cd vulkan-ps4
make -f Makefile.orbis -j$(nproc)  # builds libvulkan_ps4.so
```
