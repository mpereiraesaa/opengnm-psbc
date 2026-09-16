# opengnm-psbc

opengnm-psbc is a SPIR-V to PlayStation 4/5 Shader Binary compiler, built on
Mesa's NIR and ACO compilers.

It takes SPIR-V shaders and compiles them into the PS4/PS5
`GnmShaderFileHeader` / `GnmVsShader` / `GnmPsShader` / `GnmCsShader` /
`GnmGsShader` / `GnmHsShader` / `GnmEsShader` / `GnmLsShader`
binary container format that `sceGnmSet*Shader` functions consume.

The PS5 GPU is RDNA2 (GFX10.3). Legacy PS4 (GFX7) and PS4 Pro
(GFX8/NEO) targets are also supported for compatibility.

## Features

- SPIR-V input → NIR → ACO → GCN ISA → PS4/PS5 Shader Binary output
- All 8 shader stages supported: Vertex, Pixel (Fragment), Compute,
  Geometry, Hull (Tessellation Control), Domain (Tessellation Evaluation),
  Export (VS variant for geometry pipeline), Local (VS variant for
  tessellation pipeline)
- Primary target: PS5 (GFX10.3/RDNA2, `CHIP_NAVI21`)
- Legacy targets: PS4 base (GFX7/Sea Islands), PS4 Pro (GFX8/Polaris)
- Built on Mesa 26.2.0 (NIR + ACO compiler infrastructure)
- Uses `radv_shader_nir_to_asm()` for the full ACO compile pipeline
- Uses opengnm headers for the GNM shader binary format
- PS4-specific CRC32 validation (non-standard index transformation)
- Reusable C library (`libpsbc`) — linked by both the CLI and the
  `vulkan-ps4` ICD for runtime shader compilation on PS4
- Typed runtime descriptor metadata for up to four descriptor sets, including
  uniform buffers, storage buffers, uniform texel buffers and combined image
  samplers.  On PS5 the metadata reports the compiler-assigned user-SGPR slot
  for every descriptor-table pointer; the caller remains responsible for
  encoding and binding the matching tables.
- Vulkan specialization constants and an indirect push-constant pointer ABI
  are available through `PsbcCompileOptions`; emitted metadata reports the
  compiler-selected user-SGPR slot and the required push-constant byte range.
- Metadata version 11 includes the optimized vertex-buffer descriptor usage
  mask and whether it indexes bindings or attributes. Descriptor tables must
  be packed in increasing set-bit order; unused inputs can disappear during
  optimization. All library consumers must rebuild against the updated header.
- Metadata version 14 adds the ViewIndex user-data slot (`view_index_valid` and
  `view_index_user_data_dword`). RADV declares `ac.view_index` at its own
  user-data location rather than inside the base-vertex block DrawIndex shares,
  so the slot is that location's SGPR index; it is reported only for the stages
  this profile delivers a view index to and only when the compiled stage really
  reads the built-in. Other stages keep the conservative lowering of ViewIndex
  to zero.
- Metadata version 13 adds the DrawIndex user-data slot (`draw_id_valid` and
  `draw_id_user_data_dword`) for stages that read the built-in. It is reported
  only when the optimized shader really reads DrawIndex, and it names a dword
  inside the same user-SGPR block that already carries the base-vertex and
  start-instance slots, so a consumer that writes the block from the reported
  offsets delivers the built-in without a second ABI.
- Narrow integer and storage capabilities are explicit `PsbcCompileOptions`
  opt-ins. SPIR-V requiring 8/16-bit arithmetic or storage is rejected before
  lowering unless the caller enables the matching logical-device capability;
  storage-only support does not implicitly enable narrow arithmetic.
- OpenOrbis cross-compilation support (`libpsbc.orbis.a`, 477 objects)
- Automated test suite (`tests/verify_sb.py`) verifies shader binary
  structure (PSSL header, GNM magic, CRC32) for all 8 shader stages
- **Hardware validated**: VS+PS shaders render correctly on a real PS4
  (FW 9.00, GFX7) — confirmed via `sceGnmSetVsShader` / `sceGnmSetPsShader`
  with 600 frames of triangle rendering at 60fps

## Usage

```sh
# Compile a vertex shader (defaults to PS5 / GFX10.3)
opengnm-psbc -s vertex -f input.spv -o output.sb

# Compile a fragment shader targeting PS5
opengnm-psbc -s fragment -f input.spv -o output.sb

# Compile a compute shader (64x1x1 workgroup)
opengnm-psbc -s compute -f input.spv -o output.sb

# Emit raw PS5 code plus the runtime descriptor-table ABI. Binding records use
# set:binding:type:array-size:byte-offset:byte-stride.
opengnm-psbc -s compute -f input.spv -o output.bin --raw \
  --address32-hi 2 --metadata output.json \
  --descriptor-binding 0:0:storage_buffer:1:0:16 \
  --descriptor-binding 1:0:uniform_buffer:1:0:16 \
  --descriptor-binding 2:0:uniform_texel_buffer:1:0:16

# Compile a geometry shader
opengnm-psbc -s geometry -f input.spv -o output.sb

# Compile a hull shader (tessellation control)
opengnm-psbc -s tess-ctrl -f input.spv -o output.sb

# Compile a domain shader (tessellation evaluation)
opengnm-psbc -s tess-eval -f input.spv -o output.sb

# Compile an export shader (VS variant for geometry pipeline)
opengnm-psbc -s export -f input.spv -o output.sb

# Compile a local shader (VS variant for tessellation pipeline)
opengnm-psbc -s local -f input.spv -o output.sb

# Compile a vertex shader targeting PS4 base (GFX7)
opengnm-psbc -s vertex -f input.spv -o output.sb -4

# Compile a fragment shader targeting PS4 Pro (GFX8/NEO)
opengnm-psbc -s fragment -f input.spv -o output.sb -n

# Disable optimisations
opengnm-psbc -s vertex -f input.spv -o output.sb -Od

# Verbose output (dumps NIR at each stage)
opengnm-psbc -s vertex -f input.spv -o output.sb -vv
```

## Building

### Host build (CLI tool + libpsbc.a)

You need:
- A C11 compiler with GNU extensions
- A C++17 compiler
- GNU Make
- Python 3 with py3-mako package
- [opengnm](../opengnm) headers (for the GNM shader binary format types)

```sh
make                    # builds opengnm-psbc CLI + libpsbc.a
python3 tests/verify_sb.py  # compile + verify all 8 shader stage test shaders
make test-runtime-parameters # verify specialization + push metadata/codegen
make test-storage-widths # verify fail-closed 8/16-bit capability gates + lowering
make test-core-vertex-formats # compile all exposed 8/16-bit vertex families
make test-draw-id # verify the DrawIndex user-data slot and its absence
make install DESTDIR=/usr/local
```

### PS4 (Orbis) cross-compile (libpsbc.orbis.a)

You additionally need:
- [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain)

```sh
export OO_PS4_TOOLCHAIN=/path/to/OpenOrbis-PS4-Toolchain
make -f Makefile.orbis   # builds libpsbc.orbis.a (477 PS4/FreeBSD ELF objects)
```

The resulting `libpsbc.orbis.a` can be linked into PS4 homebrew
applications. The `vulkan-ps4` ICD links it to provide
`vkCreateShaderModule` with real SPIR-V → GCN compilation.

## Dependencies

Standalone fragment compilation recomputes NIR input bases from semantic
locations after lowering IO. This prevents multiple SPIR-V varyings from
aliasing attribute zero when no graphics-pipeline linker assigned driver
locations. Vertex input locations are not renumbered. Run
`make test-fragment-input-bases` to check sparse smooth and flat float/int/uint
inputs, their AGC semantic metadata, and both provoking-vertex modes. This
compiler test alone does not establish rendered-pixel correctness.

- **Mesa 26.2.0** — NIR, SPIRV-to-NIR, ACO, radv shader info (vendored in `src/`)
- **opengnm** — GNM shader binary format types (`GnmShaderFileHeader`, etc.)
- **OpenOrbis PS4 Toolchain** — for PS4 cross-compilation (optional)
- **openorbis-compat** — GL/POSIX stub headers for PS4 (vendored in `openorbis-compat/`)

## License

Most of this project's code is from Mesa, licensed under the MIT license.
The rest is also licensed under the MIT license, see [LICENSE](LICENSE).
