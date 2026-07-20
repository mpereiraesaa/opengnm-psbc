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
- OpenOrbis cross-compilation support (`libpsbc.orbis.a`, 477 objects)
- Automated test suite (`tests/verify_sb.py`) verifies shader binary
  structure (PSSL header, GNM magic, CRC32) for all 8 shader stages

## Usage

```sh
# Compile a vertex shader (defaults to PS5 / GFX10.3)
opengnm-psbc -s vertex -f input.spv -o output.sb

# Compile a fragment shader targeting PS5
opengnm-psbc -s fragment -f input.spv -o output.sb

# Compile a compute shader (64x1x1 workgroup)
opengnm-psbc -s compute -f input.spv -o output.sb

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

- **Mesa 26.2.0** — NIR, SPIRV-to-NIR, ACO, radv shader info (vendored in `src/`)
- **opengnm** — GNM shader binary format types (`GnmShaderFileHeader`, etc.)
- **OpenOrbis PS4 Toolchain** — for PS4 cross-compilation (optional)
- **openorbis-compat** — GL stub headers for PS4 (in `../tools/openorbis-compat/`)

## License

Most of this project's code is from Mesa, licensed under the MIT license.
The rest is also licensed under the MIT license, see [COPYING](COPYING).
