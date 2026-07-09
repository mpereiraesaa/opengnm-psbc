# opengnm-psbc

opengnm-psbc is a SPIR-V to PlayStation 5 Shader Binary compiler, built on
Mesa's NIR and ACO compilers.

It takes SPIR-V shaders and compiles them into the PS5 `GnmShaderFileHeader` /
`GnmVsShader` / `GnmPsShader` binary container format that
`sceGnmSetVsShader` / `sceGnmSetPsShader` consume.

The PS5 GPU is RDNA2 (GFX10.3). Legacy PS4 (GFX7) and PS4 Pro (GFX8/NEO)
targets are also supported for compatibility.

## Features

- SPIR-V input → NIR → ACO → GCN ISA → PS5 Shader Binary output
- Vertex and Pixel (Fragment) shaders supported
- Primary target: PS5 (GFX10.3/RDNA2, `CHIP_NAVI21`)
- Legacy targets: PS4 base (GFX7/Sea Islands), PS4 Pro (GFX8/Polaris)
- Built on Mesa 26.2.0 (NIR + ACO compiler infrastructure)
- Uses `radv_shader_nir_to_asm()` for the full ACO compile pipeline
- Uses opengnm headers for the GNM shader binary format

## Usage

```sh
# Compile a vertex shader (defaults to PS5 / GFX10.3)
opengnm-psbc -s vertex -f input.spv -o output.sb

# Compile a fragment shader targeting PS5
opengnm-psbc -s fragment -f input.spv -o output.sb

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

You need:
- A C11 compiler with GNU extensions
- A C++17 compiler
- GNU Make
- Python 3 with py3-mako package
- [opengnm](../opengnm) headers (for the GNM shader binary format types)

```sh
make
make install DESTDIR=/usr/local
```

## Dependencies

- **Mesa 26.2.0** — NIR, SPIRV-to-NIR, ACO, radv shader info (vendored in `src/`)
- **opengnm** — GNM shader binary format types (`GnmShaderFileHeader`, etc.)

## License

Most of this project's code is from Mesa, licensed under the MIT license.
The rest is also licensed under the MIT license, see [COPYING](COPYING).
