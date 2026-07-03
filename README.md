# opengnm-psbc

opengnm-psbc is a SPIR-V to PlayStation 4 Shader Binary compiler, built on
Mesa's NIR and ACO compilers.

It takes SPIR-V shaders and compiles them into the PS4 `GnmShaderFileHeader` /
`GnmVsShader` / `GnmPsShader` binary container format that
`sceGnmSetVsShader` / `sceGnmSetPsShader` consume.

## Features

- SPIR-V input → NIR → ACO → GCN ISA → PS4 Shader Binary output
- Vertex and Pixel (Fragment) shaders supported
- Targets both base PS4 (GFX7/Sea Islands) and PS4 Pro (GFX8/Polaris)
- Built on Mesa 26.2.0 (NIR + ACO compiler infrastructure)
- Uses opengnm headers for the PS4 shader binary format

## Usage

```sh
# Compile a vertex shader
opengnm-psbc -s vertex -f input.spv -o output.sb

# Compile a fragment shader targeting PS4 Pro
opengnm-psbc -s fragment -f input.spv -o output.sb -n

# Disable optimisations
opengnm-psbc -s vertex -f input.spv -o output.sb -Od
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
- **opengnm** — PS4 GNM shader binary format types (`GnmShaderFileHeader`, etc.)

## License

Most of this project's code is from Mesa, licensed under the MIT license.
The rest is also licensed under the MIT license, see [COPYING](COPYING).
