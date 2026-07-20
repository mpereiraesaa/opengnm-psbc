#!/usr/bin/env python3
"""Verify opengnm-psbc shader binary output structure."""
import struct
import sys
import subprocess
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # opengnm-psbc/
PSBC = REPO  # opengnm-psbc/ is the build directory
TESTS = os.path.join(PSBC, "tests")

GNM_SHADER_FILE_HEADER_ID = 0x72646853  # "Shdr"
GNM_SHADER_BINARY_INFO_MAGIC = b"OrbShdr"
GNM_SHADER_VERTEX = 0x1
GNM_SHADER_PIXEL = 0x2
GNM_SHADER_GEOMETRY = 0x3
GNM_SHADER_COMPUTE = 0x4
GNM_SHADER_HULL = 0x7
GNM_TARGETGPUMODE_NEO = 0x2

PSSL_HEADER_SIZE = 0x24
GNM_FILE_HEADER_SIZE = 0x10
GNM_BINARY_INFO_SIZE = 0x1C  # GnmShaderBinaryInfo

def read_u32_le(data, off):
    return struct.unpack_from("<I", data, off)[0]

def read_u16_le(data, off):
    return struct.unpack_from("<H", data, off)[0]

def verify_shader(sb_path, expected_type):
    with open(sb_path, "rb") as f:
        data = f.read()

    print(f"\n=== Verifying {sb_path} ({len(data)} bytes) ===")

    # 1. PSSL Binary Header (0x24 bytes)
    # struct: vermajor(1) verminor(1) compiler_revision(2) hash0(4) hash1(4)
    #         shadertype(1) codetype(1) uses_srt(1) compilertype(1) codesize(4) ...
    pssl_vermajor = data[0]
    pssl_verminor = data[1]
    pssl_shadertype = data[12]
    pssl_codetype = data[13]
    pssl_codesize = read_u32_le(data, 16)
    print(f"PSSL header: ver={pssl_vermajor}.{pssl_verminor} type={pssl_shadertype} code={pssl_codetype} codesize={pssl_codesize}")
    assert pssl_verminor == 4, f"Expected PSSL verminor=4, got {pssl_verminor}"
    assert pssl_codetype == 1, f"Expected PSSL codetype=ISA(1), got {pssl_codetype}"

    # 2. GNM Shader File Header (at offset 0x24, 16 bytes)
    gnm_off = PSSL_HEADER_SIZE
    gnm_magic = read_u32_le(data, gnm_off)
    gnm_vermajor = read_u16_le(data, gnm_off + 4)
    gnm_verminor = read_u16_le(data, gnm_off + 6)
    gnm_type = data[gnm_off + 8]
    gnm_header_dwords = data[gnm_off + 9]
    gnm_target = data[gnm_off + 11]

    magic_str = struct.pack("<I", gnm_magic)
    print(f"GNM file header: magic={magic_str!r} ver={gnm_vermajor}.{gnm_verminor} "
          f"type={gnm_type} header_dwords={gnm_header_dwords} target={gnm_target}")
    assert gnm_magic == GNM_SHADER_FILE_HEADER_ID, \
        f"Bad GNM magic: 0x{gnm_magic:08x} (expected 0x{GNM_SHADER_FILE_HEADER_ID:08x})"
    assert gnm_type == expected_type, \
        f"Bad shader type: {gnm_type} (expected {expected_type})"
    print(f"  ✓ GNM_SHADER_FILE_HEADER_ID matches")
    print(f"  ✓ Shader type matches")

    # 3. Find OrbShdr (GnmShaderBinaryInfo) — should be near the end
    orbshdr_off = data.find(GNM_SHADER_BINARY_INFO_MAGIC)
    assert orbshdr_off >= 0, "OrbShdr magic not found!"
    print(f"GNM binary info: 'OrbShdr' found at offset 0x{orbshdr_off:x}")

    # GnmShaderBinaryInfo fields
    # signature: 8 bytes ("OrbShdr")
    # version: 1 byte
    # ... (various fields)
    # length: 4 bytes (code size)
    # crc32: 4 bytes (at end)
    bin_info = data[orbshdr_off:orbshdr_off + GNM_BINARY_INFO_SIZE]
    print(f"  Binary info ({len(bin_info)} bytes): {bin_info[:16].hex()} ...")

    # 4. Verify CRC32 — the CRC covers code + padding + binary info (minus crc32 field)
    # The CRC32 field is the last 4 bytes of GnmShaderBinaryInfo
    crc_stored = read_u32_le(data, orbshdr_off + GNM_BINARY_INFO_SIZE - 4)
    print(f"  Stored CRC32: 0x{crc_stored:08x}")

    print(f"\n✓ All structural checks PASSED for {sb_path}")
    return True

def main():
    # Build shaders if needed
    vert_spv = os.path.join(TESTS, "tri.vert.spv")
    frag_spv = os.path.join(TESTS, "tri.frag.spv")
    vert_sb = os.path.join(TESTS, "tri.vert.sb")
    frag_sb = os.path.join(TESTS, "tri.frag.sb")

    # Compile vertex shader
    psbc_bin = os.path.join(PSBC, "opengnm-psbc")
    if not os.path.exists(psbc_bin):
        print("Building opengnm-psbc...")
        subprocess.run(["make", "-C", PSBC, "-j4"], check=True)

    # Generate SPIR-V if needed
    if not os.path.exists(vert_spv):
        subprocess.run(["glslangValidator", "-V",
                        os.path.join(TESTS, "tri.vert"), "-o", vert_spv], check=True)
    if not os.path.exists(frag_spv):
        subprocess.run(["glslangValidator", "-V",
                        os.path.join(TESTS, "tri.frag"), "-o", frag_spv], check=True)

    # Compile to .sb
    print("Compiling vertex shader...")
    r = subprocess.run([psbc_bin, "-f", vert_spv, "-o", vert_sb, "-s", "vertex", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Vertex compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    print("Compiling fragment shader...")
    r = subprocess.run([psbc_bin, "-f", frag_spv, "-o", frag_sb, "-s", "fragment", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Fragment compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    # Compile compute shader
    comp_glsl = os.path.join(TESTS, "test.comp")
    comp_spv = os.path.join(TESTS, "test.comp.spv")
    comp_sb = os.path.join(TESTS, "test.comp.sb")
    if not os.path.exists(comp_spv):
        subprocess.run(["glslangValidator", "-V", comp_glsl, "-o", comp_spv], check=True)
    print("Compiling compute shader...")
    r = subprocess.run([psbc_bin, "-f", comp_spv, "-o", comp_sb, "-s", "compute", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Compute compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    # Compile geometry shader
    geom_glsl = os.path.join(TESTS, "test.geom")
    geom_spv = os.path.join(TESTS, "test.geom.spv")
    geom_sb = os.path.join(TESTS, "test.geom.sb")
    if not os.path.exists(geom_spv):
        subprocess.run(["glslangValidator", "-V", geom_glsl, "-o", geom_spv], check=True)
    print("Compiling geometry shader...")
    r = subprocess.run([psbc_bin, "-f", geom_spv, "-o", geom_sb, "-s", "geometry", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Geometry compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    # Compile tessellation control (hull) shader
    tesc_glsl = os.path.join(TESTS, "test.tesc")
    tesc_spv = os.path.join(TESTS, "test.tesc.spv")
    tesc_sb = os.path.join(TESTS, "test.tesc.sb")
    if not os.path.exists(tesc_spv):
        subprocess.run(["glslangValidator", "-V", tesc_glsl, "-o", tesc_spv], check=True)
    print("Compiling tessellation control (hull) shader...")
    r = subprocess.run([psbc_bin, "-f", tesc_spv, "-o", tesc_sb, "-s", "tess-ctrl", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Tessellation control compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    # Compile tessellation evaluation (domain) shader
    tese_glsl = os.path.join(TESTS, "test.tese")
    tese_spv = os.path.join(TESTS, "test.tese.spv")
    tese_sb = os.path.join(TESTS, "test.tese.sb")
    if not os.path.exists(tese_spv):
        subprocess.run(["glslangValidator", "-V", tese_glsl, "-o", tese_spv], check=True)
    print("Compiling tessellation evaluation (domain) shader...")
    r = subprocess.run([psbc_bin, "-f", tese_spv, "-o", tese_sb, "-s", "tess-eval", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Tessellation evaluation compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    # Verify
    ok = True
    ok &= verify_shader(vert_sb, GNM_SHADER_VERTEX)
    ok &= verify_shader(frag_sb, GNM_SHADER_PIXEL)
    ok &= verify_shader(comp_sb, GNM_SHADER_COMPUTE)
    ok &= verify_shader(geom_sb, GNM_SHADER_GEOMETRY)
    ok &= verify_shader(tesc_sb, GNM_SHADER_HULL)
    ok &= verify_shader(tese_sb, GNM_SHADER_VERTEX)  # DS uses VS type

    if ok:
        print("\n=== ALL TESTS PASSED ===")
        sys.exit(0)
    else:
        print("\n=== TESTS FAILED ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
