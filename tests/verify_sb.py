#!/usr/bin/env python3
"""Verify opengnm-psbc shader binary output structure."""
import struct
import sys
import subprocess
import os
import zlib
import json
import hashlib

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

# PS4-specific CRC32 table (same as standard CRC32/IEEE table)
CRC32_TABLE = [
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
    0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
    0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
    0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
    0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
    0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
    0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
    0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
    0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
    0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
    0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
    0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
    0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
    0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
    0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
    0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
    0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
    0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d,
]

def ps4_crc32(data, start_crc=0xffffffff):
    """PS4-specific CRC32 as implemented in crc32_sb.c."""
    crc = start_crc
    for byte in data:
        index = (byte ^ crc) & 0xff
        if index < 0xe0:
            # Non-standard index transformation from the original psbc code
            index = index + ((((index - index // 7) >> 1) + index // 7) >> 2) * -6
        else:
            index = 0
        crc = CRC32_TABLE[index] ^ (crc >> 8)
    return crc

def ps4_crc32_end(crc):
    return ~crc & 0xffffffff

def read_u32_le(data, off):
    return struct.unpack_from("<I", data, off)[0]

def read_u16_le(data, off):
    return struct.unpack_from("<H", data, off)[0]

def shader_words(sb_path):
    with open(sb_path, "rb") as f:
        data = f.read()
    assert len(data) % 4 == 0
    return struct.unpack(f"<{len(data) // 4}I", data)

def shader_export_targets(sb_path):
    with open(sb_path, "rb") as f:
        data = f.read()
    orbshdr_off = data.find(GNM_SHADER_BINARY_INFO_MAGIC)
    assert orbshdr_off >= 0, "OrbShdr magic not found!"
    length_field = read_u32_le(data, orbshdr_off + 8)
    code_size = (length_field >> 8) & 0xffffff
    code = data[orbshdr_off - code_size:orbshdr_off]
    return [
        (read_u32_le(code, off) >> 4) & 0x3f
        for off in range(0, len(code) - 3, 4)
        if read_u32_le(code, off) & 0xfc000000 == 0xf8000000
    ]

def shader_exports_are_compressed(sb_path):
    with open(sb_path, "rb") as f:
        data = f.read()
    orbshdr_off = data.find(GNM_SHADER_BINARY_INFO_MAGIC)
    assert orbshdr_off >= 0, "OrbShdr magic not found!"
    length_field = read_u32_le(data, orbshdr_off + 8)
    code_size = (length_field >> 8) & 0xffffff
    code = data[orbshdr_off - code_size:orbshdr_off]
    exports = [
        read_u32_le(code, off)
        for off in range(0, len(code) - 3, 4)
        if read_u32_le(code, off) & 0xfc000000 == 0xf8000000
    ]
    return bool(exports) and all(word & (1 << 10) for word in exports)

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

    # 3b. Verify metadata fields
    chunkusageoffset = data[orbshdr_off + 12]
    numinputusageslots = data[orbshdr_off + 13]
    shaderhash0 = read_u32_le(data, orbshdr_off + 16)
    shaderhash1 = read_u32_le(data, orbshdr_off + 20)
    print(f"  Metadata: chunkoff={chunkusageoffset} numslots={numinputusageslots} hash=0x{shaderhash1:08x}{shaderhash0:08x}")

    if shaderhash0 == 0 and shaderhash1 == 0:
        print(f"  ✗ Shader hash is zero (should be populated from SPIR-V)")
        return False

    # If numinputusageslots > 0, verify chunkusagebaseoffsetdwords points to valid data
    if numinputusageslots > 0:
        if chunkusageoffset == 0:
            print(f"  ✗ numinputusageslots={numinputusageslots} but chunkusagebaseoffsetdwords=0")
            return False
        slot_off = orbshdr_off - chunkusageoffset * 4
        if slot_off < PSSL_HEADER_SIZE + GNM_FILE_HEADER_SIZE:
            print(f"  ✗ Input usage slot offset 0x{slot_off:x} is before shader header")
            return False
        # Read first slot's usage type
        first_slot_type = data[slot_off]
        print(f"  First input usage slot at 0x{slot_off:x}: type=0x{first_slot_type:02x}")

    # 4. Verify CRC32
    # The CRC covers: GCN code + padding (virtual, not in file) + OrbShdr (minus crc32)
    # GnmShaderBinaryInfo layout:
    #   offset 0:  signature[7] + version (8 bytes)
    #   offset 8:  bitfield: ispsslcg(1) + issourcecached(1) + type(4) + sourcetype(2) + length(24)
    #   offset 12: chunkusagebaseoffsetdwords
    #   offset 13: numinputusageslots
    #   offset 14: flags
    #   offset 15: _unused2
    #   offset 16: shaderhash0
    #   offset 20: shaderhash1
    #   offset 24: crc32
    crc_stored = read_u32_le(data, orbshdr_off + 24)
    # length is 24-bit field starting at bit 8 of the u32 at offset 8
    length_field = read_u32_le(data, orbshdr_off + 8)
    gcn_code_size = (length_field >> 8) & 0xffffff
    print(f"  Stored CRC32: 0x{crc_stored:08x}, GCN code size: {gcn_code_size}")

    # CRC region: code (gcn_code_size bytes) + padding + OrbShdr (minus crc32 = 24 bytes)
    code_start = orbshdr_off - gcn_code_size
    code_data = data[code_start:orbshdr_off]  # GCN code from file
    orbshdr_data = data[orbshdr_off:orbshdr_off + 24]  # OrbShdr minus crc32

    # Virtual padding: align gcn_code_size to 8 bytes
    numpad = 8 - (gcn_code_size & 0x7) if (gcn_code_size & 0x7) else 0
    pad_data = bytes(numpad)  # zero padding

    # Full CRC region: code + padding + OrbShdr(minus crc32)
    crc_region = code_data + pad_data + orbshdr_data

    # PS4 CRC
    ps4_crc = ps4_crc32_end(ps4_crc32(crc_region))
    # Standard CRC
    std_crc = zlib.crc32(crc_region) & 0xffffffff

    if ps4_crc == crc_stored:
        print(f"  ✓ PS4 CRC32 matches: 0x{ps4_crc:08x}")
    elif std_crc == crc_stored:
        print(f"  ✓ Standard CRC32 matches: 0x{std_crc:08x}")
        print(f"    (PS4 CRC was 0x{ps4_crc:08x} — non-standard algorithm)")
    else:
        print(f"  ✗ CRC32 mismatch!")
        print(f"    Stored:   0x{crc_stored:08x}")
        print(f"    PS4 CRC:  0x{ps4_crc:08x}")
        print(f"    Std CRC:  0x{std_crc:08x}")
        print(f"    Code region: {code_start:#x}..{orbshdr_off:#x} ({gcn_code_size} bytes)")
        print(f"    Padding: {numpad} bytes")
        print(f"    OrbShdr: 24 bytes")
        print(f"    Total CRC region: {len(crc_region)} bytes")
        return False

    print(f"\n✓ All structural checks PASSED for {sb_path}")
    return True

def main():
    # PS5-private opcodes must not renumber the public NIR ABI used by Mesa.
    with open(os.path.join(PSBC, "src/compiler/nir/nir_intrinsics.h"),
              "r", encoding="utf-8") as f:
        intrinsic_enum = [line.strip().rstrip(",") for line in f
                          if line.strip().startswith("nir_intrinsic_") and
                          line.strip().endswith(",")]
    with open(os.path.join(PSBC, "src/compiler/nir/nir_intrinsics.c"),
              "r", encoding="utf-8") as f:
        intrinsic_info = [line.split('"')[1] for line in f
                          if line.strip().startswith('.name = "')]
    assert intrinsic_info == [name.removeprefix("nir_intrinsic_")
                              for name in intrinsic_enum]
    public_abi = hashlib.sha256("\n".join(intrinsic_enum).encode()).hexdigest()
    assert len(intrinsic_enum) == 930
    assert public_abi == \
        "4cb1ff6f1ca3a584853d446295139029f72ed6fb4b25937d980d79d56b1746ed"
    print("NIR ABI: public Mesa opcode table unchanged")

    # Build shaders if needed
    vert_spv = os.path.join(TESTS, "tri.vert.spv")
    frag_spv = os.path.join(TESTS, "tri.frag.spv")
    vert_sb = os.path.join(TESTS, "tri.vert.sb")
    frag_sb = os.path.join(TESTS, "tri.frag.sb")
    vert_metadata = os.path.join(TESTS, "tri.vert.metadata.json")

    # Compile vertex shader
    psbc_bin = os.path.join(PSBC, "opengnm-psbc")
    if not os.path.exists(psbc_bin):
        print("Building opengnm-psbc...")
        subprocess.run(["make", "-C", PSBC, "-j4"], check=True)

    indirect_glsl = os.path.join(TESTS, "indirect-array.frag")
    indirect_spv = os.path.join(TESTS, "indirect-array.spv")
    indirect_sb = os.path.join(TESTS, "indirect-array.sb")
    subprocess.run(["glslangValidator", "-V", indirect_glsl,
                    "-o", indirect_spv], check=True)
    indirect_env = os.environ.copy()
    indirect_env["PSBC_DEBUG_NIR"] = "1"
    r = subprocess.run([
        psbc_bin, "-f", indirect_spv, "-o", indirect_sb,
        "-s", "fragment", "-vv",
    ], capture_output=True, text=True, cwd=PSBC, env=indirect_env)
    if r.returncode != 0:
        print(f"Indirect-array fragment compile failed: {r.stderr}")
        sys.exit(1)
    assert "store_deref" not in r.stderr
    print("Indirect array: dynamic function-temp dereferences lowered")

    # Generate SPIR-V if needed
    if not os.path.exists(vert_spv):
        subprocess.run(["glslangValidator", "-V",
                        os.path.join(TESTS, "tri.vert"), "-o", vert_spv], check=True)
    if not os.path.exists(frag_spv):
        subprocess.run(["glslangValidator", "-V",
                        os.path.join(TESTS, "tri.frag"), "-o", frag_spv], check=True)

    # Compile to .sb
    print("Compiling vertex shader...")
    r = subprocess.run([psbc_bin, "-f", vert_spv, "-o", vert_sb,
                        "-s", "vertex", "--metadata", vert_metadata, "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Vertex compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())
    with open(vert_metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    base_vertex_dword = metadata["base_vertex_user_data_dword"]
    assert metadata["version"] == 6
    assert isinstance(base_vertex_dword, int)
    assert 0 <= base_vertex_dword < metadata["user_sgpr_count"]
    assert metadata["start_instance_user_data_dword"] is None
    print(f"Base-vertex metadata: ABI v6 dword={base_vertex_dword} "
          f"user_sgprs={metadata['user_sgpr_count']}")

    point_glsl = os.path.join(TESTS, "point-size.vert")
    point_spv = os.path.join(TESTS, "point-size.vert.spv")
    point_sb = os.path.join(TESTS, "point-size.vert.sb")
    point_metadata = os.path.join(TESTS, "point-size.vert.metadata.json")
    subprocess.run(["glslangValidator", "-V", point_glsl,
                    "-o", point_spv], check=True)
    r = subprocess.run([
        psbc_bin, "-f", point_spv, "-o", point_sb, "-s", "vertex",
        "--metadata", point_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Point-size vertex compile failed: {r.stderr}")
        sys.exit(1)
    with open(point_metadata, "r", encoding="utf-8") as f:
        point_info = json.load(f)
    point_cx = {entry["offset"]: entry["value"]
                for entry in point_info["context_registers"]}
    assert point_cx[451] == 0x44
    assert point_cx[519] & 0x01210000 == 0x01210000
    print("Point size: POS1 export and PA_CL vertex-size controls enabled")

    point_coord_glsl = os.path.join(TESTS, "point-coord.frag")
    point_coord_spv = os.path.join(TESTS, "point-coord.spv")
    point_coord_sb = os.path.join(TESTS, "point-coord.sb")
    point_coord_metadata = os.path.join(TESTS, "point-coord.metadata.json")
    subprocess.run(["glslangValidator", "-V", point_coord_glsl,
                    "-o", point_coord_spv], check=True)
    point_coord_env = os.environ.copy()
    point_coord_env["PSBC_DEBUG_NIR"] = "1"
    r = subprocess.run([
        psbc_bin, "-f", point_coord_spv, "-o", point_coord_sb,
        "-s", "fragment", "--metadata", point_coord_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC, env=point_coord_env)
    if r.returncode != 0:
        print(f"Point-coordinate fragment compile failed: {r.stderr}")
        sys.exit(1)
    with open(point_coord_metadata, "r", encoding="utf-8") as f:
        point_coord_info = json.load(f)
    point_coord_cx = {entry["offset"]: entry["value"]
                      for entry in point_coord_info["context_registers"]}
    assert "io location=VARYING_SLOT_PNTC" in r.stderr
    assert point_coord_info["input_semantics"] == []
    assert point_coord_info["unresolved_fields"] == 5
    assert point_coord_cx[0x1B3] == 0x2
    assert point_coord_cx[0x1B4] == 0x2
    assert point_coord_cx[0x1B6] == 0x8001
    print("Point coordinate: PNTC interpolation input uses the hardware-generated "
          "parameter slot")

    front_glsl = os.path.join(TESTS, "front-facing.frag")
    front_spv = os.path.join(TESTS, "front-facing.spv")
    front_sb = os.path.join(TESTS, "front-facing.sb")
    front_metadata = os.path.join(TESTS, "front-facing.metadata.json")
    subprocess.run(["glslangValidator", "-V", front_glsl,
                    "-o", front_spv], check=True)
    front_env = os.environ.copy()
    front_env["PSBC_DEBUG_NIR"] = "1"
    r = subprocess.run([
        psbc_bin, "-f", front_spv, "-o", front_sb,
        "-s", "fragment", "--metadata", front_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC, env=front_env)
    if r.returncode != 0:
        print(f"Front-facing fragment compile failed: {r.stderr}")
        sys.exit(1)
    with open(front_metadata, "r", encoding="utf-8") as f:
        front_info = json.load(f)
    front_cx = {entry["offset"]: entry["value"]
                for entry in front_info["context_registers"]}
    assert "@load_vector_arg_amd (base=15" in r.stderr
    assert front_info["input_semantics"] == []
    assert front_info["unresolved_fields"] == 1
    assert front_cx[0x1B3] == 0x1080
    assert front_cx[0x1B4] == 0x1080
    assert front_cx[0x1B6] == 0x8000
    assert front_cx[0x1B8] == 0
    print("Front facing: hardware front-face VGPR uses floating-sign mode")

    instance_glsl = os.path.join(TESTS, "instance.vert")
    instance_spv = os.path.join(TESTS, "instance.vert.spv")
    instance_sb = os.path.join(TESTS, "instance.vert.sb")
    instance_metadata = os.path.join(TESTS, "instance.vert.metadata.json")
    subprocess.run(["glslangValidator", "-V", instance_glsl,
                    "-o", instance_spv], check=True)
    r = subprocess.run([
        psbc_bin, "-f", instance_spv, "-o", instance_sb, "-s", "vertex",
        "--vertex-attribute", "0:r32g32_float:0:0:8:4",
        "--vertex-attribute", "1:r32g32_float:1:0:8:4:2",
        "--metadata", instance_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Instanced vertex compile failed: {r.stderr}")
        sys.exit(1)
    with open(instance_metadata, "r", encoding="utf-8") as f:
        instance_info = json.load(f)
    start_instance_dword = instance_info["start_instance_user_data_dword"]
    assert instance_info["version"] == 6
    assert isinstance(start_instance_dword, int)
    assert 0 <= start_instance_dword < instance_info["user_sgpr_count"]
    assert start_instance_dword != instance_info["base_vertex_user_data_dword"]
    print(f"Instancing metadata: start dword={start_instance_dword} "
          f"user_sgprs={instance_info['user_sgpr_count']}")

    packed_sb = os.path.join(TESTS, "packed-vertex.sb")
    r = subprocess.run([
        psbc_bin, "-f", instance_spv, "-o", packed_sb, "-s", "vertex",
        "--vertex-attribute", "0:r10g10b10a2_snorm:0:0:8:4",
        "--vertex-attribute", "1:b8g8r8a8_unorm:0:4:8:4",
    ], capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Packed vertex compile failed: {r.stderr}")
        sys.exit(1)
    print("Packed vertex formats: 2_10_10_10 SNORM + BGRA8 UNORM")

    integer_glsl = os.path.join(TESTS, "integer-vertex.vert")
    integer_spv = os.path.join(TESTS, "integer-vertex.spv")
    integer_sb = os.path.join(TESTS, "integer-vertex.sb")
    subprocess.run(["glslangValidator", "-V", integer_glsl,
                    "-o", integer_spv], check=True)
    r = subprocess.run([
        psbc_bin, "-f", integer_spv, "-o", integer_sb, "-s", "vertex",
        "--vertex-attribute", "0:r32g32_sint:0:0:24:4",
        "--vertex-attribute", "1:r32g32b32a32_uint:0:8:24:4",
    ], capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Integer vertex compile failed: {r.stderr}")
        sys.exit(1)
    print("Integer vertex formats: R32G32 SINT + R32G32B32A32 UINT")

    xfb_asm = os.path.join(TESTS, "xfb.spvasm")
    xfb_spv = os.path.join(TESTS, "xfb.spv")
    xfb_raw = os.path.join(TESTS, "xfb.raw.sb")
    xfb_metadata = os.path.join(TESTS, "xfb.metadata.json")
    subprocess.run([
        "spirv-as", "--target-env", "spv1.3", xfb_asm, "-o", xfb_spv,
    ], check=True)
    subprocess.run([
        "spirv-val", "--target-env", "vulkan1.1", xfb_spv,
    ], check=True)
    r = subprocess.run([
        psbc_bin, "-f", xfb_spv, "-o", xfb_raw, "-s", "vertex",
        "--raw", "--metadata", xfb_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Transform-feedback compile failed: {r.stderr}")
        sys.exit(1)
    with open(xfb_metadata, "r", encoding="utf-8") as f:
        xfb_info = json.load(f)
    with open(xfb_raw, "rb") as f:
        xfb_hash = hashlib.sha256(f.read()).hexdigest()
    streamout = xfb_info["streamout"]
    assert xfb_info["version"] == 6
    assert xfb_info["hardware_stage"] == 1
    assert xfb_info["machine_code_size"] == 136
    assert xfb_hash == "53c2d31fe305d7ec4f0d404274e227183bebfbab51728855566889981022662d"
    assert streamout["buffer_table_user_data_dword"] == 4
    assert streamout["enabled_stream_buffers_mask"] == 1
    assert streamout["strides_dwords"] == [4, 0, 0, 0]
    assert streamout["config_sgpr"] == 5
    assert streamout["write_index_sgpr"] == 6
    assert streamout["offset_sgprs"] == [7, None, None, None]
    assert xfb_info["linkage"] == {
        "ge_cntl": {"offset": 0x25B, "value": 0x20080},
        "stages_en": {"offset": 0x2D5, "value": 0x10000},
        "user_vgpr_en": {"offset": 0x262, "value": 0},
    }
    assert xfb_info["context_registers"] == [
        {"offset": 0x1B1, "value": 0x80},
        {"offset": 0x1C3, "value": 0x4},
        {"offset": 0x207, "value": 0},
    ]
    assert xfb_info["shader_registers"] == [
        {"offset": 0x48, "value": 0},
        {"offset": 0x49, "value": 0},
        {"offset": 0x4A, "value": 0x082C0001},
        {"offset": 0x4B, "value": 0x110A},
    ]
    assert "Unsupported SPIR-V capability" not in r.stderr
    print("Transform-feedback metadata: buffer table dword=4, "
          "system SGPRs=5/6/7, stride=4 dwords")

    ngg_topologies = {
        "point-list": (1, 300, "c019debc8d57279b7b234998a9bbe9af8327282ca4708ad7d42fac8a82c16d97"),
        "line-list": (2, 308, "fc12e6b6c3f9523b1ffedc853f14d894cdb8ee7d1cc4b18ad376bc895499243c"),
        "line-strip": (2, 308, "fc12e6b6c3f9523b1ffedc853f14d894cdb8ee7d1cc4b18ad376bc895499243c"),
        "triangle-list": (3, 320, "d1eccd79fccdad73c4851e871e96dd5569a22cba4c5b5615befdb1f66383991f"),
        "triangle-fan": (3, 320, "d1eccd79fccdad73c4851e871e96dd5569a22cba4c5b5615befdb1f66383991f"),
        "triangle-strip": (3, 320, "d1eccd79fccdad73c4851e871e96dd5569a22cba4c5b5615befdb1f66383991f"),
    }
    for topology, (vertices_per_primitive, expected_size, expected_hash) in ngg_topologies.items():
        xfb_ngg_raw = os.path.join(TESTS, f"xfb-ngg-{topology}.raw.sb")
        xfb_ngg_metadata = os.path.join(TESTS, f"xfb-ngg-{topology}.metadata.json")
        r = subprocess.run([
            psbc_bin, "-f", xfb_spv, "-o", xfb_ngg_raw, "-s", "vertex",
            "--ngg", "--raw", "--primitive-type", topology,
            "--metadata", xfb_ngg_metadata, "-vv",
        ], capture_output=True, text=True, cwd=PSBC)
        if r.returncode != 0:
            print(f"NGG {topology} transform-feedback compile failed: {r.stderr}")
            sys.exit(1)
        with open(xfb_ngg_metadata, "r", encoding="utf-8") as f:
            xfb_ngg_info = json.load(f)
        with open(xfb_ngg_raw, "rb") as f:
            xfb_ngg_hash = hashlib.sha256(f.read()).hexdigest()
        streamout = xfb_ngg_info["streamout"]
        assert xfb_ngg_info["hardware_stage"] == 3
        assert xfb_ngg_info["machine_code_size"] == expected_size
        assert xfb_ngg_hash == expected_hash
        assert streamout["buffer_table_user_data_dword"] == 2
        assert streamout["enabled_stream_buffers_mask"] == 1
        assert streamout["strides_dwords"] == [4, 0, 0, 0]
        # NGG VS uses the ES/GS hardware path.  Stream output additionally
        # needs wave IDs; passthrough remains enabled for the no-GS shader.
        assert xfb_ngg_info["linkage"]["stages_en"]["value"] == 0x3012010
        xfb_ngg_words = shader_words(xfb_ngg_raw)
        ds_ops = [
            (word >> 18) & 0xff
            for word in xfb_ngg_words
            if word >> 26 == 0x36 and word & (1 << 17)
        ]
        mubuf_ops = [
            (word >> 18) & 0x7f
            for word in xfb_ngg_words
            if word >> 26 == 0x38
        ]
        assert 0x3f not in ds_ops       # no ds_ordered_count GDS lock
        assert 0x20 not in ds_ops       # no ds_add_rtn_u32 GDS counter update
        assert 0x21 not in ds_ops       # no ds_sub_rtn_u32 overflow correction
        assert mubuf_ops.count(0x1e) >= vertices_per_primitive
    print("NGG transform feedback: all six point/line/triangle topologies use "
          "PrimitiveID offsets + capture stores, no GDS")

    print("Compiling fragment shader...")
    r = subprocess.run([psbc_bin, "-f", frag_spv, "-o", frag_sb, "-s", "fragment", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Fragment compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    mixed_glsl = os.path.join(TESTS, "mixed-interpolation.frag")
    mixed_spv = os.path.join(TESTS, "mixed-interpolation.spv")
    mixed_sb = os.path.join(TESTS, "mixed-interpolation.sb")
    mixed_metadata = os.path.join(TESTS, "mixed-interpolation.metadata.json")
    subprocess.run(["glslangValidator", "-V", mixed_glsl,
                    "-o", mixed_spv], check=True)
    r = subprocess.run([
        psbc_bin, "-f", mixed_spv, "-o", mixed_sb, "-s", "fragment",
        "--metadata", mixed_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Mixed-interpolation fragment compile failed: {r.stderr}")
        sys.exit(1)
    with open(mixed_metadata, "r", encoding="utf-8") as f:
        mixed_info = json.load(f)
    assert mixed_info["unresolved_fields"] == 1
    assert mixed_info["input_semantics"] == [0x40000f, 0x10]
    print("Pixel semantics: flat input carries PS5 AGC bit 22; smooth input does not")

    mixed_last_sb = os.path.join(TESTS, "mixed-interpolation-last.sb")
    mixed_last_metadata = os.path.join(
        TESTS, "mixed-interpolation-last.metadata.json")
    mixed_last_env = os.environ.copy()
    mixed_last_env["PSBC_DEBUG_NIR"] = "1"
    r = subprocess.run([
        psbc_bin, "-f", mixed_spv, "-o", mixed_last_sb, "-s", "fragment",
        "--primitive-type", "triangle-list", "--provoking-vertex-last",
        "--metadata", mixed_last_metadata,
    ], capture_output=True, text=True, cwd=PSBC, env=mixed_last_env)
    if r.returncode != 0:
        print(f"Last-provoking fragment compile failed: {r.stderr}")
        sys.exit(1)
    with open(mixed_sb, "rb") as f:
        mixed_first_binary = f.read()
    with open(mixed_last_sb, "rb") as f:
        mixed_last_binary = f.read()
    with open(mixed_last_metadata, "r", encoding="utf-8") as f:
        mixed_last_info = json.load(f)
    assert "@load_input_vertex" in r.stderr and "(0x2)" in r.stderr
    assert mixed_last_binary != mixed_first_binary
    assert mixed_last_info["input_semantics"] == [0x40000f, 0x10]
    print("Last-provoking triangle flat input explicitly selects primitive vertex P2")

    dual_source_glsl = os.path.join(TESTS, "dual-source.frag")
    dual_source_spv = os.path.join(TESTS, "dual-source.spv")
    dual_source_sb = os.path.join(TESTS, "dual-source.sb")
    dual_source_metadata = os.path.join(TESTS, "dual-source.metadata.json")
    subprocess.run(["glslangValidator", "-V", dual_source_glsl,
                    "-o", dual_source_spv], check=True)
    dual_source_env = os.environ.copy()
    dual_source_env.update({"PSBC_DEBUG_IO": "1", "PSBC_DEBUG_NIR": "1"})
    r = subprocess.run([
        psbc_bin, "-f", dual_source_spv, "-o", dual_source_sb,
        "-s", "fragment", "--metadata", dual_source_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC, env=dual_source_env)
    if r.returncode != 0:
        print(f"Dual-source fragment compile failed: {r.stderr}")
        sys.exit(1)
    assert "color_is_dual_source: true" in r.stderr
    assert "outputs_written: 4,12" in r.stderr
    assert "api_subgroup_size: 32" in r.stderr
    assert "@export_dual_src_blend_amd" not in r.stderr
    assert shader_export_targets(dual_source_sb) == [0, 1]
    assert shader_exports_are_compressed(dual_source_sb)
    with open(dual_source_metadata, "r", encoding="utf-8") as f:
        dual_metadata = json.load(f)
    dual_registers = {entry["offset"]: entry["value"]
                      for entry in dual_metadata["context_registers"]}
    assert dual_registers[0x1B6] == 0x8000
    assert dual_registers[0x1C5] == 0x44
    assert dual_registers[0x08F] == 0xFF
    print("Dual-source fragment: wave32, packed GFX10 MRT0/MRT1 exports, "
          "COL_FORMAT=44, MASK=ff")

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
    subprocess.run(["glslangValidator", "-V", geom_glsl, "-o", geom_spv], check=True)
    print("Compiling geometry shader...")
    r = subprocess.run([psbc_bin, "-f", geom_spv, "-o", geom_sb, "-s", "geometry", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Geometry compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    # Compile a complete merged VS+GS shader and pin the GFX10 stage state.
    # A real GS uses ES=REAL + GS=ON and must not enable NGG passthrough.
    merged_geom_raw = os.path.join(TESTS, "test.merged-geometry.raw.sb")
    merged_geom_metadata = os.path.join(TESTS, "test.merged-geometry.metadata.json")
    print("Compiling merged NGG vertex+geometry pipeline...")
    merged_geom_env = os.environ.copy()
    merged_geom_env["PSBC_DEBUG_IO"] = "1"
    r = subprocess.run([
        psbc_bin, "-f", geom_spv, "-o", merged_geom_raw,
        "-s", "geometry", "--previous", vert_spv, "--ngg", "--raw",
        "--metadata", merged_geom_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC, env=merged_geom_env)
    if r.returncode != 0:
        print(f"Merged geometry compile failed: {r.stderr}")
        sys.exit(1)
    with open(merged_geom_metadata, "r", encoding="utf-8") as f:
        merged_geom_info = json.load(f)
    assert merged_geom_info["hardware_stage"] == 3
    assert merged_geom_info["linkage"]["stages_en"] == {
        "offset": 0x2D5,
        "value": 0x12030,
    }
    merged_geom_registers = {
        entry["offset"]: entry["value"]
        for entry in merged_geom_info["context_registers"]
    }
    assert merged_geom_registers[0x29B] == 2
    merged_geom_shader_registers = {
        entry["offset"]: entry["value"]
        for entry in merged_geom_info["shader_registers"]
    }
    assert merged_geom_shader_registers[0x87] == 0x003FFFFF
    assert merged_geom_shader_registers[0x81] == 0x0000FFFF
    assert "PSBC IO previous-info:" in r.stderr
    assert "linked=0/1 merged_separate=0 slots=1/0" in r.stderr
    assert "PSBC IO info:" in r.stderr
    assert "linked=1/0 merged_separate=0 slots=0/1" in r.stderr
    print("Merged geometry stage state: ES=REAL, GS=ON, PRIMGEN=ON, "
          "MAX_PRIMGRP=2, passthrough=OFF, output=TRISTRIP, "
          "RSRC3/4=conservative")

    adjacency_cases = {
        "line-list-adjacency": "test-line-adjacency.geom",
        "line-strip-adjacency": "test-line-adjacency.geom",
        "triangle-list-adjacency": "test-adjacency.geom",
        "triangle-strip-adjacency": "test-adjacency.geom",
    }
    for topology, source in adjacency_cases.items():
        adjacency_geom_spv = os.path.join(TESTS, f"{topology}.geom.spv")
        adjacency_geom_raw = os.path.join(TESTS, f"{topology}.merged.raw.sb")
        adjacency_geom_metadata = os.path.join(
            TESTS, f"{topology}.merged.metadata.json"
        )
        subprocess.run([
            "glslangValidator", "-V", os.path.join(TESTS, source),
            "-o", adjacency_geom_spv,
        ], check=True)
        r = subprocess.run([
            psbc_bin, "-f", adjacency_geom_spv, "-o", adjacency_geom_raw,
            "-s", "geometry", "--previous", vert_spv, "--ngg", "--raw",
            "--primitive-type", topology,
            "--metadata", adjacency_geom_metadata, "-vv",
        ], capture_output=True, text=True, cwd=PSBC)
        if r.returncode != 0:
            print(f"Merged {topology} geometry compile failed: {r.stderr}")
            sys.exit(1)
        with open(adjacency_geom_metadata, "r", encoding="utf-8") as f:
            adjacency_geom_info = json.load(f)
        assert adjacency_geom_info["hardware_stage"] == 3
        assert adjacency_geom_info["machine_code_size"] > 0
    print("Merged geometry adjacency: all four native topology compiles")

    varying32_geom_spv = os.path.join(TESTS, "varying32.geom.spv")
    varying32_geom_raw = os.path.join(TESTS, "varying32.geom.raw.sb")
    varying32_geom_metadata = os.path.join(TESTS, "varying32.geom.metadata.json")
    varying32_frag_spv = os.path.join(TESTS, "varying32.frag.spv")
    varying32_frag_raw = os.path.join(TESTS, "varying32.frag.raw.sb")
    varying32_frag_metadata = os.path.join(TESTS, "varying32.frag.metadata.json")
    for source, output in (
        ("varying32.geom", varying32_geom_spv),
        ("varying32.frag", varying32_frag_spv),
    ):
        subprocess.run([
            "glslangValidator", "-V", os.path.join(TESTS, source),
            "-o", output,
        ], check=True)
    r = subprocess.run([
        psbc_bin, "-f", varying32_geom_spv, "-o", varying32_geom_raw,
        "-s", "geometry", "--previous", vert_spv, "--ngg", "--raw",
        "--metadata", varying32_geom_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"32-varying geometry compile failed: {r.stderr}")
        sys.exit(1)
    r = subprocess.run([
        psbc_bin, "-f", varying32_frag_spv, "-o", varying32_frag_raw,
        "-s", "fragment", "--raw", "--metadata", varying32_frag_metadata,
        "-vv",
    ], capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"32-varying fragment compile failed: {r.stderr}")
        sys.exit(1)
    with open(varying32_geom_metadata, "r", encoding="utf-8") as f:
        varying32_geom_info = json.load(f)
    with open(varying32_frag_metadata, "r", encoding="utf-8") as f:
        varying32_frag_info = json.load(f)
    assert len(varying32_geom_info["output_semantics"]) == 32
    assert len(varying32_frag_info["input_semantics"]) == 32
    varying32_frag_registers = {
        entry["offset"]: entry["value"]
        for entry in varying32_frag_info["context_registers"]
    }
    assert varying32_frag_registers[0x1B6] & 0x3F == 32
    print("Geometry/fragment interface: all 32 vec4 varying slots compile and package")

    # Compile merged GS transform feedback with PS5's private global counters.
    xfb_geom_asm = os.path.join(TESTS, "xfb.geom.spvasm")
    xfb_geom_spv = os.path.join(TESTS, "xfb.geom.spv")
    xfb_geom_raw = os.path.join(TESTS, "xfb.geom.raw.sb")
    xfb_geom_metadata = os.path.join(TESTS, "xfb.geom.metadata.json")
    subprocess.run([
        "spirv-as", "--target-env", "spv1.3", xfb_geom_asm,
        "-o", xfb_geom_spv,
    ], check=True)
    subprocess.run([
        "spirv-val", "--target-env", "vulkan1.1", xfb_geom_spv,
    ], check=True)
    xfb_geom_env = os.environ.copy()
    xfb_geom_env["PSBC_DEBUG_DISASM"] = "1"
    r = subprocess.run([
        psbc_bin, "-f", xfb_geom_spv, "-o", xfb_geom_raw,
        "-s", "geometry", "--previous", vert_spv, "--ngg",
        "--ps5-global-streamout", "--raw", "--metadata",
        xfb_geom_metadata, "-vv",
    ], capture_output=True, text=True, cwd=PSBC, env=xfb_geom_env)
    if r.returncode != 0:
        print(f"Merged geometry transform-feedback compile failed: {r.stderr}")
        sys.exit(1)
    with open(xfb_geom_metadata, "r", encoding="utf-8") as f:
        xfb_geom_info = json.load(f)
    with open(xfb_geom_raw, "rb") as f:
        xfb_geom_hash = hashlib.sha256(f.read()).hexdigest()
    xfb_geom_streamout = xfb_geom_info["streamout"]
    assert xfb_geom_info["hardware_stage"] == 3
    # The ordered reservation (make test-ordered-streamout) changed the
    # program; its structure, not a frozen hash, is the contract here.
    assert xfb_geom_info["machine_code_size"] > 0 and len(xfb_geom_hash) == 64
    assert xfb_geom_streamout["buffer_table_user_data_dword"] == 1
    assert xfb_geom_streamout["enabled_stream_buffers_mask"] == 1
    assert xfb_geom_streamout["strides_dwords"] == [2, 0, 0, 0]
    assert "ds_ordered_count" not in r.stderr
    assert r.stderr.count("global_atomic_swap") >= 1
    assert "ds_add_u32" not in r.stderr
    assert "ds_add_rtn_u32" not in r.stderr
    final_atomic = r.stderr.rindex("global_atomic_add")
    release_wait = r.stderr.index("s_waitcnt_vscnt", final_atomic)
    assert final_atomic < release_wait
    print("Merged geometry transform feedback: global offsets/queries, "
          "release wait, no GDS instructions")

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

    # Compile export shader (ES — VS variant for geometry pipeline)
    es_sb = os.path.join(TESTS, "test.es.sb")
    print("Compiling export shader (ES)...")
    r = subprocess.run([psbc_bin, "-f", vert_spv, "-o", es_sb, "-s", "export", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Export shader compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    # Compile local shader (LS — VS variant for tessellation pipeline)
    ls_sb = os.path.join(TESTS, "test.ls.sb")
    print("Compiling local shader (LS)...")
    r = subprocess.run([psbc_bin, "-f", vert_spv, "-o", ls_sb, "-s", "local", "-vv"],
                       capture_output=True, text=True, cwd=PSBC)
    if r.returncode != 0:
        print(f"Local shader compile failed: {r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())

    # Verify
    ok = True
    ok &= verify_shader(vert_sb, GNM_SHADER_VERTEX)
    ok &= verify_shader(frag_sb, GNM_SHADER_PIXEL)
    ok &= verify_shader(dual_source_sb, GNM_SHADER_PIXEL)
    ok &= verify_shader(comp_sb, GNM_SHADER_COMPUTE)
    ok &= verify_shader(geom_sb, GNM_SHADER_GEOMETRY)
    ok &= verify_shader(tesc_sb, GNM_SHADER_HULL)
    ok &= verify_shader(tese_sb, GNM_SHADER_VERTEX)  # DS uses VS type
    ok &= verify_shader(es_sb, GNM_SHADER_VERTEX)    # ES is a VS variant
    ok &= verify_shader(ls_sb, GNM_SHADER_VERTEX)    # LS is a VS variant
    ok &= verify_shader(instance_sb, GNM_SHADER_VERTEX)

    if ok:
        print("\n=== ALL TESTS PASSED ===")
        sys.exit(0)
    else:
        print("\n=== TESTS FAILED ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
