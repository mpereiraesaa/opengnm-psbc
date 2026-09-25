"""Check the descriptor loads of tests/test_separate_sampler.c.

The compute shader's one set table holds S# at byte 0, T# at 16, the uniform
texel V# at 48 and the storage texel V# at 64; the fragment shader's holds S#
at 0 and T# at 16. Every record must be read from its own offset.
"""
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
shaders = log.split("shader: MESA_SHADER_")[1:]
assert [s.split("\n", 1)[0] for s in shaders] == ["COMPUTE", "FRAGMENT"], "unexpected stages"
load = re.compile(r"32x(\d+)\s+%\d+ = @load_global_amd \(.*?\) \(base=(\d+),")


def loads(text):
    return sorted((int(base), int(width)) for width, base in load.findall(text))


compute, fragment = shaders
# The S# is one four-dword load at 0; T#, uniform texel V# and storage texel
# V# are read together as sixteen dwords from 16 (16 + 64 = 80 = 64 + 16).
assert loads(compute) == [(0, 4), (16, 16)], loads(compute)
# Within the sixteen dwords read from 16: the T# is dwords 0-7, the uniform
# texel V# (byte 48) dwords 8-11 (.i-.l) and the storage texel V# (byte 64)
# dwords 12-15 (.m-.p).
wide = re.search(r"32x16\s+%(\d+) = @load_global_amd \(.*?\) \(base=16,", compute).group(1)
assert re.search(r"vec8 %{0}\.a, %{0}\.b, %{0}\.c, %{0}\.d, %{0}\.e, %{0}\.f, %{0}\.g, %{0}\.h"
                 .format(wide), compute), "T# must be the first eight dwords"
uniform_texel = re.search(r"%(\d+) = vec4 %{0}\.i, %{0}\.j, %{0}\.k, %{0}\.l".format(wide), compute)
storage_texel = re.search(r"%(\d+) = vec4 %{0}\.m, %{0}\.n, %{0}\.o, %{0}\.p".format(wide), compute)
assert uniform_texel and storage_texel, "texel V# records at bytes 48 and 64"
assert re.search(r"@load_buffer_amd \(%{}, .*uses-format-amd".format(uniform_texel.group(1)), compute)
assert re.search(r"@bindless_image_store \(%{}, .*image_dim=Buf".format(storage_texel.group(1)), compute)
assert "txl " in compute
assert loads(fragment) == [(0, 4), (16, 8)], loads(fragment)
assert " tex " in fragment or ")tex " in fragment
print("separate sampler descriptor loads: pass")
