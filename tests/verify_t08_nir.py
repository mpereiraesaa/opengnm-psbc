#!/usr/bin/env python3
"""Check the three T08 compiler probes after PSBC's memory-model lowering."""
from pathlib import Path
import sys


def verify_barriers(shader: str, scope: str) -> None:
    release = f"memory_scope={scope}, mem_semantics=REL, mem_modes=ssbo|global"
    acquire = f"memory_scope={scope}, mem_semantics=ACQ, mem_modes=ssbo|global"
    assert shader.count(release) == 1, (scope, "release barrier")
    assert shader.count(acquire) == 1, (scope, "acquire barrier")
    assert shader.index("@store_ssbo") < shader.index(release) < shader.index(acquire)
    assert shader.index(acquire) < shader.index("@load_ssbo")
    assert "@store_ssbo" in shader and "access=coherent" in shader
    assert "@load_ssbo" in shader and "(access=coherent" in shader
    assert "MAKE_AVAILABLE" not in shader and "MAKE_VISIBLE" not in shader


def main() -> None:
    text = Path(sys.argv[1]).read_text()
    shaders = ["shader: MESA_SHADER_COMPUTE" + part
               for part in text.split("shader: MESA_SHADER_COMPUTE")[1:]]
    assert len(shaders) == 3, f"expected three compiled compute probes, got {len(shaders)}"
    assert "@store_global_amd" in shaders[0] and "access=writeonly" in shaders[0]
    verify_barriers(shaders[1], "QUEUE_FAMILY")
    verify_barriers(shaders[2], "DEVICE")
    print("T08 memory-model NIR lowering: coherent access and scoped barriers pass")


if __name__ == "__main__":
    main()
