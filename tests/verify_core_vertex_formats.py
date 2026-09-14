#!/usr/bin/env python3
"""Compile every added Vulkan 1.0 core vertex-format family through PSBC."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "no diagnostics"
        raise RuntimeError(
            f"command failed with exit {error.returncode}: "
            f"{' '.join(command)}\n{detail}"
        ) from error


def compile_pair(psbc: Path, spirv: Path, output: Path, first: str,
                 second: str, size: int, alignment: int) -> None:
    run([
        str(psbc), "-f", str(spirv), "-o", str(output), "-s", "vertex",
        "--vertex-attribute", f"0:{first}:0:0:{size * 2}:{alignment}",
        "--vertex-attribute", f"1:{second}:0:{size}:{size * 2}:{alignment}",
    ])
    if not output.is_file() or not output.stat().st_size:
        raise RuntimeError(f"PSBC produced no shader binary for {first}/{second}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psbc", type=Path, default=ROOT / "opengnm-psbc")
    parser.add_argument("--glslang", default="glslangValidator")
    args = parser.parse_args()
    psbc = args.psbc.resolve()

    float_pairs = (
        ("r8_unorm", "r8_snorm", 1, 1),
        ("r8g8_unorm", "r8g8_snorm", 2, 1),
        ("r8g8b8a8_unorm", "r8g8b8a8_snorm", 4, 1),
        ("r16_unorm", "r16_snorm", 2, 2),
        ("r16g16_unorm", "r16g16_snorm", 4, 2),
        ("r16g16b16a16_unorm", "r16g16b16a16_snorm", 8, 2),
        ("r16_float", "r16_float", 2, 2),
        ("r16g16_float", "r16g16_float", 4, 2),
        ("r16g16b16a16_float", "r16g16b16a16_float", 8, 2),
    )
    integer_pairs = (
        ("r8_sint", "r8_uint", 1, 1),
        ("r8g8_sint", "r8g8_uint", 2, 1),
        ("r8g8b8a8_sint", "r8g8b8a8_uint", 4, 1),
        ("r16_sint", "r16_uint", 2, 2),
        ("r16g16_sint", "r16g16_uint", 4, 2),
        ("r16g16b16a16_sint", "r16g16b16a16_uint", 8, 2),
    )
    exercised = {item for pair in (*float_pairs, *integer_pairs)
                 for item in pair[:2]}
    # R8G8B8A8_UNORM predated this extension; the other 26 names are new.
    if len(exercised) != 27 or "r8g8b8a8_unorm" not in exercised:
        raise RuntimeError("vertex-format test matrix is incomplete")

    with tempfile.TemporaryDirectory(prefix="psbc-core-vertex-") as directory:
        temporary = Path(directory)
        float_spirv = temporary / "float.spv"
        integer_spirv = temporary / "integer.spv"
        run([args.glslang, "-V", str(ROOT / "tests/instance.vert"),
             "-o", str(float_spirv)])
        run([args.glslang, "-V", str(ROOT / "tests/integer-vertex.vert"),
             "-o", str(integer_spirv)])
        for index, (first, second, size, alignment) in enumerate(float_pairs):
            compile_pair(psbc, float_spirv, temporary / f"float-{index}.sb",
                         first, second, size, alignment)
        for index, (first, second, size, alignment) in enumerate(integer_pairs):
            compile_pair(psbc, integer_spirv, temporary / f"integer-{index}.sb",
                         first, second, size, alignment)

        unknown = subprocess.run([
            str(psbc), "-f", str(float_spirv), "-o",
            str(temporary / "unknown.sb"), "-s", "vertex",
            "--vertex-attribute", "0:r8_typo:0:0:1:1",
        ], cwd=ROOT, capture_output=True, text=True)
        if unknown.returncode == 0:
            raise RuntimeError("PSBC accepted an unknown vertex format")

    print("Core 8/16-bit vertex formats: 26/26 compile through NIR/ACO")


if __name__ == "__main__":
    main()
