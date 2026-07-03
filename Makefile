# opengnm-psbc — SPIR-V to PS4 Shader Binary Compiler
#
# Build plan:
# 1. Generate Mesa sources (NIR opcodes, ACO opcodes, SPIRV info, etc.) via Python scripts
# 2. Compile all Mesa vendored sources (NIR, ACO, SPIRV, radv, util, amd/common)
# 3. Compile psbc-specific code (cmd/psbc/main.c, crc32_sb.c)
# 4. Link into the psbc binary
#
# Mesa 26.2.0 sources are vendored in src/. The Makefile will be populated
# with the full source list once the Mesa vendoring is complete.
#
# This is a placeholder — the full Makefile with all source files will be
# generated after the Mesa 26.2.0 vendoring is verified.

include config.mak

PSBC = opengnm-psbc

# AMD register JSON files (for codegen)
AMD_REGISTER_FILES = \
	src/amd/registers/gfx6.json \
	src/amd/registers/gfx7.json \
	src/amd/registers/gfx8.json \
	src/amd/registers/gfx81.json \
	src/amd/registers/gfx9.json

# Generated sources (via Python codegen from Mesa)
GENERATED_SRCS = \
	src/amd/common/amdgfxregs.h \
	src/amd/common/sid_tables.h \
	src/amd/compiler/aco_builder.h \
	src/amd/compiler/aco_opcodes.cpp \
	src/amd/compiler/aco_opcodes.h \
	src/compiler/nir/nir_builder_opcodes.h \
	src/compiler/nir/nir_constant_expressions.c \
	src/compiler/nir/nir_intrinsics.c \
	src/compiler/nir/nir_intrinsics.h \
	src/compiler/nir/nir_intrinsics_indices.h \
	src/compiler/nir/nir_opcodes.c \
	src/compiler/nir/nir_opcodes.h \
	src/compiler/nir/nir_opt_algebraic.c \
	src/compiler/spirv/spirv_info.c \
	src/compiler/spirv/vtn_gather_types.c \
	src/compiler/spirv/vtn_generator_ids.h

# psbc-specific sources
PSBC_SRCS = \
	cmd/psbc/main.c \
	cmd/psbc/crc32_sb.c

# TODO: Full Mesa source list will be added here after vendoring verification.
# The source list will include:
# - src/compiler/nir/*.c (NIR core + passes)
# - src/compiler/spirv/*.c (SPIRV-to-NIR)
# - src/amd/compiler/*.cpp (ACO)
# - src/amd/common/*.c (AMD common utilities)
# - src/amd/vulkan/radv_*.c (radv shader info/pipeline)
# - src/amd/vulkan/nir/*.c (radv NIR passes)
# - src/util/*.c (Mesa utilities)
# - src/mesa/main/*.c (Mesa main, minimal)

.PHONY: all clean install generated

all: $(PSBC)

# Generate Mesa codegen sources
generated: $(GENERATED_SRCS)

# TODO: Add generation rules for each generated source
# These use Mesa's Python scripts (src/compiler/nir/nir_opcodes.py, etc.)

# TODO: Compile rules and link rule
# $(PSBC): $(PSBC_OBJS) $(MESA_OBJS)
#	$(LD) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	find . -name '*.o' -delete
	rm -f $(PSBC)
	rm -f $(GENERATED_SRCS)

install: all
	install -d $(DESTDIR)$(BINDIR)
	install -m 755 $(PSBC) $(DESTDIR)$(BINDIR)/
