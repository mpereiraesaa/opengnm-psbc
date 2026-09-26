/* ACO's monotonic arena: growth stops at maximum_block_size (a large program
 * is held in many modest blocks), one oversized allocation still gets a block
 * that fits, and a failed block allocation stops the compile through
 * aco_arena_exhausted instead of writing through a null block. */
#include "aco_util.h"
#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>

static size_t largest_block, fail_at_or_above = SIZE_MAX;
static bool counting;
extern "C" void* malloc(size_t size)
{
   static void* (*real)(size_t) = (void* (*)(size_t))dlsym(RTLD_NEXT, "malloc");
   if (counting) {
      if (size >= fail_at_or_above)
         return nullptr;
      if (size > largest_block)
         largest_block = size;
   }
   return real(size);
}

static char last_label[96];
/* The test links only this header, so it supplies the accessor the library
 * normally defines. */
static void record_label(const char* label);
extern "C" void (*psbc_get_stage_hook(void))(const char*) { return record_label; }
static void record_label(const char* label)
{
   strncpy(last_label, label, sizeof(last_label) - 1);
}

static void on_abort(int)
{
   /* The exhaustion path reported the block size before aborting. */
   _exit(strncmp(last_label, "arena-exhausted 262144", 22) == 0 ? 0 : 1);
}

int main()
{
   counting = true;
   {
      aco::monotonic_buffer_resource arena;
      for (unsigned i = 0; i < 4096; ++i)
         assert(arena.allocate(1024, 8));
      assert(largest_block == 256 * 1024);
      largest_block = 0;
      assert(arena.allocate(600 * 1024, 8));
      assert(largest_block >= 600 * 1024 + 16 && largest_block <= 2 * 1024 * 1024);
   }
   puts("ACO arena: bounded growth and oversized single blocks");
   fflush(stdout);
   /* A refused 256 KiB block ends in aco_arena_exhausted (abort). */
   signal(SIGABRT, on_abort);
   fail_at_or_above = 256 * 1024;
   aco::monotonic_buffer_resource arena;
   for (unsigned i = 0; i < 4096; ++i)
      arena.allocate(1024, 8);
   return 1; /* unreachable: the refused block must abort */
}
