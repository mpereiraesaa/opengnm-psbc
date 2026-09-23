#ifndef VK_PIPELINE_H
#define VK_PIPELINE_H

#include "nir/nir.h"
#include "vulkan/vulkan.h"

struct vk_pipeline_robustness_state {
   VkPipelineRobustnessBufferBehaviorEXT storage_buffers;
   VkPipelineRobustnessBufferBehaviorEXT uniform_buffers;
   VkPipelineRobustnessBufferBehaviorEXT vertex_inputs;
   VkPipelineRobustnessImageBehaviorEXT images;
   bool null_uniform_buffer_descriptor;
   bool null_storage_buffer_descriptor;
   bool _pad[2];
};

static inline void
vk_set_subgroup_size(nir_shader *shader,
                     uint32_t subgroup_size,
                     uint32_t min_subgroup_size,
                     uint32_t max_subgroup_size,
                     uint32_t spirv_version,
                     const void *info_pNext,
                     bool allow_varying,
                     bool require_full)
{
   /* The standalone compiler has no Vulkan pipeline-stage pNext chain. Honor
    * its fixed subgroup configuration before RADV selects and lowers waves. */
   (void)info_pNext;
   if (subgroup_size && !allow_varying && spirv_version < 0x10600) {
      shader->info.api_subgroup_size = subgroup_size;
      shader->info.max_subgroup_size = subgroup_size;
      if (require_full)
         shader->info.min_subgroup_size = subgroup_size;
   }
   if (max_subgroup_size) {
      if (shader->info.max_subgroup_size > max_subgroup_size)
         shader->info.max_subgroup_size = max_subgroup_size;
      if (shader->info.min_subgroup_size < min_subgroup_size)
         shader->info.min_subgroup_size = min_subgroup_size;
   }
}

#endif
