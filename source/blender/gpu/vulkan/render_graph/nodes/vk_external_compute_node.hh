/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <functional>

#include "vk_node_info.hh"

namespace blender::gpu::render_graph {

/** External compute commands must leave images in GENERAL and not retain the command buffer.
 * The owner must keep SDK resources alive until GPU completion, not just graph recording. */
struct VKExternalComputeData {
  std::function<void(VkCommandBuffer)> record;
  Vector<VkImage> inputs;
  Vector<VkImage> outputs;
};

class VKExternalComputeNode : public VKNodeInfo<VKNodeType::EXTERNAL_COMPUTE,
                                                VKExternalComputeData,
                                                VKExternalComputeData,
                                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                                VKResourceType::IMAGE> {
 public:
  template<typename Node, typename Storage>
  static void set_node_data(Node &node, Storage &storage, const CreateInfo &info)
  {
    node.storage_index = storage.external_compute.size();
    storage.external_compute.append(info);
  }

  void build_links(VKResourceStateTracker &resources,
                   VKRenderGraphLinks &links,
                   const CreateInfo &info) override
  {
    /* SDKs can mix compute dispatches with transfer clears/copies internally. */
    for (VkImage image : info.inputs) {
      links.images.append({{resources.get_image(image), VK_ACCESS_MEMORY_READ_BIT},
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_ASPECT_COLOR_BIT});
    }
    for (VkImage image : info.outputs) {
      links.images.append(
          {{resources.get_image_and_increase_stamp(image),
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT},
           VK_IMAGE_LAYOUT_GENERAL,
           VK_IMAGE_ASPECT_COLOR_BIT});
    }
  }

  void build_commands(VKCommandBufferInterface &commands,
                      Data &data,
                      Span<uint8_t>,
                      VKBoundPipelines &bound) override
  {
    commands.external_compute(data.record);
    /* The SDK may bind pipelines, descriptors, and push constants. */
    bound = {};
  }
};
}  // namespace blender::gpu::render_graph
