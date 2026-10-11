/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "vk_node_info.hh"

namespace blender::gpu::render_graph {
struct VKBuildAccelerationStructureData {
  VkAccelerationStructureBuildGeometryInfoKHR info = {};
  VkAccelerationStructureGeometryKHR geometry = {};
  VkAccelerationStructureBuildRangeInfoKHR range = {};
  VkBuffer destination = VK_NULL_HANDLE;
  VkBuffer scratch = VK_NULL_HANDLE;
  Vector<VkBuffer> inputs;
  Vector<VkBuffer> bottom_levels;
};

class VKBuildAccelerationStructureNode
    : public VKNodeInfo<VKNodeType::BUILD_ACCELERATION_STRUCTURE,
                        VKBuildAccelerationStructureData,
                        VKBuildAccelerationStructureData,
                        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        VKResourceType::BUFFER> {
 public:
  template<typename Node, typename Storage>
  static void set_node_data(Node &node, Storage &storage, const CreateInfo &create_info)
  {
    node.storage_index = storage.build_acceleration_structure.size();
    storage.build_acceleration_structure.append(create_info);
  }
  void build_links(VKResourceStateTracker &resources,
                   VKRenderGraphLinks &links,
                   const CreateInfo &info) override
  {
    for (VkBuffer input : info.inputs) {
      links.buffers.append({resources.get_buffer(input), VK_ACCESS_SHADER_READ_BIT});
    }
    for (VkBuffer blas : info.bottom_levels) {
      links.buffers.append(
          {resources.get_buffer(blas), VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR});
    }
    links.buffers.append({resources.get_buffer_and_increase_stamp(info.scratch),
                          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                              VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR});
    VkAccessFlags access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    if (info.info.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
      access |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    }
    links.buffers.append({resources.get_buffer_and_increase_stamp(info.destination), access});
  }
  void build_commands(VKCommandBufferInterface &commands,
                      Data &data,
                      Span<uint8_t>,
                      VKBoundPipelines &) override
  {
    data.info.pGeometries = &data.geometry;
    commands.build_acceleration_structure(data.info, data.range);
  }
};
}  // namespace blender::gpu::render_graph
